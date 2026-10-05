import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeriesDecomp(nn.Module):
    """Moving-average trend with replicate padding; remainder is the seasonal part."""

    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel

    def forward(self, x):
        half = self.kernel // 2
        padded = F.pad(x.transpose(1, 2), (half, half), mode="replicate")
        trend = F.avg_pool1d(padded, self.kernel, stride=1).transpose(1, 2)
        return x - trend, trend


class AutoCorrelation(nn.Module):
    """Period-level dependency: FFT correlation, top-k delays, time-delay aggregation."""

    def __init__(self, d_model, n_heads, factor):
        super().__init__()
        self.h = n_heads
        self.factor = factor
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        S = keys.shape[1]
        q = self.q(queries).view(B, L, self.h, -1)
        k = self.k(keys).view(B, S, self.h, -1)
        v = self.v(values).view(B, S, self.h, -1)
        if L > S:  # decoder queries are longer than encoder keys: zero-pad keys and values
            k = F.pad(k, (0, 0, 0, 0, 0, L - S))
            v = F.pad(v, (0, 0, 0, 0, 0, L - S))
        else:
            k, v = k[:, :L], v[:, :L]
        qf = torch.fft.rfft(q.permute(0, 2, 3, 1).contiguous(), dim=-1)
        kf = torch.fft.rfft(k.permute(0, 2, 3, 1).contiguous(), dim=-1)
        corr = torch.fft.irfft(qf * torch.conj(kf), n=L, dim=-1)  # [B,H,d,L]
        corr = corr.mean(dim=(1, 2))  # series-wise score per delay, [B,L]
        top_k = int(self.factor * math.log(L))
        scores, delays = torch.topk(corr, top_k, dim=-1)
        weights = torch.softmax(scores, dim=-1)
        vals = v.permute(0, 2, 3, 1)  # [B,H,d,L]
        pos = torch.arange(L, device=q.device)
        src = (pos[None, None, :] + delays[:, :, None]) % L  # roll back by delay
        idx = src[:, :, None, None, :].expand(-1, -1, self.h, vals.shape[2], -1)
        shifted = vals[:, None].expand(-1, top_k, -1, -1, -1).gather(-1, idx)
        agg = (shifted * weights[:, :, None, None, None]).sum(1)  # [B,H,d,L]
        return self.out(agg.permute(0, 3, 1, 2).reshape(B, L, -1))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.c1 = nn.Conv1d(d_model, d_ff, 1)
        self.c2 = nn.Conv1d(d_ff, d_model, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.drop(F.gelu(self.c1(x.transpose(1, 2))))
        return self.drop(self.c2(y)).transpose(1, 2)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, kernel, factor, dropout):
        super().__init__()
        self.corr = AutoCorrelation(d_model, n_heads, factor)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dec1, self.dec2 = SeriesDecomp(kernel), SeriesDecomp(kernel)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.dec1(x + self.drop(self.corr(x, x, x)))
        x, _ = self.dec2(x + self.ff(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, kernel, factor, dropout, c_out):
        super().__init__()
        self.self_corr = AutoCorrelation(d_model, n_heads, factor)
        self.cross_corr = AutoCorrelation(d_model, n_heads, factor)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dec1, self.dec2, self.dec3 = SeriesDecomp(kernel), SeriesDecomp(kernel), SeriesDecomp(kernel)
        self.trend_proj = nn.Conv1d(d_model, c_out, 3, padding=1, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, enc):
        x, t1 = self.dec1(x + self.drop(self.self_corr(x, x, x)))
        x, t2 = self.dec2(x + self.drop(self.cross_corr(x, enc, enc)))
        x, t3 = self.dec3(x + self.ff(x))
        trend = self.trend_proj((t1 + t2 + t3).transpose(1, 2)).transpose(1, 2)
        return x, trend


class Autoformer(nn.Module):
    def __init__(self, c_in, c_mark, seq_len, label_len, pred_len, d_model=16, n_heads=2,
                 d_ff=32, e_layers=1, d_layers=1, kernel=25, factor=1, dropout=0.05, cov_head=False):
        super().__init__()
        # exogenous trend term: horizon-known covariates contribute directly to the trend stream
        self.cov_head = nn.Sequential(nn.Linear(c_mark, 32), nn.GELU(), nn.Linear(32, 1)) if cov_head and c_mark else None
        self.label_len, self.pred_len = label_len, pred_len
        self.decomp = SeriesDecomp(kernel)
        self.enc_embed = nn.Linear(c_in, d_model)
        self.dec_embed = nn.Linear(c_in, d_model)
        self.mark_embed = nn.Linear(c_mark, d_model, bias=False) if c_mark else None
        self.encoder = nn.ModuleList(EncoderLayer(d_model, n_heads, d_ff, kernel, factor, dropout) for _ in range(e_layers))
        self.decoder = nn.ModuleList(DecoderLayer(d_model, n_heads, d_ff, kernel, factor, dropout, 1) for _ in range(d_layers))
        self.enc_norm = nn.LayerNorm(d_model)
        self.dec_norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, 1)

    def embed(self, layer, x, mark):
        e = layer(x)
        return e + self.mark_embed(mark) if self.mark_embed is not None else e

    def forward(self, x_enc, x_mark_enc, x_mark_dec):
        B = x_enc.shape[0]
        mean = x_enc[:, :, :1].mean(1, keepdim=True)
        seasonal, trend = self.decomp(x_enc[:, :, :1])
        trend_init = torch.cat([trend[:, -self.label_len:], mean.expand(-1, self.pred_len, -1)], 1)
        seasonal_init = torch.cat([seasonal[:, -self.label_len:], torch.zeros(B, self.pred_len, 1, device=x_enc.device)], 1)
        enc = self.embed(self.enc_embed, x_enc, x_mark_enc)
        for layer in self.encoder:
            enc = layer(enc)
        enc = self.enc_norm(enc)
        dec = self.embed(self.dec_embed, seasonal_init, x_mark_dec)
        trend = trend_init + (self.cov_head(x_mark_dec) if self.cov_head is not None else 0)
        for layer in self.decoder:
            dec, residual_trend = layer(dec, enc)
            trend = trend + residual_trend
        out = self.proj(self.dec_norm(dec)) + trend
        return out[:, -self.pred_len:, 0]
