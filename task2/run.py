"""Task 2 pipeline: ablation (external data on/off, several seeds) on a chronological
validation split, then a refit on the full history and the 168-step leaderboard forecast.

Usage: python run.py [--epochs 4] [--seeds 0 1 2] [--data DIR] [--out results]
"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from autoformer import Autoformer

SEQ, LABEL, PRED = 336, 48, 168
VAL_BLOCKS = 8  # the last 8 weeks of history form the validation region
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load(data_dir):
    tr = pd.read_csv(Path(data_dir) / "student_train.csv")
    ex = pd.read_csv(Path(data_dir) / "optional_external_data.csv").sort_values("time_idx")
    y = tr["value"].to_numpy(np.float32)
    feats = ex.drop(columns="time_idx").to_numpy(np.float32)
    feats[:, 3] = np.log1p(feats[:, 3])  # feature_D is heavy-tailed too
    return y, feats


def gbm_baseline(y, feats, upto, n_total):
    """Gradient-boosted regression from horizon-known covariates (plus hour and season
    phases and rolling covariate means) to the target. Fit on the first `upto` points only."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    df = pd.DataFrame(feats[:n_total])
    idx = np.arange(1, n_total + 1)
    df["hs"], df["hc"] = np.sin(2 * np.pi * idx / 24), np.cos(2 * np.pi * idx / 24)
    df["ys"], df["yc"] = np.sin(2 * np.pi * idx / 8766), np.cos(2 * np.pi * idx / 8766)
    for w in (6, 24, 72):
        for c in range(feats.shape[1]):
            df[f"{c}_m{w}"] = df[c].rolling(w, min_periods=1).mean()
    X = df.to_numpy(np.float32)
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=0)
    model.fit(X[:upto], y[:upto])
    return model.predict(X).astype(np.float32), model.n_iter_ * 31


def rich_marks(feats):
    """Season phase plus rolling covariate means: the slow weather state a single hour's reading hides."""
    n = len(feats)
    idx = np.arange(1, n + 1)
    season = np.stack([np.sin(2 * np.pi * idx / 8766), np.cos(2 * np.pi * idx / 8766)], 1)
    df = pd.DataFrame(feats)
    rolls = [df.rolling(w, min_periods=1).mean().to_numpy() for w in (6, 24, 72)]
    return np.concatenate([feats, season.astype(np.float32), *rolls], 1).astype(np.float32)


def hour_marks(n):
    ph = 2 * np.pi * np.arange(1, n + 1) / 24
    return np.stack([np.sin(ph), np.cos(ph)], 1).astype(np.float32)


def standardise(arr, upto):
    mu, sd = arr[:upto].mean(0), arr[:upto].std(0) + 1e-6
    return (arr - mu) / sd, mu, sd


def windows(y, marks, origins):
    """One sample per forecast origin t: encoder = y[t-SEQ:t], target = y[t:t+PRED]."""
    xe = np.stack([y[t - SEQ:t] for t in origins])[:, :, None]
    me = np.stack([marks[t - SEQ:t] for t in origins])
    md = np.stack([marks[t - LABEL:t + PRED] for t in origins])
    tgt = np.stack([y[t:t + PRED] for t in origins])
    return [torch.tensor(a) for a in (xe, me, md, tgt)]


def train(model, data, epochs, seed, lr=2e-3, batch=64, after_epoch=None, loss="mse"):
    criterion = nn.functional.mse_loss if loss == "mse" else nn.functional.huber_loss
    torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * ((len(data[0]) + batch - 1) // batch))
    n = len(data[0])
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xe, me, md, tgt = (d[idx].to(DEVICE) for d in data)
            step_loss = criterion(model(xe, me, md), tgt)
            opt.zero_grad(); step_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        if after_epoch:
            after_epoch()


@torch.no_grad()
def predict(model, data):
    model.eval()
    return torch.cat([model(*(d[i:i + 256].to(DEVICE) for d in data[:3])).cpu() for i in range(0, len(data[0]), 256)])


def metrics(pred_raw, true_raw):
    err = pred_raw - true_raw
    return dict(RMSE=float(np.sqrt((err ** 2).mean())), MAE=float(np.abs(err).mean()),
                sMAPE=float(100 * np.mean(2 * np.abs(err) / (np.abs(pred_raw) + np.abs(true_raw) + 1e-6))))


def build(c_mark, args):
    return Autoformer(1, c_mark, SEQ, LABEL, PRED, d_model=args.d_model, n_heads=args.heads,
                      d_ff=2 * args.d_model, kernel=args.kernel, e_layers=args.layers, cov_head=args.cov_head).to(DEVICE)


def blend(args, out):
    """Average two seed-ensembles (e.g. raw-loss and log-loss) and pick the weight by validation RMSE."""
    dirs = [Path(d) for d in args.dirs]
    val_true = np.load(dirs[0] / "val_true.npy")
    ens = [np.mean([np.load(p) for p in sorted(d.glob("val_pred_ext1_seed*.npy"))], 0) for d in dirs]
    rows = []
    for w in np.arange(0, 1.01, 0.1):
        rows.append(dict(weight_first=round(float(w), 1), **metrics(w * ens[0] + (1 - w) * ens[1], val_true)))
    df = pd.DataFrame(rows); print(df.round(2).to_string(index=False))
    w = float(df.loc[df.RMSE.idxmin(), "weight_first"])
    subs = [json.load(open(d / "submission.json")) for d in dirs]
    fcs = [np.array(s["predictions"].split(","), float) for s in subs]
    fc = np.clip(w * fcs[0] + (1 - w) * fcs[1], 0, None)
    P = sum(s["P"] for s in subs) if 0 < w < 1 else subs[0 if w == 1 else 1]["P"]
    E = sum(s["E"] for s in subs) if 0 < w < 1 else subs[0 if w == 1 else 1]["E"]
    sub = dict(config=f"blend {w:.1f}*{dirs[0]} + {1-w:.1f}*{dirs[1]}", P=P, E=E,
               validation_RMSE=round(float(df.RMSE.min()), 2), predictions=",".join(f"{v:.4f}" for v in fc))
    json.dump(sub, open(out / "submission.json", "w"), indent=1)
    df.to_csv(out / "blend_weights.csv", index=False)
    print("BLEND weight", w, "validation RMSE", sub["validation_RMSE"], "P =", P, "E =", E)


def analog(args, out):
    """Covariate-analog forecast: each horizon hour takes the mean target of its k nearest
    history hours in standardised covariate space. No fitted parameters; a level reference
    the Autoformer blend is validated against."""
    y, feats = load(args.data)
    N, PRED_ = len(y), PRED
    def forecast(origins, upto):
        z, _, _ = standardise(feats, upto)
        hist, yh = z[:upto], y[:upto]
        rows = []
        for t in origins:
            d = ((hist[None, :, :] - z[t:t + PRED_, None, :]) ** 2).sum(-1)  # [PRED, upto]
            idx = np.argpartition(d, args.k, axis=1)[:, :args.k]
            rows.append(yh[idx].mean(1))
        return np.stack(rows)
    val_start = N - VAL_BLOCKS * PRED_
    val_origins = [val_start + j * PRED_ for j in range(VAL_BLOCKS)]
    val_true = np.stack([y[t:t + PRED_] for t in val_origins])
    val_pred = forecast(val_origins, val_start)
    print("analog k=%d validation" % args.k, metrics(val_pred, val_true))
    np.save(out / "val_true.npy", val_true); np.save(out / "val_pred_ext1_seed0.npy", val_pred)
    fc = forecast([N], N)[0]
    json.dump(dict(config=f"analog k={args.k}", P=0, E=0, n_models=1,
                   predictions=",".join(f"{v:.4f}" for v in fc)), open(out / "submission.json", "w"), indent=1)
    print("analog forecast mean %.1f min %.1f max %.1f" % (fc.mean(), fc.min(), fc.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Data"); ap.add_argument("--out", default="results")
    ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--d_model", type=int, default=16); ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--kernel", type=int, default=25); ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--final_seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--phase", choices=["ablate", "summarize", "final", "blend", "analog"], default="ablate")
    ap.add_argument("--k", type=int, default=100, help="analog: number of nearest covariate hours averaged")
    ap.add_argument("--dirs", nargs=2, help="blend: two result dirs (e.g. a raw-loss run and a log-loss run)")
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--target", choices=["log", "raw"], default="log")
    ap.add_argument("--hour", action="store_true", help="add sin/cos of the 24-step cycle as covariates")
    ap.add_argument("--cov_head", action="store_true", help="add a covariate-to-trend head in the decoder")
    ap.add_argument("--rich", action="store_true",
                    help="add season phase and 6/24/72-step rolling means of the covariates as extra marks")
    ap.add_argument("--loss", choices=["mse", "huber"], default="mse")
    ap.add_argument("--base", choices=["none", "gbm"], default="none",
                    help="gbm: covariate regression baseline; the Autoformer then forecasts the residual")
    ap.add_argument("--floor", type=float, default=0.0, help="lowest value a final forecast may take")
    ap.add_argument("--ext", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=True)
    if args.phase == "blend":
        return blend(args, out)
    if args.phase == "analog":
        return analog(args, out)
    y_raw, feats = load(args.data)
    y_log = np.log1p(y_raw) if args.target == "log" else y_raw
    inv = (lambda z: np.expm1(z)) if args.target == "log" else (lambda z: z)
    base_params = 0
    if args.base == "gbm":  # the network forecasts what the covariate regression cannot explain
        assert args.target == "raw", "--base gbm needs --target raw"
    if args.rich:
        feats = rich_marks(feats)
    if args.hour:
        feats = np.concatenate([feats, hour_marks(len(feats))], 1)
    N = len(y_log)
    val_start = N - VAL_BLOCKS * PRED
    base_val = np.zeros(len(feats), np.float32)
    if args.base == "gbm":
        base_val, base_params = gbm_baseline(y_raw, feats, val_start, len(feats))
        y_log = y_raw - base_val[:N]
    y_std, y_mu, y_sd = standardise(y_log, val_start)
    f_std, _, _ = standardise(feats, val_start)
    to_raw = lambda z: inv(z * y_sd + y_mu)

    val_origins = [val_start + j * PRED for j in range(VAL_BLOCKS)]
    base_blocks = np.stack([base_val[t:t + PRED] for t in val_origins])
    train_origins = list(range(SEQ, val_start - PRED + 1, args.stride))
    val_true = to_raw(np.stack([y_std[t:t + PRED] for t in val_origins])) + base_blocks
    rows = []
    if args.base == "gbm":
        rows.append(dict(config="baseline: gbm covariates", seed=-1, params=base_params, epochs=0, seconds=0,
                         **metrics(base_blocks, val_true)))
        np.save(out / "val_pred_gbm.npy", base_blocks)
        print(rows[-1], flush=True)
    # baselines give the reader a floor: a forecast must beat the last-value and recent-mean rules
    last = np.repeat(to_raw(y_std[[t - 1 for t in val_origins]])[:, None], PRED, 1)
    recent = np.repeat(to_raw(np.stack([y_std[t - 24:t].mean() for t in val_origins]))[:, None], PRED, 1)
    rows.append(dict(config="baseline: last value", seed=-1, params=0, epochs=0, seconds=0, **metrics(last, val_true)))
    rows.append(dict(config="baseline: mean of last 24", seed=-1, params=0, epochs=0, seconds=0, **metrics(recent, val_true)))

    if args.phase == "ablate":
      for use_ext in [bool(e) for e in args.ext]:
        marks = f_std if use_ext else (f_std[:, -2:] if args.hour else np.zeros((N, 0), np.float32))
        c_mark = marks.shape[1]
        tr_data, va_data = windows(y_std, marks, train_origins), windows(y_std, marks, val_origins)
        for seed in args.seeds:
            torch.manual_seed(seed)
            model = build(c_mark, args)
            curve = []
            t0 = time.time()
            train(model, tr_data, args.epochs, seed, loss=args.loss,
                  after_epoch=lambda: curve.append(metrics(to_raw(predict(model, va_data).numpy()) + base_blocks, val_true)["RMSE"]))
            secs = time.time() - t0
            pred = to_raw(predict(model, va_data).numpy()) + base_blocks
            params = sum(p.numel() for p in model.parameters() if p.requires_grad) + base_params
            m = metrics(pred, val_true)
            per_block = [metrics(pred[j:j + 1], val_true[j:j + 1])["RMSE"] for j in range(VAL_BLOCKS)]
            rows.append(dict(config="with external" if use_ext else "no external", seed=seed, params=params,
                             epochs=args.epochs, seconds=round(secs, 1), **m, block_rmse=per_block,
                             rmse_by_epoch=[round(c, 2) for c in curve]))
            np.save(out / f"val_pred_ext{int(use_ext)}_seed{seed}.npy", pred)
            json.dump(rows[-1], open(out / f"run_ext{int(use_ext)}_seed{seed}.json", "w"))
            print(rows[-1], flush=True)
      np.save(out / "val_true.npy", val_true)
      return
    rows += [json.load(open(p)) for p in sorted(out.glob("run_ext*_seed*.json"))]
    df = pd.DataFrame(rows); df.to_csv(out / "ablation.csv", index=False)
    summary = df[df.seed >= 0].groupby("config")[["RMSE", "MAE", "sMAPE"]].agg(["mean", "std"]).round(3)
    print(summary); summary.to_csv(out / "ablation_summary.csv")
    for _, r in df[df.seed >= 0].iterrows():
        print(r.config, "seed", r.seed, "RMSE by epoch:", r.rmse_by_epoch)
    runs = df[df.seed >= 0]
    best_cfg = runs.groupby("config")["RMSE"].mean().idxmin()
    curves = runs[runs.config == best_cfg].rmse_by_epoch.tolist()
    width = min(len(c) for c in curves)  # seeds may have been run for different epoch counts
    mean_curve = np.array([c[:width] for c in curves], dtype=float).mean(0)
    best_epochs = int(mean_curve.argmin()) + 1  # epoch count that minimises the seed-mean validation RMSE
    json.dump(dict(config=best_cfg, epochs=best_epochs, mean_curve=[round(c, 2) for c in mean_curve],
                   rmse_at_best=round(float(mean_curve.min()), 2)), open(out / "best.json", "w"), indent=1)
    print("seed-mean RMSE by epoch:", [round(float(c), 2) for c in mean_curve], "-> best epochs =", best_epochs)

    # final: the configuration with the lower mean validation RMSE, refit on the full history
    best = df[df.seed >= 0].groupby("config")["RMSE"].mean().idxmin()
    if args.phase == "summarize":
        return
    if args.epochs == 0:  # take the epoch count chosen by summarize
        args.epochs = json.load(open(out / "best.json"))["epochs"]
    use_ext = best == "with external"
    base_full = np.zeros(len(feats), np.float32)
    if args.base == "gbm":
        base_full, base_params = gbm_baseline(y_raw, feats, N, len(feats))
        y_log = y_raw - base_full[:N]
    y_std_full, y_mu, y_sd = standardise(y_log, N); f_full, _, _ = standardise(feats, N)
    to_raw = lambda z: inv(z * y_sd + y_mu)
    marks = f_full if use_ext else (f_full[:, -2:] if args.hour else np.zeros((len(feats), 0), np.float32))
    full_origins = list(range(SEQ, N - PRED + 1, args.stride))
    tr_data = windows(y_std_full, marks, full_origins)
    x_enc = torch.tensor(y_std_full[N - SEQ:N][None, :, None])
    m_enc = torch.tensor(marks[N - SEQ:N][None]); m_dec = torch.tensor(marks[N - LABEL:N + PRED][None])
    forecasts, total_params, total_epochs = [], 0, 0
    for seed in args.final_seeds:
        torch.manual_seed(seed); model = build(marks.shape[1], args)
        train(model, tr_data, args.epochs, seed, loss=args.loss)
        model.eval()
        with torch.no_grad():
            forecasts.append(to_raw(model(x_enc.to(DEVICE), m_enc.to(DEVICE), m_dec.to(DEVICE)).cpu().numpy()[0]) + base_full[N:N + PRED])
        total_params += sum(p.numel() for p in model.parameters() if p.requires_grad); total_epochs += args.epochs
    total_params += base_params
    fc = np.clip(np.mean(forecasts, 0), args.floor, None)
    sub = dict(config=best, P=total_params, E=total_epochs, n_models=len(args.final_seeds),
               predictions=",".join(f"{v:.4f}" for v in fc))
    json.dump(sub, open(out / "submission.json", "w"), indent=1)
    pd.DataFrame(dict(time_idx=np.arange(N + 1, N + PRED + 1), value=fc)).to_csv(out / "forecast_168.csv", index=False)
    print("FINAL", best, "P =", total_params, "E =", total_epochs)


if __name__ == "__main__":
    main()
