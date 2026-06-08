"""
train_channel_active_ae.py - PHM 채널별 활성 윈도우 AE 학습 (서버용)

Per-channel statistical feature AutoEncoder.
  • 채널별 독립적으로 13개 통계 피처를 계산
  • active_score 기준으로 active/inactive/uncertain 윈도우 분류
  • active 윈도우만으로 채널별 Linear AE 학습
  • ONNX 모델 출력 + ch_ae_meta.json 사이드카 저장

사용법:
    python train_channel_active_ae.py --params params.json

params JSON 예:
{
    "data_dir":              "/opt/phm/data",
    "csv_file":              null,   // 단일 CSV 직접 지정 시 (null이면 data_dir 스캔)
    "output":                "/opt/phm/models/profile/ch_ae_meta.json",
    "channels":              ["x", "y", "z", "Ax0_Trq(%)", "Ax1_Trq(%)"],
    "window_size":           128,
    "stride":                 64,
    "chunksize":            2000,
    "active_top_ratio":     0.20,
    "inactive_bottom_ratio":0.50,
    "latent_dim":              4,
    "epochs":                100,
    "batch_size":             64,
    "lr":                  0.001,
    "patience":               15,
    "min_active_windows":     50,
    "val_ratio":             0.2,
    "seed":                   42,
    "save_plots":           true
}
"""

import argparse
import json
import os
import random
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Precision loss occurred in moment calculation")

# ── 패키지 자동 설치 ───────────────────────────────────────────────────────────
def _ensure_packages():
    required = [
        ("numpy",   "numpy"),
        ("pandas",  "pandas"),
        ("scipy",   "scipy"),
        ("sklearn", "scikit-learn"),
        ("onnx",    "onnx"),
        ("joblib",  "joblib"),
    ]
    missing = [pip for imp, pip in required if not _try_import(imp)]
    if missing:
        import subprocess
        print(f"[setup] 패키지 설치: {missing}", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet"] + missing,
                              stdout=subprocess.DEVNULL)

    try:
        import torch  # noqa
    except ImportError:
        import subprocess
        print("[setup] torch CPU 버전 설치 중...", flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "torch",
             "--index-url", "https://download.pytorch.org/whl/cpu"],
            stdout=subprocess.DEVNULL,
        )

def _try_import(name):
    try:
        __import__(name); return True
    except ImportError:
        return False

_ensure_packages()

# ── 임포트 ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ── 상수 ───────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "min", "max", "rms", "std", "var",
    "p2p", "mean_abs", "peak_abs",
    "skewness", "kurtosis",
    "crest", "shape", "impulse",
]

SCORE_COLS = ("p2p", "rms", "std")

# 이름에서 파일명에 사용할 수 없는 문자를 치환
def _safe_name(name: str) -> str:
    return (str(name)
            .replace("/", "_").replace("\\", "_")
            .replace(":", "_").replace("%", "pct")
            .replace("(", "").replace(")", "")
            .replace(" ", "_"))


# ── 재현성 ─────────────────────────────────────────────────────────────────────
def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── CSV 파일 탐색 ──────────────────────────────────────────────────────────────
def _find_normal_csvs(data_dir: str):
    """data_dir 하위에서 'normal' 서브디렉터리의 CSV 파일을 찾습니다."""
    data_dir = Path(data_dir)
    # 1순위: normal/ 서브디렉터리
    normal_csvs = sorted(data_dir.rglob("normal/*.csv")) + sorted(data_dir.rglob("*/normal/*.csv"))
    seen = set()
    result = []
    for p in normal_csvs:
        if p not in seen:
            seen.add(p)
            result.append(p)
    if result:
        print(f"[Data] 'normal' 클래스 CSV {len(result)}개 발견", flush=True)
        return result

    # 폴백: 비정상 클래스 디렉터리를 제외한 모든 CSV
    skip_dirs = {"overload", "looseness", "fault", "error", "abnormal", "anomaly",
                 "bearing_fault", "imbalance"}
    result = []
    for csv in sorted(data_dir.rglob("*.csv")):
        parts = {p.lower() for p in csv.parts}
        if not (parts & skip_dirs):
            result.append(csv)
    print(f"[Data] 폴백: 비정상 클래스 제외 후 CSV {len(result)}개", flush=True)
    return result


# ── 단일 채널 윈도우 생성 (청크 처리) ────────────────────────────────────────
def _iter_channel_windows(csv_path, channel_col, window_size, stride, chunksize=2000):
    """단일 채널에서 슬라이딩 윈도우를 yield합니다. (메모리 효율적)"""
    buffer = pd.DataFrame()
    try:
        for chunk in pd.read_csv(csv_path, usecols=[channel_col],
                                 chunksize=chunksize, on_bad_lines="skip"):
            if len(buffer) > 0:
                chunk = pd.concat([buffer, chunk], axis=0, ignore_index=True)
            n = len(chunk)
            max_start = n - window_size
            for start in range(0, max_start + 1, stride):
                yield chunk.iloc[start:start + window_size][channel_col].to_numpy(dtype=np.float32)
            keep_start = max(0, n - window_size + stride)
            buffer = chunk.iloc[keep_start:].copy()
    except Exception as e:
        print(f"  [warn] {csv_path.name} 채널 {channel_col} 읽기 실패: {e}", flush=True)


# ── 단일 채널 피처 계산 ─────────────────────────────────────────────────────
def _calc_channel_features(window: np.ndarray) -> dict:
    """13개 통계 피처를 계산합니다."""
    eps = 1e-8
    x = window.astype(np.float32)

    min_v    = float(np.min(x))
    max_v    = float(np.max(x))
    rms_v    = float(np.sqrt(np.mean(x ** 2)))
    std_v    = float(np.std(x))
    var_v    = float(np.var(x))
    p2p_v    = max_v - min_v
    mabs_v   = float(np.mean(np.abs(x)))
    pabs_v   = float(np.max(np.abs(x)))

    if std_v < eps:
        skew_v = 0.0
        kurt_v = 3.0
    else:
        skew_v = float(skew(x, bias=False))
        kurt_v = float(kurtosis(x, fisher=False, bias=False))

    crest_v   = pabs_v / (rms_v + eps)
    shape_v   = rms_v  / (mabs_v + eps)
    impulse_v = pabs_v / (mabs_v + eps)

    return {
        "min": min_v, "max": max_v, "rms": rms_v, "std": std_v, "var": var_v,
        "p2p": p2p_v, "mean_abs": mabs_v, "peak_abs": pabs_v,
        "skewness": skew_v, "kurtosis": kurt_v,
        "crest": crest_v, "shape": shape_v, "impulse": impulse_v,
    }


# ── 피처 DataFrame 빌드 (다중 파일) ───────────────────────────────────────────
def _build_feature_df(csv_paths, channel_cols, window_size, stride, chunksize):
    rows = []
    for ch in channel_cols:
        print(f"  [Feature] 채널: {ch}", flush=True)
        widx = 0
        for csv_path in csv_paths:
            for window in _iter_channel_windows(csv_path, ch, window_size, stride, chunksize):
                row = _calc_channel_features(window)
                row["channel"] = ch
                row["window_index"] = widx
                rows.append(row)
                widx += 1
        print(f"    → 총 {widx}개 윈도우", flush=True)

    if not rows:
        raise RuntimeError("피처 행이 하나도 없습니다. data_dir / channels 설정을 확인하세요.")
    return pd.DataFrame(rows)


# ── Robust Z-score ─────────────────────────────────────────────────────────────
def _robust_zscore(s: pd.Series, eps=1e-8):
    med = s.median()
    mad = (s - med).abs().median()
    return 0.6745 * (s - med) / (mad + eps)


# ── 채널별 active/inactive 라벨링 ─────────────────────────────────────────────
def _add_active_state(df, score_cols, active_top_ratio, inactive_bottom_ratio):
    result = df.copy()
    result["active_score"]       = np.nan
    result["active_threshold"]   = np.nan
    result["inactive_threshold"] = np.nan
    result["state"]              = "uncertain"

    for channel, idx in result.groupby("channel").groups.items():
        sub = result.loc[idx].copy()

        score = np.zeros(len(sub), dtype=np.float32)
        for col in score_cols:
            z = _robust_zscore(sub[col]).clip(lower=0)
            score += z.to_numpy(dtype=np.float32)
        score /= len(score_cols)

        sub["active_score"] = score
        active_thr   = sub["active_score"].quantile(1.0 - active_top_ratio)
        inactive_thr = sub["active_score"].quantile(inactive_bottom_ratio)

        result.loc[sub.index, "active_score"]       = score
        result.loc[sub.index, "active_threshold"]   = active_thr
        result.loc[sub.index, "inactive_threshold"] = inactive_thr

        result.loc[sub.index[score >= active_thr],   "state"] = "active"
        result.loc[sub.index[score <= inactive_thr],  "state"] = "inactive"

    return result


# ── Dataset ────────────────────────────────────────────────────────────────────
class _FeatureDataset(Dataset):
    def __init__(self, x: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.x[idx]


# ── 모델 ───────────────────────────────────────────────────────────────────────
class FeatureAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 4):
        super().__init__()
        h1 = max(8, input_dim * 2)
        h2 = max(4, input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2),       nn.ReLU(),
            nn.Linear(h2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2), nn.ReLU(),
            nn.Linear(h2, h1),         nn.ReLU(),
            nn.Linear(h1, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ── AE 학습 ────────────────────────────────────────────────────────────────────
def _train_ae(model, train_loader, val_loader, epochs, lr, patience, device):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val, best_state, wait = np.inf, None, 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        tr_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward(); optimizer.step()
            tr_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                val_losses.append(criterion(model(xb.to(device)), yb.to(device)).item())

        tr_loss  = float(np.mean(tr_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d} | train={tr_loss:.6f} | val={val_loss:.6f}", flush=True)

        if wait >= patience:
            print(f"  Early stopping at epoch {epoch}", flush=True)
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, history


# ── ONNX 내보내기 ──────────────────────────────────────────────────────────────
def _export_onnx(model: nn.Module, input_dim: int, onnx_path: str):
    import onnx  # noqa — 설치 확인용
    model.cpu().eval()
    dummy = torch.zeros(1, input_dim)
    try:
        torch.onnx.export(
            model, dummy, onnx_path,
            input_names=["features"],
            output_names=["reconstruction"],
            opset_version=11,
            dynamic_axes={"features": {0: "batch"}, "reconstruction": {0: "batch"}},
        )
        print(f"  ONNX 저장: {onnx_path}", flush=True)
    except Exception as e:
        print(f"  [warn] ONNX 내보내기 실패: {e} → .pt만 저장됩니다", flush=True)
        onnx_path = None
    return onnx_path


# ── 재구성 오차 계산 ───────────────────────────────────────────────────────────
def _recon_errors(model: nn.Module, X: np.ndarray, device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_t  = torch.tensor(X, dtype=torch.float32).to(device)
        recon = model(x_t).cpu().numpy()
    return np.mean((X - recon) ** 2, axis=1)


# ── 플롯 (서버: 파일로만 저장) ────────────────────────────────────────────────
def _save_plot_loss(channel, history, plot_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(history["train_loss"], label="train")
        plt.plot(history["val_loss"],   label="val")
        plt.title(f"{channel} AE Loss"); plt.xlabel("epoch"); plt.ylabel("MSE")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"{_safe_name(channel)}_loss.png"), dpi=120)
        plt.close()
    except Exception:
        pass


# ── 채널별 학습 메인 ──────────────────────────────────────────────────────────
def _train_channel_models(df, params, output_dir, device, save_plots):
    active_df = df[df["state"] == "active"].copy()
    active_df = active_df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS)

    print("\n[Active 윈도우 수 by 채널]")
    print(active_df.groupby("channel").size().to_string(), flush=True)

    plot_dir = os.path.join(output_dir, "plots")
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)

    channel_results = {}

    for channel in sorted(active_df["channel"].unique()):
        print(f"\n{'='*60}", flush=True)
        print(f"[AE 학습] 채널: {channel}", flush=True)

        ch_df = active_df[active_df["channel"] == channel].copy()
        n = len(ch_df)
        min_windows = params.get("min_active_windows", 50)
        if n < min_windows:
            print(f"  → 건너뜀: active 윈도우 부족 ({n} < {min_windows})", flush=True)
            continue

        X = ch_df[FEATURE_COLS].to_numpy(dtype=np.float32)

        # ── StandardScaler ───────────────────────────────────────────
        scaler = StandardScaler()
        train_idx, val_idx = train_test_split(
            np.arange(len(X)), test_size=params.get("val_ratio", 0.2),
            random_state=params.get("seed", 42), shuffle=True,
        )
        X_train = scaler.fit_transform(X[train_idx]).astype(np.float32)
        X_val   = scaler.transform(X[val_idx]).astype(np.float32)

        # ── AE 학습 ──────────────────────────────────────────────────
        input_dim  = X_train.shape[1]
        latent_dim = params.get("latent_dim", 4)
        batch_size = params.get("batch_size", 64)

        train_loader = DataLoader(_FeatureDataset(X_train), batch_size=batch_size,
                                  shuffle=True, drop_last=False)
        val_loader   = DataLoader(_FeatureDataset(X_val),   batch_size=batch_size,
                                  shuffle=False, drop_last=False)

        model = FeatureAutoEncoder(input_dim, latent_dim).to(device)
        model, history = _train_ae(
            model, train_loader, val_loader,
            epochs   = params.get("epochs",   100),
            lr       = params.get("lr",      1e-3),
            patience = params.get("patience",  15),
            device   = device,
        )

        # ── 재구성 오차 임계값 ────────────────────────────────────────
        train_errors = _recon_errors(model, X_train, device)
        val_errors   = _recon_errors(model, X_val,   device)

        thr95 = float(np.quantile(train_errors, 0.95))
        thr99 = float(np.quantile(train_errors, 0.99))

        print(f"  train_error: mean={np.mean(train_errors):.6f}  "
              f"95%={thr95:.6f}  99%={thr99:.6f}", flush=True)
        print(f"  val_over_95: {np.mean(val_errors > thr95):.1%}", flush=True)

        # ── 파일명 및 경로 ────────────────────────────────────────────
        ch_safe    = _safe_name(channel)
        onnx_fname = f"ch_ae_{ch_safe}_model.onnx"
        onnx_path  = os.path.join(output_dir, onnx_fname)

        saved_onnx = _export_onnx(model, input_dim, onnx_path)

        # ── active_score 분포 → 임계값 메타 ─────────────────────────
        ch_all = df[df["channel"] == channel].copy()
        p2p_s = ch_all["p2p"]
        rms_s = ch_all["rms"]
        std_s = ch_all["std"]

        sub_active = df[(df["channel"] == channel) & (df["state"] != "")].copy()
        active_thr_val   = float(sub_active["active_threshold"].iloc[0])
        inactive_thr_val = float(sub_active["inactive_threshold"].iloc[0])

        channel_results[channel] = {
            "model_file":          onnx_fname if saved_onnx else None,
            "active_threshold":    active_thr_val,
            "inactive_threshold":  inactive_thr_val,
            "score_p2p_median":    float(p2p_s.median()),
            "score_p2p_mad":       float((p2p_s - p2p_s.median()).abs().median()),
            "score_rms_median":    float(rms_s.median()),
            "score_rms_mad":       float((rms_s - rms_s.median()).abs().median()),
            "score_std_median":    float(std_s.median()),
            "score_std_mad":       float((std_s - std_s.median()).abs().median()),
            "scaler_mean":         scaler.mean_.tolist(),
            "scaler_std":          scaler.scale_.tolist(),
            "recon_error_thr_95":  thr95,
            "recon_error_thr_99":  thr99,
            "n_train":             len(X_train),
            "n_val":               len(X_val),
            "n_active":            int((df["channel"] == channel).sum() and
                                       (df[df["channel"] == channel]["state"] == "active").sum()),
            "n_inactive":          int((df[df["channel"] == channel]["state"] == "inactive").sum()),
            "n_uncertain":         int((df[df["channel"] == channel]["state"] == "uncertain").sum()),
            "best_val_loss":       float(min(history["val_loss"])),
        }

        if save_plots:
            _save_plot_loss(channel, history, plot_dir)

    return channel_results


# ── ch_ae_meta.json 저장 ──────────────────────────────────────────────────────
def _save_meta(output_path, channel_results, params):
    meta = {
        "kind":                  "ChannelActiveAE",
        "window_size":           params.get("window_size", 128),
        "stride":                params.get("stride",       64),
        "feature_cols":          FEATURE_COLS,
        "score_cols":            list(SCORE_COLS),
        "active_top_ratio":      params.get("active_top_ratio",      0.20),
        "inactive_bottom_ratio": params.get("inactive_bottom_ratio", 0.50),
        "channels":              channel_results,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[Meta] ch_ae_meta.json 저장: {output_path}", flush=True)


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True, help="params JSON 파일 경로")
    args = parser.parse_args()

    params = json.loads(Path(args.params).read_text(encoding="utf-8"))

    data_dir    = params["data_dir"]
    output_path = params["output"]  # ch_ae_meta.json 경로
    output_dir  = str(Path(output_path).parent)
    channels    = params.get("channels", ["x", "y", "z"])
    window_size = params.get("window_size",  128)
    stride      = params.get("stride",        64)
    chunksize   = params.get("chunksize",    2000)
    seed        = params.get("seed",           42)
    save_plots  = params.get("save_plots",   True)

    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Config] device={device}  window_size={window_size}  stride={stride}", flush=True)
    print(f"[Config] channels={channels}", flush=True)
    print(f"[Config] output_dir={output_dir}", flush=True)

    os.makedirs(output_dir, exist_ok=True)

    # ── 1. CSV 파일 탐색 ─────────────────────────────────────────────────────
    csv_file_single = params.get("csv_file")  # 단일 파일 직접 지정 (재현성 검증용)
    if csv_file_single:
        csv_path_single = Path(csv_file_single)
        if not csv_path_single.exists():
            raise FileNotFoundError(f"csv_file 이 존재하지 않습니다: {csv_file_single}")
        csv_paths = [csv_path_single]
        print(f"[Data] 단일 CSV 지정 모드 (재현성 검증): {csv_path_single}", flush=True)
    else:
        csv_paths = _find_normal_csvs(data_dir)
        if not csv_paths:
            raise RuntimeError(f"CSV 파일을 찾을 수 없습니다: {data_dir}")
        print(f"[Data] 학습 CSV 목록 (최대 5개 표시):", flush=True)
        for p in csv_paths[:5]:
            print(f"  {p}", flush=True)
        if len(csv_paths) > 5:
            print(f"  ... 총 {len(csv_paths)}개", flush=True)

    # ── 2. 피처 DataFrame 빌드 ───────────────────────────────────────────────
    print("\n[Step 1] 채널별 윈도우 피처 계산 중...", flush=True)
    df = _build_feature_df(csv_paths, channels, window_size, stride, chunksize)
    print(f"  DataFrame shape: {df.shape}", flush=True)

    # ── 3. Active/inactive 라벨링 ────────────────────────────────────────────
    print("\n[Step 2] Active/inactive 라벨링 중...", flush=True)
    active_top_ratio      = params.get("active_top_ratio",      0.20)
    inactive_bottom_ratio = params.get("inactive_bottom_ratio", 0.50)
    df = _add_active_state(df, SCORE_COLS, active_top_ratio, inactive_bottom_ratio)

    print("\n[State 분포 by 채널]")
    print(df.groupby(["channel", "state"]).size().to_string(), flush=True)

    # ── 4. 채널별 AE 학습 ────────────────────────────────────────────────────
    print("\n[Step 3] 채널별 AE 학습 중...", flush=True)
    channel_results = _train_channel_models(df, params, output_dir, device, save_plots)

    if not channel_results:
        print("[warn] 학습된 채널이 없습니다. min_active_windows 설정을 확인하세요.", flush=True)
        return

    # ── 5. 메타 저장 ─────────────────────────────────────────────────────────
    _save_meta(output_path, channel_results, params)

    print("\n[완료] 채널별 AE 학습 완료:")
    for ch, info in channel_results.items():
        print(f"  {ch}: model={info['model_file']}  thr95={info['recon_error_thr_95']:.6f}  "
              f"active={info['n_active']}  val_loss={info['best_val_loss']:.6f}", flush=True)


if __name__ == "__main__":
    main()
