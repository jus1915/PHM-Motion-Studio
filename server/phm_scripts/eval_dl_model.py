"""
eval_dl_model.py — 학습된 AE/CLS ONNX 모델로 CSV 파일 배치 평가

사용법:
    python eval_dl_model.py --params eval_params.json

params JSON 구조:
{
    "model":            "models/accel_ax0_ae.onnx",
    "csv_files": [
        {"path": "...", "label": "normal"},
        {"path": "...", "label": "unlabeled"}
    ],
    "channels":         ["x","y","z"],
    "filter_op_column": "Op_Ax0",         // Pos 구간만 평가
    "window_size":      256,
    "stride":           128,
    "seg_agg":          "max",            // 구간 집계 방식: max(기본)|mean|p90
    "output_csv":       "eval_result.csv" // 선택: 창별 상세 결과 저장
}

출력:
    - 구간(segment) 단위 통계: 이동 1회 = 1구간
    - 윈도우(window) 단위 통계
    - stdout JSON 결과
    - output_csv 지정 시 구간·창별 상세 CSV
"""

import sys
import subprocess


def _ensure():
    for pkg in ("numpy", "onnxruntime"):
        try:
            __import__("onnxruntime" if pkg == "onnxruntime" else pkg)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg],
                                  stdout=subprocess.DEVNULL)

_ensure()

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import onnxruntime as ort


# ── 전처리 ───────────────────────────────────────────────────────────────────

def _resolve_channels(headers: List[str], channels: List[str]) -> List[str]:
    import re as _re
    resolved = []
    for ch in channels:
        if ch in headers:
            resolved.append(ch)
        else:
            pat = _re.compile(r"^Ax\d+_" + _re.escape(ch) + r"$", _re.IGNORECASE)
            matches = sorted([h for h in headers if pat.match(h)],
                             key=lambda h: int(_re.search(r"\d+", h).group()))
            resolved.extend(matches) if matches else resolved.append(ch)
    return resolved


def _read_segments(
    path: str,
    channels: List[str],
    filter_op_column: Optional[str],
) -> List[np.ndarray]:
    """CSV → 연속 Pos 구간별 ndarray 목록."""
    segments: List[np.ndarray] = []
    current: List[List[float]] = []

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        actual_ch = _resolve_channels(list(reader.fieldnames), channels)
        fl = [x.strip().lower() for x in reader.fieldnames]

        if filter_op_column:
            foc = filter_op_column.strip().lower()
            op_cols = [reader.fieldnames[i] for i, f in enumerate(fl) if f == foc]
        else:
            op_cols = [reader.fieldnames[i] for i, f in enumerate(fl)
                       if f.startswith("op_ax") or f == "op"]

        def flush():
            if current:
                segments.append(np.array(current, dtype=np.float32))
                current.clear()

        for row in reader:
            is_idle = False
            if op_cols:
                idle_flags = [(row.get(c) or "").strip().lower() == "idle" for c in op_cols]
                is_idle = idle_flags[0] if filter_op_column else all(idle_flags)
            if is_idle:
                flush()
                continue
            try:
                vals = [float(row[c]) for c in actual_ch]
            except Exception:
                flush()
                continue
            if any(math.isnan(v) or math.isinf(v) for v in vals):
                flush()
                continue
            current.append(vals)
        flush()

    return segments


def _zscore(arr: np.ndarray) -> np.ndarray:
    """(1, T, C) → per-sample z-score 정규화 (레거시 fallback)."""
    mean = arr.mean(axis=1, keepdims=True)
    std  = arr.std(axis=1, keepdims=True)
    std  = np.where(std < 1e-8, 1.0, std)
    return (arr - mean) / std


def _global_norm(arr: np.ndarray, mean: list, std: list) -> np.ndarray:
    """(1, T, C) → 전역 통계 기반 정규화 (학습 시 저장된 norm_mean/norm_std 사용).

    per-sample z-score 와 달리 절대 진폭·에너지 정보를 보존하여
    AE 재구성 오차가 고장 신호에서 실제로 커지도록 합니다.
    """
    m = np.array(mean, dtype=np.float32).reshape(1, 1, -1)
    s = np.array(std,  dtype=np.float32).reshape(1, 1, -1)
    s = np.where(s < 1e-8, 1.0, s)
    return (arr - m) / s


def _add_fft_channels(window: np.ndarray) -> np.ndarray:
    """(T, C) → (T, 2C): 채널별 FFT 크기 스펙트럼을 추가 채널로 붙입니다.

    train_dl_model.py 의 동일 함수와 동일 로직 — 학습/평가 전처리 일관성 유지.
    """
    T, C = window.shape
    fft_mag = np.abs(np.fft.rfft(window, axis=0))   # (T//2+1, C)
    fft_len = fft_mag.shape[0]
    x_old   = np.linspace(0.0, 1.0, fft_len)
    x_new   = np.linspace(0.0, 1.0, T)
    fft_resized = np.stack(
        [np.interp(x_new, x_old, fft_mag[:, c]) for c in range(C)],
        axis=1,
    ).astype(np.float32)
    return np.concatenate([window, fft_resized], axis=1)   # (T, 2C)


# ── 모델 판별 ─────────────────────────────────────────────────────────────────

def _detect_is_ae(sess: ort.InferenceSession) -> bool:
    """출력 shape으로 AE/CLS 자동 판별. 3차원 → AE, 2차원 → CLS."""
    return len(sess.get_outputs()[0].shape) == 3


# ── 추론 — 구간별 윈도우 점수 목록 반환 ──────────────────────────────────────

def run_eval(
    sess: ort.InferenceSession,
    meta: dict,
    segments: List[np.ndarray],
    window_size: int,
    stride: int,
) -> List[List[float]]:
    """구간별 윈도우 점수 목록을 반환합니다.

    Returns:
        seg_scores[i] = i번째 구간의 윈도우 점수 목록 (창이 없으면 빈 리스트)
    """
    input_name  = sess.get_inputs()[0].name
    model_ws    = int(meta.get("window_size", window_size))
    thr         = float(meta.get("threshold", 0.1))
    is_ae       = _detect_is_ae(sess)
    norm_mean_m = meta.get("norm_mean")   # 전역 정규화: 학습 시 저장된 채널별 mean
    norm_std_m  = meta.get("norm_std")    # 전역 정규화: 학습 시 저장된 채널별 std
    use_global  = (norm_mean_m is not None) and (norm_std_m is not None)
    use_fft     = bool(meta.get("add_fft_channels", False))

    seg_scores: List[List[float]] = []

    for seg in segments:
        n = seg.shape[0]
        win_scores: List[float] = []

        for start in range(0, n - window_size + 1, stride):
            window = seg[start: start + window_size]

            if model_ws != window_size:
                if window.shape[0] < model_ws:
                    continue
                window = window[-model_ws:]

            if use_fft:
                window = _add_fft_channels(window)          # (T, C) → (T, 2C)
            arr  = window[np.newaxis].astype(np.float32)   # (1, T, C or 2C)
            norm = _global_norm(arr, norm_mean_m, norm_std_m) if use_global else _zscore(arr)

            if is_ae:
                recon = sess.run(None, {input_name: norm})[0]
                mae   = float(np.abs(norm - recon).mean())
                score = mae / max(thr, 1e-8)
            else:
                logits = sess.run(None, {input_name: norm})[0].reshape(1, -1)
                exp_l  = np.exp(logits - logits.max(axis=1, keepdims=True))
                probs  = (exp_l / exp_l.sum(axis=1, keepdims=True))[0]
                pred   = int(np.argmax(probs))
                names  = meta.get("class_names", ["normal", "fault"])
                is_anom = (names[pred].lower() != "normal") if pred < len(names) else True
                score  = float(probs[pred]) if is_anom else 1.0 - float(probs[pred])

            win_scores.append(score)

        seg_scores.append(win_scores)

    return seg_scores


# ── 구간 점수 집계 ────────────────────────────────────────────────────────────

def _aggregate_seg(win_scores: List[float], method: str) -> float:
    """윈도우 점수 목록 → 구간 대표 점수.

    method: "max"(기본) | "mean" | "p90"
      max  — 구간 내 가장 나쁜 창 기준 (이상 탐지에 민감)
      mean — 구간 전체 평균 (안정적)
      p90  — 상위 10% 수준 (이상값 1~2개 노이즈 필터)
    """
    if not win_scores:
        return 0.0
    arr = np.array(win_scores)
    if method == "mean":
        return float(arr.mean())
    if method == "p90":
        return float(np.percentile(arr, 90))
    return float(arr.max())   # default: max


# ── 통계 ─────────────────────────────────────────────────────────────────────

def _stats(values: List[float], threshold: float) -> dict:
    """점수 목록 → 통계 dict."""
    if not values:
        return {"count": 0}
    arr  = np.array(values)
    anom = int((arr >= threshold).sum())
    return {
        "count":          len(values),
        "score_mean":     round(float(arr.mean()), 4),
        "score_max":      round(float(arr.max()),  4),
        "score_min":      round(float(arr.min()),  4),
        "score_p95":      round(float(np.percentile(arr, 95)), 4),
        "anomaly_count":  anom,
        "anomaly_ratio":  round(anom / len(values), 4),
        "anomaly_threshold": threshold,
    }


def summarize(
    label: str,
    seg_scores_list: List[List[float]],
    anomaly_threshold: float,
    seg_agg: str,
) -> dict:
    """구간 단위 + 윈도우 단위 통계를 함께 반환합니다."""
    # 구간 대표 점수
    seg_repr   = [_aggregate_seg(ws, seg_agg) for ws in seg_scores_list if ws]
    # 전체 윈도우 점수 (flat)
    all_wins   = [sc for ws in seg_scores_list for sc in ws]

    return {
        "label":        label,
        "seg_agg":      seg_agg,
        "segments":     _stats(seg_repr,  anomaly_threshold),
        "windows":      _stats(all_wins,  anomaly_threshold),
    }


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AE/CLS 모델 구간별 배치 평가")
    parser.add_argument("--params", required=True, help="평가 params JSON 경로")
    args = parser.parse_args()

    with open(args.params, encoding="utf-8-sig") as f:
        p = json.load(f)

    model_path      = Path(p["model"])
    meta_path       = model_path.with_name(model_path.stem + "_meta.json")
    channels        = p.get("channels", ["x", "y", "z"])
    foc             = p.get("filter_op_column", None)
    window_size     = int(p.get("window_size", 256))
    stride          = int(p.get("stride", 128))
    seg_agg         = p.get("seg_agg", "max")        # max | mean | p90
    output_csv      = p.get("output_csv", None)
    thr_norm        = float(p.get("anomaly_threshold", 1.0))
    normal_label    = p.get("normal_label", "unlabeled")   # 자동 임계값 보정 기준 레이블
    auto_thr_pct    = int(p.get("auto_threshold_percentile", 0))   # 0=비활성, 99 권장
    sweep_thrs      = sorted(float(t) for t in p.get("threshold_sweep", []))

    if not model_path.exists():
        print(f"[eval] 모델 없음: {model_path}", file=sys.stderr)
        sys.exit(1)

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    meta: dict = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    is_ae      = _detect_is_ae(sess)
    kind       = "AE" if is_ae else "CLS"
    label_kind = "이상탐지(AE)" if is_ae else "결함진단(CLS)"

    # meta session/kind 불일치 경고
    kind_meta   = meta.get("kind", "")
    session_meta = meta.get("session", "")
    meta_says_ae = (session_meta == "AE") or ("AE" in kind_meta.upper())
    if meta_says_ae != is_ae:
        print(f"[eval] ⚠ meta session={session_meta!r}({kind_meta}) 와 실제 출력 shape 불일치 → {kind} 로 추론합니다.", file=sys.stderr)

    fft_active = bool(meta.get("add_fft_channels", False))
    print(f"[eval] 모델: {model_path.name}  {label_kind}  "
          f"window_size={meta.get('window_size', window_size)}  "
          f"seg_agg={seg_agg}  threshold={thr_norm}"
          f"{'  FFT채널=ON' if fft_active else ''}", file=sys.stderr)

    csv_entries = p.get("csv_files", [])
    all_results: list  = []
    detail_rows: list  = []

    for entry in csv_entries:
        csv_path = entry["path"]
        label    = entry.get("label", Path(csv_path).parent.name)

        print(f"[eval] 읽는 중: {Path(csv_path).name}  label={label}", file=sys.stderr)
        segments = _read_segments(csv_path, channels, foc)

        # 윈도우 없는 구간(짧은 구간) 제외 후 통계
        valid_segs   = [s for s in segments if s.shape[0] >= window_size]
        skipped_segs = len(segments) - len(valid_segs)
        total_rows   = sum(s.shape[0] for s in valid_segs)
        print(f"[eval]   → 전체 {len(segments)}구간 중 유효 {len(valid_segs)}구간 "
              f"({skipped_segs}개 스킵), {total_rows:,}행", file=sys.stderr)

        seg_scores_list = run_eval(sess, meta, valid_segs, window_size, stride)
        total_wins = sum(len(ws) for ws in seg_scores_list)
        print(f"[eval]   → {total_wins}개 윈도우 추론 완료", file=sys.stderr)

        summary = summarize(label, seg_scores_list, thr_norm, seg_agg)
        all_results.append(summary)

        # 상세 CSV용 행 구성
        for seg_idx, win_scores in enumerate(seg_scores_list):
            seg_score = _aggregate_seg(win_scores, seg_agg)
            for win_idx, sc in enumerate(win_scores):
                detail_rows.append({
                    "label":       label,
                    "seg_idx":     seg_idx,
                    "seg_score":   round(seg_score, 6),
                    "seg_anomaly": seg_score >= thr_norm,
                    "win_idx":     win_idx,
                    "win_score":   round(sc, 6),
                    "win_anomaly": sc >= thr_norm,
                })

    # ── 구간 대표 점수 수집 (sweep·auto-threshold 공용) ──────────────────────
    seg_scores_per_label: dict = {}
    seen: set = set()
    for row in detail_rows:
        key = (row["label"], row["seg_idx"])
        if key not in seen:
            seen.add(key)
            seg_scores_per_label.setdefault(row["label"], []).append(float(row["seg_score"]))

    # ── 자동 임계값 보정 ──────────────────────────────────────────────────────
    auto_threshold: Optional[float] = None
    if auto_thr_pct > 0 and normal_label in seg_scores_per_label:
        normal_arr   = np.array(seg_scores_per_label[normal_label])
        auto_threshold = float(np.percentile(normal_arr, auto_thr_pct))
        print(f"[eval] 자동 임계값 (normal={normal_label!r}, p{auto_thr_pct}): "
              f"{auto_threshold:.4f}", file=sys.stderr)

    # ── 임계값 스윕 ───────────────────────────────────────────────────────────
    sweep_rows: list = []
    if sweep_thrs:
        print(f"[eval] 임계값 스윕: {sweep_thrs}", file=sys.stderr)
        for thr in sweep_thrs:
            for lbl in sorted(seg_scores_per_label):
                arr  = np.array(seg_scores_per_label[lbl])
                anom = int((arr >= thr).sum())
                sweep_rows.append({
                    "threshold":    round(thr, 4),
                    "label":        lbl,
                    "count":        len(arr),
                    "anomaly_count": anom,
                    "anomaly_ratio": round(anom / len(arr), 4) if len(arr) else 0.0,
                })

    # ── stdout JSON ───────────────────────────────────────────────────────────
    output = {
        "model":   model_path.name,
        "session": kind,
        "seg_agg": seg_agg,
        "results": all_results,
    }
    if auto_threshold is not None:
        output["auto_threshold"]            = round(auto_threshold, 4)
        output["auto_threshold_percentile"] = auto_thr_pct
        output["auto_threshold_normal_label"] = normal_label
    if sweep_rows:
        output["threshold_sweep"] = sweep_rows
    print(json.dumps(output, ensure_ascii=False, indent=2))

    # ── CSV 저장 ──────────────────────────────────────────────────────────────
    if output_csv and detail_rows:
        fields = ["label", "seg_idx", "seg_score", "seg_anomaly",
                  "win_idx", "win_score", "win_anomaly"]
        with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(detail_rows)
        print(f"[eval] 구간·창별 결과 저장: {output_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
