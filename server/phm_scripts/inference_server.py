"""
PHM 추론 서버 — FastAPI + ONNX Runtime
========================================
POST /predict  : 신호 윈도우 → 이상탐지 / 분류 결과 반환
GET  /health   : 로드된 모델 목록 반환
GET  /models/reload : 모델 캐시 재로드

Per-axis 모델 지원:
  PredictRequest.axis (int, optional) 를 지정하면 해당 축 전용 모델을 우선 로드합니다.
    axis=0 → ae_fd_ax0.onnx 우선, 없으면 ae_fd.onnx 로 폴백
    axis=1 → ae_fd_ax1.onnx 우선, 없으면 ae_fd.onnx 로 폴백
  axis 미지정 시 기존 동작(ae_fd.onnx / cnn1d_fd.onnx) 유지

환경변수:
  PHM_MODELS_ROOT : ONNX 모델 루트 경로 (기본 /opt/phm/models)

의존성 (자동 설치):
  fastapi uvicorn onnxruntime numpy pydantic
"""

import json
import os
import subprocess
import sys

# ── 패키지 자동 설치 ──────────────────────────────────────────────────────────
_REQUIRED = ["fastapi", "uvicorn", "onnxruntime", "numpy", "pydantic"]
_missing = []
for pkg in _REQUIRED:
    try:
        __import__(pkg.split("[")[0].replace("-", "_"))
    except ImportError:
        _missing.append(pkg)
if _missing:
    print(f"[setup] 패키지 설치 중: {_missing}", flush=True)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet"] + _missing,
        stdout=subprocess.DEVNULL,
    )
    print("[setup] 설치 완료", flush=True)

# ── 임포트 ────────────────────────────────────────────────────────────────────
from pathlib import Path
from typing import List, Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── 설정 ──────────────────────────────────────────────────────────────────────
MODELS_ROOT = Path(os.getenv("PHM_MODELS_ROOT", "/opt/phm/models"))

app = FastAPI(title="PHM Inference Server", version="2.0.0")

# ── 모델 캐시 {cache_key: (session, meta)} ────────────────────────────────────
# cache_key = "{sensor_type}" 또는 "{sensor_type}_ax{n}"
_sessions: dict = {}

# 기본(레거시) 후보 파일 — axis 미지정 시 또는 per-axis 모델 없을 때 폴백
_FALLBACK_CANDIDATES = {
    "accel":  ["ae_fd.onnx",     "cnn1d_fd.onnx"],
    "torque": ["ae_torque.onnx", "cnn1d_torque.onnx"],
}

# 최대 지원 축 수 (health 엔드포인트에서 스캔용)
_MAX_AXIS_SCAN = 8


def _per_axis_candidates(sensor_type: str, axis: int) -> List[str]:
    """per-axis 모델 후보 파일명 목록 (우선순위 높은 순)."""
    if sensor_type == "accel":
        return [
            f"ae_fd_ax{axis}.onnx",
            f"cnn1d_fd_ax{axis}.onnx",
        ]
    if sensor_type == "torque":
        return [
            f"ae_torque_ax{axis}.onnx",
            f"cnn1d_torque_ax{axis}.onnx",
        ]
    return []


def _cache_key(sensor_type: str, axis: Optional[int]) -> str:
    return f"{sensor_type}_ax{axis}" if axis is not None else sensor_type


def _load_model(sensor_type: str, axis: Optional[int] = None):
    """캐시에서 꺼내거나 디스크에서 로드합니다.

    axis 지정 시: per-axis 후보 → 폴백 순으로 탐색
    axis 미지정 시: 기존 후보 목록 탐색
    """
    key = _cache_key(sensor_type, axis)
    if key in _sessions:
        return _sessions[key]

    # 후보 파일 목록 구성
    if axis is not None:
        candidates = _per_axis_candidates(sensor_type, axis) + \
                     _FALLBACK_CANDIDATES.get(sensor_type, [])
    else:
        # axis 미지정: 전역 모델 우선, 없으면 per-axis 모델(Ax0부터) 폴백
        candidates = list(_FALLBACK_CANDIDATES.get(sensor_type, []))
        for _ax in range(_MAX_AXIS_SCAN):
            for _fname in _per_axis_candidates(sensor_type, _ax):
                if (MODELS_ROOT / _fname).exists():
                    candidates.append(_fname)
                    break   # 해당 축의 최우선 모델 1개만 추가

    for fname in candidates:
        model_path = MODELS_ROOT / fname
        if not model_path.exists():
            continue
        meta_path = MODELS_ROOT / (fname.replace(".onnx", "_meta.json"))
        try:
            sess = ort.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
            meta: dict = {}
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            _sessions[key] = (sess, meta)
            print(
                f"[inference] 모델 로드: {fname}  key={key}  kind={meta.get('kind','?')}",
                flush=True,
            )
            return sess, meta
        except Exception as e:
            print(f"[inference] 모델 로드 실패 {fname}: {e}", flush=True)

    return None, None


# ── Pydantic 모델 ─────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    sensor_type: str = "accel"     # "accel" | "torque"
    axis: Optional[int] = None     # 축 인덱스 (None = 레거시/전축 모델)
    window: List[float]            # flat float32 배열, 길이 = window_size × n_channels
    window_size: int = 1024
    n_channels: int = 3


class PredictResponse(BaseModel):
    model_type: str                # "AE-CNN1D" | "CNN1D"
    sensor_type: str
    axis: Optional[int] = None     # 실제 추론에 사용된 축 인덱스
    model_file: Optional[str] = None  # 실제 로드된 모델 파일명
    is_anomaly: bool
    anomaly_score: float           # 정규화된 이상 점수 (>1.0 이면 이상)
    threshold: float               # 항상 1.0 (정규화 기준)
    class_name: Optional[str] = None
    confidence: Optional[float] = None
    raw_mae: Optional[float] = None
    raw_threshold: Optional[float] = None


# ── 헬퍼 — Z-score per-sample 정규화 ─────────────────────────────────────────
def _zscore(arr: np.ndarray) -> np.ndarray:
    """arr shape (1, T, C) → 정규화된 동일 shape"""
    mean = arr.mean(axis=1, keepdims=True)      # (1, 1, C)
    std  = arr.std(axis=1, keepdims=True)
    std  = np.where(std < 1e-8, 1.0, std)
    return (arr - mean) / std


def _loaded_model_file(sensor_type: str, axis: Optional[int]) -> Optional[str]:
    """현재 캐시에 로드된 모델의 파일명을 반환합니다."""
    key = _cache_key(sensor_type, axis)
    if key not in _sessions:
        return None
    meta = _sessions[key][1]
    return meta.get("source_file")   # meta에 없으면 None


# ── 엔드포인트 ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    loaded = {k: v[1].get("kind", "?") for k, v in _sessions.items()}

    # 사용 가능한 모델 스캔
    available: dict = {}
    for st, fallbacks in _FALLBACK_CANDIDATES.items():
        files = []
        # per-axis 모델 스캔 (ae_fd_ax0.onnx ~ ae_fd_ax7.onnx)
        for ax in range(_MAX_AXIS_SCAN):
            for fname in _per_axis_candidates(st, ax):
                if (MODELS_ROOT / fname).exists():
                    files.append(fname)
                    break   # 해당 축의 최우선 모델만 1개 표시
        # 폴백(레거시) 모델
        for fname in fallbacks:
            if (MODELS_ROOT / fname).exists():
                files.append(fname)
        available[st] = files

    return {
        "status":           "ok",
        "loaded_models":    loaded,
        "available_models": available,
        "models_root":      str(MODELS_ROOT),
    }


@app.get("/models/reload")
def reload_models():
    _sessions.clear()
    return {"status": "reloaded", "message": "다음 /predict 호출 시 재로드됩니다."}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    sess, meta = _load_model(req.sensor_type, req.axis)
    if sess is None:
        # 사용 가능한 모델 목록 수집
        avail: List[str] = []
        if req.axis is not None:
            for fname in _per_axis_candidates(req.sensor_type, req.axis):
                if (MODELS_ROOT / fname).exists():
                    avail.append(fname)
        for fname in _FALLBACK_CANDIDATES.get(req.sensor_type, []):
            if (MODELS_ROOT / fname).exists():
                avail.append(fname)
        raise HTTPException(
            status_code=404,
            detail=(
                f"sensor_type='{req.sensor_type}' axis={req.axis} 모델 없음. "
                f"사용 가능: {avail if avail else '없음 — Airflow 학습을 먼저 실행하세요.'}"
            ),
        )

    # ── 모델 기대 채널 수 검증 ────────────────────────────────────────────────
    model_n_channels = meta.get("n_channels")
    if model_n_channels and req.n_channels != model_n_channels:
        # per-axis 요청인데 폴백 전역 모델(다채널)로 떨어진 경우:
        # 채널 수가 다른 더 적합한 모델을 재탐색하지 않고 에러 반환
        # (→ 해결책: Airflow per-axis 토크 학습 실행)
        raise HTTPException(
            status_code=400,
            detail=(
                f"채널 수 불일치: 모델={model_n_channels}ch, 요청={req.n_channels}ch. "
                f"모델 학습 채널: {meta.get('channels', '?')}. "
                + (
                    f"per-axis 모델(ae_torque_ax{req.axis}.onnx)이 없어 전역 모델로 폴백됨. "
                    f"Airflow train_torque 태스크를 실행하세요."
                    if req.axis is not None
                    else "Airflow 재학습 또는 CSV 재수집 후 retry."
                )
            ),
        )

    expected = req.window_size * req.n_channels
    if len(req.window) != expected:
        raise HTTPException(
            status_code=400,
            detail=f"window 길이 {len(req.window)} ≠ {req.window_size}×{req.n_channels}={expected}",
        )

    # 실제 사용된 모델 파일명 (meta에 source_file 없으면 axis로 추정)
    model_file = meta.get("source_file")

    try:
        # (1, T, C) float32
        raw_arr  = np.array(req.window, dtype=np.float32).reshape(1, req.window_size, req.n_channels)
        norm_arr = _zscore(raw_arr)
        if req.sensor_type == "accel":
            print(
                f"[accel] axis={req.axis} first8={raw_arr.reshape(-1)[:8].tolist()} "
                f"mean={raw_arr.mean():.6f} std={raw_arr.std():.6f}",
                flush=True
            )
        model_kind = meta.get("kind", "CNN1D")
        input_name = sess.get_inputs()[0].name

        # ── AE-CNN1D: 재구성 오차로 이상 탐지 ────────────────────────────────
        if model_kind == "AE-CNN1D":
            recon = sess.run(None, {input_name: norm_arr})[0]   # (1, T, C)
            mae   = float(np.abs(norm_arr - recon).mean())
            thr   = float(meta.get("threshold", 0.1))

            rms      = float(np.sqrt(np.mean(raw_arr.astype(np.float64) ** 2)))
            rms_thr  = float(meta.get("rms_thr",  float("inf")))
            rms_mean = float(meta.get("rms_mean", 0.0))
            # rms_std가 meta에 있으면 1-std 단위로 정규화, 없으면 기존 방식(thr-mean 기준)
            rms_std_m = meta.get("rms_std")
            if rms_thr < 1e30:
                denom    = float(rms_std_m) if rms_std_m else max(rms_thr - rms_mean, 1e-8)
                rms_norm = max(0.0, (rms - rms_mean) / denom)
            else:
                rms_norm = 0.0

            mae_norm     = mae / max(thr, 1e-8)
            # 가중합: MAE(형태 이상) + RMS(진폭 이상)
            # RMS 항: z-score 정규화로 MAE가 감지 못하는 진폭 변화(외력 등)를 보완.
            # 가중치 0.15 → 0.5: 외력처럼 진폭만 크게 바뀌는 경우도 score 1.0 초과 가능
            score_normed = mae_norm + 0.5 * rms_norm
            is_anomaly   = score_normed >= 1.0

            return PredictResponse(
                model_type="AE-CNN1D",
                sensor_type=req.sensor_type,
                axis=req.axis,
                model_file=model_file,
                is_anomaly=is_anomaly,
                anomaly_score=round(score_normed, 6),
                threshold=1.0,
                class_name="anomaly" if is_anomaly else "normal",
                raw_mae=round(mae, 6),
                raw_threshold=round(thr, 6),
            )

        # ── CNN1D: 분류 ──────────────────────────────────────────────────────
        logits      = sess.run(None, {input_name: norm_arr})[0]
        exp_l       = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs       = exp_l / exp_l.sum(axis=1, keepdims=True)
        pred_idx    = int(np.argmax(probs[0]))
        confidence  = float(probs[0][pred_idx])
        class_names = meta.get("class_names", ["normal", "fault"])
        pred_class  = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
        is_anomaly  = pred_class.lower() != "normal"
        anomaly_score = (1.0 - confidence) if not is_anomaly else confidence

        return PredictResponse(
            model_type="CNN1D",
            sensor_type=req.sensor_type,
            axis=req.axis,
            model_file=model_file,
            is_anomaly=is_anomaly,
            anomaly_score=round(anomaly_score, 6),
            threshold=0.5,
            class_name=pred_class,
            confidence=round(confidence, 6),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 실패: {type(e).__name__}: {e}")


# ── 엔트리포인트 ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print(f"[inference] 모델 루트: {MODELS_ROOT}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
