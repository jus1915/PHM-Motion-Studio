"""
PHM 추론 서버 — FastAPI + ONNX Runtime
========================================
POST /predict  : 신호 윈도우 → 이상탐지 / 분류 결과 반환
GET  /health   : 로드된 모델 목록 반환
GET  /models/reload : 모델 캐시 재로드

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

app = FastAPI(title="PHM Inference Server", version="1.0.0")

# ── 모델 캐시 {sensor_type: (session, meta)} ──────────────────────────────────
_sessions: dict = {}

_SENSOR_CANDIDATES = {
    "accel":  ["ae_fd.onnx",     "cnn1d_fd.onnx"],
    "torque": ["ae_torque.onnx", "cnn1d_torque.onnx"],
}


def _load_model(sensor_type: str):
    """캐시에서 꺼내거나 디스크에서 로드합니다."""
    if sensor_type in _sessions:
        return _sessions[sensor_type]

    candidates = _SENSOR_CANDIDATES.get(sensor_type, [])
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
            _sessions[sensor_type] = (sess, meta)
            print(
                f"[inference] 모델 로드: {fname}  kind={meta.get('kind','?')}",
                flush=True,
            )
            return sess, meta
        except Exception as e:
            print(f"[inference] 모델 로드 실패 {fname}: {e}", flush=True)

    return None, None


# ── Pydantic 모델 ─────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    sensor_type: str = "accel"     # "accel" | "torque"
    window: List[float]            # flat float32 배열, 길이 = window_size × n_channels
    window_size: int = 1024
    n_channels: int = 3


class PredictResponse(BaseModel):
    model_type: str                # "AE-CNN1D" | "CNN1D"
    sensor_type: str
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


# ── 엔드포인트 ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    loaded = {k: v[1].get("kind", "?") for k, v in _sessions.items()}
    available = {}
    for st, candidates in _SENSOR_CANDIDATES.items():
        available[st] = [f for f in candidates if (MODELS_ROOT / f).exists()]
    return {
        "status": "ok",
        "loaded_models": loaded,
        "available_models": available,
        "models_root": str(MODELS_ROOT),
    }


@app.get("/models/reload")
def reload_models():
    _sessions.clear()
    return {"status": "reloaded", "message": "다음 /predict 호출 시 재로드됩니다."}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    sess, meta = _load_model(req.sensor_type)
    if sess is None:
        avail = [f for f in _SENSOR_CANDIDATES.get(req.sensor_type, [])
                 if (MODELS_ROOT / f).exists()]
        raise HTTPException(
            status_code=404,
            detail=(
                f"sensor_type='{req.sensor_type}' 모델 없음. "
                f"사용 가능: {avail if avail else '없음 — Airflow 학습을 먼저 실행하세요.'}"
            ),
        )

    # ── 모델 기대 채널 수 검증 ────────────────────────────────────────────────
    model_n_channels = meta.get("n_channels")
    if model_n_channels and req.n_channels != model_n_channels:
        raise HTTPException(
            status_code=400,
            detail=(
                f"채널 수 불일치: 모델={model_n_channels}ch, 요청={req.n_channels}ch. "
                f"모델 학습 채널: {meta.get('channels', '?')}. "
                f"Airflow 재학습 또는 CSV 재수집 후 retry."
            ),
        )

    expected = req.window_size * req.n_channels
    if len(req.window) != expected:
        raise HTTPException(
            status_code=400,
            detail=f"window 길이 {len(req.window)} ≠ {req.window_size}×{req.n_channels}={expected}",
        )

    try:
        # (1, T, C) float32
        raw_arr  = np.array(req.window, dtype=np.float32).reshape(1, req.window_size, req.n_channels)
        norm_arr = _zscore(raw_arr)

        model_kind = meta.get("kind", "CNN1D")
        input_name = sess.get_inputs()[0].name

        # ── AE-CNN1D: 재구성 오차로 이상 탐지 ────────────────────────────────
        if model_kind == "AE-CNN1D":
            recon = sess.run(None, {input_name: norm_arr})[0]   # (1, T, C)
            mae   = float(np.abs(norm_arr - recon).mean())
            thr   = float(meta.get("threshold", 0.1))

            rms     = float(np.sqrt(np.mean(raw_arr.astype(np.float64) ** 2)))
            rms_thr = float(meta.get("rms_thr", float("inf")))
            rms_mean= float(meta.get("rms_mean", 0.0))
            rms_norm= max(0.0, (rms - rms_mean) / max(rms_thr - rms_mean, 1e-8)) \
                      if rms_thr < 1e30 else 0.0

            mae_norm     = mae / max(thr, 1e-8)
            score_normed = mae_norm + 0.3 * rms_norm
            is_anomaly   = mae >= thr or rms >= rms_thr

            return PredictResponse(
                model_type="AE-CNN1D",
                sensor_type=req.sensor_type,
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
            is_anomaly=is_anomaly,
            anomaly_score=round(anomaly_score, 6),
            threshold=0.5,
            class_name=pred_class,
            confidence=round(confidence, 6),
        )

    except HTTPException:
        raise
    except Exception as e:
        # ONNX 실행 오류 등 — 상세 메시지를 500으로 반환
        raise HTTPException(status_code=500, detail=f"추론 실패: {type(e).__name__}: {e}")


# ── 엔트리포인트 ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print(f"[inference] 모델 루트: {MODELS_ROOT}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
