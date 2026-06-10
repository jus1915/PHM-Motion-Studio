"""
PHM 추론 서버 — FastAPI + ONNX Runtime
========================================
POST /predict           : 신호 윈도우 → 이상탐지 / 분류 결과 반환
POST /predict/combined  : AE 이상탐지 + CLS 결함진단 동시 수행
GET  /health            : 로드된 모델 목록 반환
GET  /models/reload     : 모델 캐시 재로드

sensor_type 지원:
  "accel"    — 가속도 전용 모델 (ae_fd / cls_accel)
  "torque"   — 토크 전용 모델   (ae_torque / cls_torque)
  "combined" — 가속도+토크 결합 CLS 모델 (cls_combined) — AE 없이 CLS 단독 운영

Per-axis 모델 지원:
  axis=0 → cls_accel_ax0.onnx 우선, 없으면 cls_accel.onnx 로 폴백
  axis=1 → cls_accel_ax1.onnx 우선, ...

채널 증강 재현:
  meta.json 의 augment_mode / add_* 플래그를 읽어 학습과 동일한 전처리 적용
    "standard" → FFT + derivative + abs (모든 채널 균일)
    "mixed"    → 가속도: FFT+deriv / 토크: deriv+stats×6 / 공통: abs

환경변수:
  PHM_MODELS_ROOT : ONNX 모델 루트 경로 (기본 /opt/phm/models)

의존성 (자동 설치):
  fastapi uvicorn onnxruntime numpy pydantic
"""

import json
import math
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
from typing import List, Optional, Tuple

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
    # combined: AE 모델은 별도로 학습하지 않음 — CLS 전용
}

# CLS(결함진단) 전용 후보 파일 — 신규 cls_ 접두사 우선, cnn1d_ 레거시 폴백
_CLS_CANDIDATES = {
    "accel":    ["cls_accel.onnx",    "cls_fd.onnx",     "cnn1d_fd.onnx"],
    "torque":   ["cls_torque.onnx",   "cnn1d_torque.onnx"],
    "combined": ["cls_combined.onnx"],
}

# 최대 지원 축 수 (health 엔드포인트에서 스캔용)
_MAX_AXIS_SCAN = 8


def _per_axis_candidates(sensor_type: str, axis: int) -> List[str]:
    """AE per-axis 모델 후보 파일명 목록 (우선순위 높은 순)."""
    if sensor_type == "accel":
        return [f"ae_fd_ax{axis}.onnx", f"cnn1d_fd_ax{axis}.onnx"]
    if sensor_type == "torque":
        return [f"ae_torque_ax{axis}.onnx", f"cnn1d_torque_ax{axis}.onnx"]
    # combined: AE 모델 없음
    return []


def _per_axis_cls_candidates(sensor_type: str, axis: int) -> List[str]:
    """CLS per-axis 모델 후보 파일명 목록 (신규 cls_ 접두사 우선)."""
    if sensor_type == "accel":
        return [f"cls_accel_ax{axis}.onnx", f"cls_fd_ax{axis}.onnx", f"cnn1d_fd_ax{axis}.onnx"]
    if sensor_type == "torque":
        return [f"cls_torque_ax{axis}.onnx", f"cnn1d_torque_ax{axis}.onnx"]
    if sensor_type == "combined":
        return [f"cls_combined_ax{axis}.onnx"]
    return []


def _cache_key(sensor_type: str, axis: Optional[int]) -> str:
    return f"{sensor_type}_ax{axis}" if axis is not None else sensor_type


def _cls_cache_key(sensor_type: str, axis: Optional[int]) -> str:
    return f"cls_{sensor_type}_ax{axis}" if axis is not None else f"cls_{sensor_type}"


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


def _load_cls_model(sensor_type: str, axis: Optional[int] = None):
    """CLS 결함진단 모델 로드 (AE 캐시와 독립 관리).

    axis 지정 시: per-axis CLS 후보 → 전역 CLS 폴백 순으로 탐색.
    AE 모델(kind에 "AE" 포함)이 발견되면 건너뜀 — AE 모델이 CLS 슬롯에 로드되는 것 방지.
    """
    key = _cls_cache_key(sensor_type, axis)
    if key in _sessions:
        return _sessions[key]

    if axis is not None:
        candidates = _per_axis_cls_candidates(sensor_type, axis) + \
                     _CLS_CANDIDATES.get(sensor_type, [])
    else:
        candidates = list(_CLS_CANDIDATES.get(sensor_type, []))
        for _ax in range(_MAX_AXIS_SCAN):
            for _fname in _per_axis_cls_candidates(sensor_type, _ax):
                if (MODELS_ROOT / _fname).exists():
                    candidates.append(_fname)
                    break

    for fname in candidates:
        model_path = MODELS_ROOT / fname
        if not model_path.exists():
            continue
        meta_path = MODELS_ROOT / (fname.replace(".onnx", "_meta.json"))
        try:
            sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            meta: dict = {}
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            # AE 모델이 CLS 슬롯에 로드되지 않도록 확인
            model_kind = meta.get("kind", "CNN1D")
            if "AE" in model_kind.upper():
                continue
            _sessions[key] = (sess, meta)
            print(f"[inference] CLS모델 로드: {fname}  key={key}  kind={model_kind}", flush=True)
            return sess, meta
        except Exception as e:
            print(f"[inference] CLS모델 로드 실패 {fname}: {e}", flush=True)

    return None, None


# ── Pydantic 모델 ─────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    sensor_type: str = "accel"     # "accel" | "torque"
    axis: Optional[int] = None     # 축 인덱스 (None = 레거시/전축 모델)
    window: List[float]            # flat float32 배열, 길이 = window_size × n_channels
    window_size: int = 1024
    n_channels: int = 3


class CombinedPredictRequest(BaseModel):
    """AE 이상탐지 + CLS 결함진단 동시 요청."""
    sensor_type: str = "accel"
    axis: Optional[int] = None
    window: List[float]
    window_size: int = 256
    n_channels: int = 3


class CombinedPredictResponse(BaseModel):
    """AE 이상탐지 결과(필수) + CLS 결함진단 결과(선택)."""
    sensor_type: str
    axis: Optional[int] = None
    # ── AE 이상탐지 ──────────────────────────────
    ae_model_file: Optional[str] = None
    is_anomaly: bool
    anomaly_score: float           # 정규화 스코어 (>1.0 이상)
    threshold: float = 1.0
    raw_mae: Optional[float] = None
    raw_threshold: Optional[float] = None
    # ── CLS 결함진단 ─────────────────────────────
    cls_available: bool = False    # True이면 CLS 모델이 있고 결과도 유효
    cls_model_file: Optional[str] = None
    cls_class_name: Optional[str] = None   # 예: "looseness", "normal"
    cls_confidence: Optional[float] = None
    cls_is_fault: Optional[bool] = None    # True이면 결함 클래스


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


def _zscore_ch(arr: np.ndarray) -> np.ndarray:
    """arr shape (T, C) → 채널별 z-score 정규화"""
    mean = arr.mean(axis=0, keepdims=True)
    std  = arr.std(axis=0, keepdims=True)
    std  = np.where(std < 1e-8, 1.0, std)
    return ((arr - mean) / std).astype(np.float32)


# ── 헬퍼 — mixed 채널 증강 (학습 _add_mixed_feature_channels 와 동일 로직) ─────
def _augment_mixed(raw: np.ndarray, meta: dict) -> np.ndarray:
    """
    augment_mode="mixed" 전처리.
    raw: (T, C_raw)  — normalize=False 로 로드된 원시 데이터

    가속도 채널(x/y/z): FFT magnitude(z-score) + derivative
    토크 채널(*_Trq%):  derivative + stats tile(mean/std/rms/min/max/ptp)
    공통:              abs (전채널)

    채널 식별은 meta["channels"] 로 수행합니다.
    """
    channels  = meta.get("channels", [])
    accel_idx = [i for i, ch in enumerate(channels) if ch.strip().lower() in ("x", "y", "z")]
    torque_idx= [i for i, ch in enumerate(channels) if "trq" in ch.lower()]

    add_fft   = bool(meta.get("add_fft_channels",        True))
    add_deriv = bool(meta.get("add_derivative_channels", True))
    add_stats = bool(meta.get("add_torque_stats",        True))
    add_abs   = bool(meta.get("add_abs_channels",        True))

    T, C = raw.shape
    parts: List[np.ndarray] = [raw]

    # ── 가속도 ───────────────────────────────────────────────────────────────
    if accel_idx:
        accel = raw[:, accel_idx]

        if add_fft:
            fft_raw   = np.abs(np.fft.rfft(accel, axis=0)).astype(np.float32)
            half      = fft_raw.shape[0]
            repeats   = math.ceil(T / half)
            fft_tiled = np.tile(fft_raw, (repeats, 1))[:T]
            fft_tiled = _zscore_ch(fft_tiled)   # 주파수 패턴 강조
            parts.append(fft_tiled)

        if add_deriv:
            parts.append(np.diff(accel, axis=0, prepend=accel[:1]).astype(np.float32))

    # ── 토크 ─────────────────────────────────────────────────────────────────
    if torque_idx:
        torque = raw[:, torque_idx]

        if add_deriv:
            parts.append(np.diff(torque, axis=0, prepend=torque[:1]).astype(np.float32))

        if add_stats:
            stats_rows = np.array([
                torque.mean(axis=0),
                torque.std(axis=0),
                np.sqrt(np.mean(torque.astype(np.float64) ** 2, axis=0)),  # RMS
                torque.min(axis=0),
                torque.max(axis=0),
                torque.ptp(axis=0),   # peak-to-peak
            ], dtype=np.float32)                                # (6, n_torque)
            stats_flat  = stats_rows.T.reshape(1, -1)           # (1, n_torque*6)
            stats_tiled = np.tile(stats_flat, (T, 1)).astype(np.float32)
            parts.append(stats_tiled)

    # ── 공통: 절댓값 ─────────────────────────────────────────────────────────
    if add_abs:
        parts.append(np.abs(raw).astype(np.float32))

    return np.concatenate(parts, axis=1).astype(np.float32)


# ── 헬퍼 — standard 채널 증강 (학습 _add_feature_channels 와 동일 로직) ─────────
def _augment_standard(raw: np.ndarray, meta: dict) -> np.ndarray:
    """
    augment_mode="standard" 전처리.
    raw: (T, C)  — per-window z-score 정규화된 데이터

    전채널 균일: FFT magnitude(z-score) + derivative + abs
    """
    add_fft   = bool(meta.get("add_fft_channels",        True))
    add_deriv = bool(meta.get("add_derivative_channels", True))
    add_abs   = bool(meta.get("add_abs_channels",        True))

    T, C = raw.shape
    parts: List[np.ndarray] = [raw]

    if add_fft:
        fft_raw   = np.abs(np.fft.rfft(raw, axis=0)).astype(np.float32)
        half      = fft_raw.shape[0]
        repeats   = math.ceil(T / half)
        fft_tiled = np.tile(fft_raw, (repeats, 1))[:T]
        fft_tiled = _zscore_ch(fft_tiled)
        parts.append(fft_tiled)

    if add_deriv:
        parts.append(np.diff(raw, axis=0, prepend=raw[:1]).astype(np.float32))

    if add_abs:
        parts.append(np.abs(raw).astype(np.float32))

    return np.concatenate(parts, axis=1).astype(np.float32) if len(parts) > 1 else parts[0]


# ── 헬퍼 — 채널 증강 + 정규화 (학습과 동일한 전처리) ─────────────────────────
def _preprocess_window(
    window_flat,          # list[float] | flat np.ndarray
    window_size: int,
    n_channels:  int,     # C# 가 전송한 RAW 채널 수
    meta:        dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    meta["augment_mode"] 에 따라 학습 시와 동일한 채널 증강 + 정규화를 적용합니다.

    augment_mode="mixed"    → _augment_mixed()  — normalize=False (RAW 진폭 보존)
    augment_mode="standard" → _augment_standard() + per-window z-score

    Returns:
        raw_arr:  (1, T, n_raw)   — 원본 데이터 (RMS 계산용)
        proc_arr: (1, T, n_model) — 증강+정규화 완료 (모델 입력용)
    """
    raw = np.array(window_flat, dtype=np.float32).reshape(window_size, n_channels)
    raw_arr = raw[np.newaxis, :, :].astype(np.float32)   # (1, T, n_raw)

    augment_mode = str(meta.get("augment_mode", "standard")).lower()
    do_augment   = bool(meta.get("add_augmented_channels", True))

    if augment_mode == "mixed":
        # RAW 진폭 보존 — 정규화 없이 mixed 증강 적용
        aug = _augment_mixed(raw, meta) if do_augment else raw
        proc_arr = aug[np.newaxis, :, :].astype(np.float32)
    else:
        # standard: per-window z-score 후 증강
        norm_raw = _zscore_ch(raw)
        aug = _augment_standard(norm_raw, meta) if do_augment else norm_raw
        proc_arr = aug[np.newaxis, :, :].astype(np.float32)

        # global norm (메타에 저장된 통계) 이 있으면 덮어쓰기
        norm_mean = meta.get("norm_mean")
        norm_std  = meta.get("norm_std")
        if norm_mean is not None and norm_std is not None:
            m        = np.array(norm_mean, dtype=np.float32).reshape(1, 1, -1)
            s        = np.array(norm_std,  dtype=np.float32).reshape(1, 1, -1)
            s        = np.where(s < 1e-8, 1.0, s)
            proc_arr = ((aug[np.newaxis] - m) / s).astype(np.float32)

    return raw_arr, proc_arr


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

    # 사용 가능한 AE 모델 스캔
    available_ae: dict = {}
    for st, fallbacks in _FALLBACK_CANDIDATES.items():
        files = []
        for ax in range(_MAX_AXIS_SCAN):
            for fname in _per_axis_candidates(st, ax):
                if (MODELS_ROOT / fname).exists():
                    files.append(fname)
                    break
        for fname in fallbacks:
            if (MODELS_ROOT / fname).exists():
                files.append(fname)
        available_ae[st] = files

    # 사용 가능한 CLS 모델 스캔 (accel / torque / combined)
    available_cls: dict = {}
    for st in list(_CLS_CANDIDATES.keys()) + ["combined"]:
        fallbacks = _CLS_CANDIDATES.get(st, [])
        files = []
        for ax in range(_MAX_AXIS_SCAN):
            for fname in _per_axis_cls_candidates(st, ax):
                if (MODELS_ROOT / fname).exists():
                    files.append(fname)
                    break
        for fname in fallbacks:
            if (MODELS_ROOT / fname).exists() and fname not in files:
                files.append(fname)
        if files:
            available_cls[st] = files

    return {
        "status":               "ok",
        "loaded_models":        loaded,
        "available_ae_models":  available_ae,
        "available_cls_models": available_cls,
        "models_root":          str(MODELS_ROOT),
    }


def reload_models():  # 채널 AE 섹션의 reload_models_v2 로 대체됩니다
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

    # ── 모델 학습 window_size 와 요청 window_size 불일치 처리 ─────────────────
    # 모델이 256 샘플로 학습됐는데 클라이언트가 1024를 보내는 경우 등:
    # 요청 윈도우 끝(가장 최신) 에서 모델 window_size 만큼 슬라이싱해서 사용.
    model_window_size = int(meta.get("window_size", req.window_size))
    if model_window_size != req.window_size:
        need = model_window_size * req.n_channels
        if len(req.window) < need:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"모델 window_size={model_window_size} 이지만 "
                    f"요청 데이터({len(req.window)} 샘플)가 부족합니다."
                ),
            )
        # 가장 최신 데이터(끝 부분) 사용
        effective_window = list(req.window[-need:])
        effective_window_size = model_window_size
        print(
            f"[inference] window_size 조정: {req.window_size}→{model_window_size} "
            f"({req.sensor_type} axis={req.axis})",
            flush=True,
        )
    else:
        effective_window      = req.window
        effective_window_size = req.window_size

    # 실제 사용된 모델 파일명 (meta에 source_file 없으면 axis로 추정)
    model_file = meta.get("source_file")

    try:
        # 채널 증강 + 정규화 (학습과 동일한 전처리)
        raw_arr, proc_arr = _preprocess_window(
            effective_window, effective_window_size, req.n_channels, meta)

        model_kind = meta.get("kind", "CNN1D")
        input_name = sess.get_inputs()[0].name

        # ── AE-CNN1D: 재구성 오차로 이상 탐지 ────────────────────────────────
        if model_kind == "AE-CNN1D":
            is_anomaly, score_normed, mae, thr = _ae_score(
                proc_arr, raw_arr, sess, meta, req.sensor_type
            )
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
        logits      = sess.run(None, {input_name: proc_arr})[0]
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


# ── AE 스코어링 헬퍼 (predict / predict_combined 공용) ───────────────────────
def _ae_score(
    proc_arr: "np.ndarray",
    raw_arr:  "np.ndarray",
    sess,
    meta: dict,
    sensor_type: str,
) -> tuple:
    """AE 재구성 오차 기반 이상 점수를 계산합니다.

    Args:
        proc_arr: 증강+정규화 완료 배열 (1, T, C_model) — 모델 입력
        raw_arr:  원본 배열        (1, T, C_raw)   — RMS 계산용

    Returns:
        (is_anomaly, score_normed, mae, thr)
    """
    input_name = sess.get_inputs()[0].name
    recon  = sess.run(None, {input_name: proc_arr})[0]
    mae    = float(np.abs(proc_arr - recon).mean())
    thr    = float(meta.get("threshold", 0.1))

    rms      = float(np.sqrt(np.mean(raw_arr.astype(np.float64) ** 2)))
    rms_thr  = float(meta.get("rms_thr",  float("inf")))
    rms_mean = float(meta.get("rms_mean", 0.0))
    rms_std_m = meta.get("rms_std")
    if rms_thr < 1e30:
        denom    = float(rms_std_m) if rms_std_m else max(rms_thr - rms_mean, 1e-8)
        rms_norm = max(0.0, (rms - rms_mean) / denom)
    else:
        rms_norm = 0.0

    if sensor_type == "accel":
        TH_SCALE = 2.0; RMS_WEIGHT = 0.25; ANOMALY_THRESHOLD = 1.5
    else:
        TH_SCALE = 1.3; RMS_WEIGHT = 0.35; ANOMALY_THRESHOLD = 1.3

    mae_norm     = mae / max(thr * TH_SCALE, 1e-8)
    score_normed = mae_norm + RMS_WEIGHT * rms_norm
    is_anomaly   = score_normed >= ANOMALY_THRESHOLD
    return is_anomaly, score_normed, mae, thr


# ── /predict/combined ─────────────────────────────────────────────────────────
@app.post("/predict/combined", response_model=CombinedPredictResponse)
def predict_combined(req: CombinedPredictRequest):
    """AE 이상탐지 + CLS 결함진단을 한 번에 수행합니다.

    sensor_type="accel" | "torque":
      AE 모델 필수 + CLS 모델 선택.
      AE 없으면 404.

    sensor_type="combined":
      CLS 전용 모드 — AE 모델 없이 CLS 단독 운영.
      CLS 예측 결과로 is_anomaly 를 파생합니다 (비정상 클래스 → True).
      CLS도 없으면 404.

    window 전처리(z-score, window_size 조정)는 AE/CLS 각각 독립 수행.
    """
    expected = req.window_size * req.n_channels
    if len(req.window) != expected:
        raise HTTPException(
            status_code=400,
            detail=f"window 길이 {len(req.window)} ≠ {req.window_size}×{req.n_channels}={expected}",
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # "combined" 센서 타입: CLS 전용 경로 (AE 모델 없음)
    # ═══════════════════════════════════════════════════════════════════════════
    if req.sensor_type == "combined":
        cls_sess, cls_meta = _load_cls_model(req.sensor_type, req.axis)
        if cls_sess is None:
            avail: List[str] = []
            if req.axis is not None:
                for fn in _per_axis_cls_candidates(req.sensor_type, req.axis):
                    if (MODELS_ROOT / fn).exists():
                        avail.append(fn)
            for fn in _CLS_CANDIDATES.get(req.sensor_type, []):
                if (MODELS_ROOT / fn).exists():
                    avail.append(fn)
            raise HTTPException(
                status_code=404,
                detail=(
                    f"combined CLS 모델 없음 (axis={req.axis}). "
                    f"사용 가능: {avail if avail else '없음 — Airflow train_combined 을 먼저 실행하세요.'}"
                ),
            )
        try:
            cls_ws = int(cls_meta.get("window_size", req.window_size))
            if cls_ws != req.window_size:
                need = cls_ws * req.n_channels
                if len(req.window) < need:
                    raise HTTPException(
                        status_code=400,
                        detail=f"CLS 모델 window_size={cls_ws} 이지만 데이터({len(req.window)}샘플) 부족",
                    )
                eff_win, eff_ws = list(req.window[-need:]), cls_ws
            else:
                eff_win, eff_ws = req.window, req.window_size

            _, cls_proc = _preprocess_window(eff_win, eff_ws, req.n_channels, cls_meta)
            cls_input   = cls_sess.get_inputs()[0].name
            logits      = cls_sess.run(None, {cls_input: cls_proc})[0]
            exp_l       = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs       = exp_l / exp_l.sum(axis=1, keepdims=True)
            pred_idx    = int(np.argmax(probs[0]))
            confidence  = float(probs[0][pred_idx])
            cls_names   = cls_meta.get("class_names", ["normal", "fault"])
            class_name  = cls_names[pred_idx] if pred_idx < len(cls_names) else str(pred_idx)
            is_fault    = class_name.lower() != "normal"

            # AE 스코어 없이 CLS 확신도로 anomaly_score 파생
            # 비정상 클래스: confidence 그대로, 정상: 1.0 - confidence
            anomaly_score = round(confidence if is_fault else (1.0 - confidence), 6)

            return CombinedPredictResponse(
                sensor_type    = req.sensor_type,
                axis           = req.axis,
                ae_model_file  = None,
                is_anomaly     = is_fault,
                anomaly_score  = anomaly_score,
                threshold      = 1.0,
                raw_mae        = None,
                raw_threshold  = None,
                cls_available  = True,
                cls_model_file = cls_meta.get("source_file"),
                cls_class_name = class_name,
                cls_confidence = round(confidence, 6),
                cls_is_fault   = is_fault,
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"combined CLS 추론 실패: {type(e).__name__}: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # "accel" | "torque": AE 필수 + CLS 선택
    # ═══════════════════════════════════════════════════════════════════════════
    ae_sess, ae_meta = _load_model(req.sensor_type, req.axis)
    if ae_sess is None:
        avail: List[str] = []
        if req.axis is not None:
            for fn in _per_axis_candidates(req.sensor_type, req.axis):
                if (MODELS_ROOT / fn).exists():
                    avail.append(fn)
        for fn in _FALLBACK_CANDIDATES.get(req.sensor_type, []):
            if (MODELS_ROOT / fn).exists():
                avail.append(fn)
        raise HTTPException(
            status_code=404,
            detail=(
                f"AE 모델 없음 (sensor_type={req.sensor_type!r}, axis={req.axis}). "
                f"사용 가능: {avail if avail else '없음 — Airflow 학습(AE)을 먼저 실행하세요.'}"
            ),
        )

    # ── AE window 전처리 ─────────────────────────────────────────
    ae_window_size = int(ae_meta.get("window_size", req.window_size))
    if ae_window_size != req.window_size:
        need = ae_window_size * req.n_channels
        if len(req.window) < need:
            raise HTTPException(status_code=400,
                detail=f"AE 모델 window_size={ae_window_size} 이지만 데이터({len(req.window)}샘플) 부족")
        eff_win = list(req.window[-need:])
        eff_ws  = ae_window_size
    else:
        eff_win = req.window
        eff_ws  = req.window_size

    try:
        ae_raw_arr, ae_proc_arr = _preprocess_window(eff_win, eff_ws, req.n_channels, ae_meta)

        ae_model_kind = ae_meta.get("kind", "AE-CNN1D")
        if "AE" not in ae_model_kind.upper():
            print(f"[inference] 경고: AE 슬롯에 CLS 모델({ae_model_kind}) 로드됨", flush=True)

        is_anomaly, ae_score, mae, thr = _ae_score(ae_proc_arr, ae_raw_arr, ae_sess, ae_meta, req.sensor_type)
        ae_model_file = ae_meta.get("source_file")

        # ── CLS 모델 로드 & 추론 (선택) ─────────────────────────
        cls_sess, cls_meta = _load_cls_model(req.sensor_type, req.axis)
        cls_available   = cls_sess is not None
        cls_class_name: Optional[str]  = None
        cls_confidence: Optional[float] = None
        cls_is_fault:   Optional[bool]  = None
        cls_model_file: Optional[str]   = None

        if cls_sess is not None:
            cls_ws = int(cls_meta.get("window_size", eff_ws))
            if cls_ws != eff_ws:
                need2 = cls_ws * req.n_channels
                if len(req.window) >= need2:
                    _, cls_proc = _preprocess_window(
                        list(req.window[-need2:]), cls_ws, req.n_channels, cls_meta)
                else:
                    cls_proc = ae_proc_arr
            else:
                _, cls_proc = _preprocess_window(eff_win, eff_ws, req.n_channels, cls_meta)

            cls_input_name = cls_sess.get_inputs()[0].name
            logits     = cls_sess.run(None, {cls_input_name: cls_proc})[0]
            exp_l      = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs      = exp_l / exp_l.sum(axis=1, keepdims=True)
            pred_idx   = int(np.argmax(probs[0]))
            cls_confidence = float(probs[0][pred_idx])
            cls_names  = cls_meta.get("class_names", ["normal", "fault"])
            cls_class_name = cls_names[pred_idx] if pred_idx < len(cls_names) else str(pred_idx)
            cls_is_fault   = cls_class_name.lower() != "normal"
            cls_model_file = cls_meta.get("source_file")

        return CombinedPredictResponse(
            sensor_type    = req.sensor_type,
            axis           = req.axis,
            ae_model_file  = ae_model_file,
            is_anomaly     = is_anomaly,
            anomaly_score  = round(ae_score, 6),
            threshold      = 1.0,
            raw_mae        = round(mae, 6),
            raw_threshold  = round(thr, 6),
            cls_available  = cls_available,
            cls_model_file = cls_model_file,
            cls_class_name = cls_class_name,
            cls_confidence = round(cls_confidence, 6) if cls_confidence is not None else None,
            cls_is_fault   = cls_is_fault,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 실패: {type(e).__name__}: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  채널별 활성 윈도우 AE (Channel Active AE)
# ═══════════════════════════════════════════════════════════════════════════════

CH_AE_META_FILENAME = "ch_ae_meta.json"

# 채널 AE 전용 캐시
_ch_ae_meta_cache: Optional[dict] = None      # ch_ae_meta.json 내용
_ch_ae_sessions: dict = {}                    # {channel_name: ort.InferenceSession}


def _load_ch_ae_meta() -> Optional[dict]:
    """ch_ae_meta.json 을 로드합니다 (프로파일 디렉토리 스캔)."""
    global _ch_ae_meta_cache
    if _ch_ae_meta_cache is not None:
        return _ch_ae_meta_cache

    # 프로파일 서브디렉토리 포함 탐색
    candidates = [MODELS_ROOT / CH_AE_META_FILENAME] + \
                 sorted((MODELS_ROOT).glob(f"*/{CH_AE_META_FILENAME}"))
    for path in candidates:
        if path.exists():
            try:
                _ch_ae_meta_cache = json.loads(path.read_text(encoding="utf-8"))
                _ch_ae_meta_cache["_meta_dir"] = str(path.parent)
                print(f"[ch_ae] 메타 로드: {path}", flush=True)
                return _ch_ae_meta_cache
            except Exception as e:
                print(f"[ch_ae] 메타 로드 실패 {path}: {e}", flush=True)
    return None


def _load_ch_ae_session(channel: str) -> Optional[ort.InferenceSession]:
    """채널 AE ONNX 세션을 로드합니다."""
    if channel in _ch_ae_sessions:
        return _ch_ae_sessions[channel]

    meta = _load_ch_ae_meta()
    if meta is None:
        return None

    ch_info = meta.get("channels", {}).get(channel)
    if ch_info is None:
        return None

    model_file = ch_info.get("model_file")
    if not model_file:
        return None

    meta_dir = Path(meta.get("_meta_dir", str(MODELS_ROOT)))
    model_path = meta_dir / model_file
    if not model_path.exists():
        # MODELS_ROOT 에서도 시도
        model_path = MODELS_ROOT / model_file
    if not model_path.exists():
        print(f"[ch_ae] 모델 파일 없음: {model_file}", flush=True)
        return None

    try:
        sess = ort.InferenceSession(str(model_path))
        _ch_ae_sessions[channel] = sess
        print(f"[ch_ae] 모델 로드: {model_path}", flush=True)
        return sess
    except Exception as e:
        print(f"[ch_ae] 모델 로드 실패 {model_path}: {e}", flush=True)
        return None


def _ch_ae_compute_features(window: np.ndarray) -> dict:
    """단일 채널 윈도우 → 13개 통계 피처 (학습과 동일 로직)."""
    try:
        from scipy.stats import skew as _skew, kurtosis as _kurtosis
        _use_scipy = True
    except ImportError:
        _use_scipy = False

    eps = 1e-8
    x = window.astype(np.float32)

    min_v  = float(np.min(x))
    max_v  = float(np.max(x))
    rms_v  = float(np.sqrt(np.mean(x ** 2)))
    std_v  = float(np.std(x))
    var_v  = float(np.var(x))
    p2p_v  = max_v - min_v
    mabs_v = float(np.mean(np.abs(x)))
    pabs_v = float(np.max(np.abs(x)))

    if std_v < eps:
        skew_v, kurt_v = 0.0, 3.0
    elif _use_scipy:
        skew_v = float(_skew(x, bias=False))
        kurt_v = float(_kurtosis(x, fisher=False, bias=False))
    else:
        # scipy 없을 때 numpy 근사
        n = len(x)
        mu = x.mean(); s = x.std()
        skew_v = float(np.mean(((x - mu) / (s + eps)) ** 3))
        kurt_v = float(np.mean(((x - mu) / (s + eps)) ** 4))

    crest_v   = pabs_v / (rms_v + eps)
    shape_v   = rms_v  / (mabs_v + eps)
    impulse_v = pabs_v / (mabs_v + eps)

    return {
        "min": min_v, "max": max_v, "rms": rms_v, "std": std_v, "var": var_v,
        "p2p": p2p_v, "mean_abs": mabs_v, "peak_abs": pabs_v,
        "skewness": skew_v, "kurtosis": kurt_v,
        "crest": crest_v, "shape": shape_v, "impulse": impulse_v,
    }


def _ch_ae_active_score(features: dict, ch_meta: dict) -> float:
    """active_score 를 계산합니다 (robust z-score of p2p/rms/std, clipped ≥ 0)."""
    score = 0.0
    for col in ("p2p", "rms", "std"):
        med_key = f"score_{col}_median"
        mad_key = f"score_{col}_mad"
        med = ch_meta.get(med_key, 0.0)
        mad = ch_meta.get(mad_key, 1.0)
        if mad < 1e-8:
            mad = 1.0
        z = 0.6745 * (features[col] - med) / mad
        score += max(0.0, z)
    return score / 3.0


# ── Pydantic 모델 ──────────────────────────────────────────────────────────────
class ChannelAePredictRequest(BaseModel):
    channel: str                    # 예: "x", "Ax0_Trq(%)"
    window: List[float]             # 단일 채널 원시 샘플 (window_size 개)
    window_size: int = 128


class ChannelAePredictResponse(BaseModel):
    channel: str
    active_state: str               # "active" | "inactive" | "uncertain"
    active_score: float
    is_anomaly: bool = False
    anomaly_score: float = 0.0      # 0~1+ 정규화 (thr95 기준)
    threshold: float = 1.0
    recon_error: Optional[float] = None
    raw_threshold: Optional[float] = None


# ── 엔드포인트 ────────────────────────────────────────────────────────────────
@app.get("/channel_ae_info")
def channel_ae_info():
    """ch_ae_meta.json 내용을 반환합니다."""
    meta = _load_ch_ae_meta()
    if meta is None:
        raise HTTPException(status_code=404, detail="ch_ae_meta.json 없음 — 학습을 먼저 실행하세요.")
    # _meta_dir 는 내부 필드이므로 제외
    return {k: v for k, v in meta.items() if not k.startswith("_")}


@app.post("/predict/channel_ae", response_model=ChannelAePredictResponse)
def predict_channel_ae(req: ChannelAePredictRequest):
    """단일 채널 윈도우 → active 판정 + (active면) AE 이상탐지."""
    if len(req.window) < req.window_size:
        raise HTTPException(
            status_code=400,
            detail=f"window 길이 {len(req.window)} < window_size {req.window_size}",
        )

    meta = _load_ch_ae_meta()
    if meta is None:
        raise HTTPException(status_code=404, detail="ch_ae_meta.json 없음")

    ch_info = meta.get("channels", {}).get(req.channel)
    if ch_info is None:
        raise HTTPException(status_code=404, detail=f"채널 없음: {req.channel}")

    # ── 1. 13개 피처 계산 ──────────────────────────────────────────────────────
    window_arr = np.array(req.window[-req.window_size:], dtype=np.float32)
    features   = _ch_ae_compute_features(window_arr)

    # ── 2. active_score 계산 ──────────────────────────────────────────────────
    active_score = _ch_ae_active_score(features, ch_info)
    active_thr   = ch_info.get("active_threshold",   float("inf"))
    inactive_thr = ch_info.get("inactive_threshold", 0.0)

    if active_score >= active_thr:
        active_state = "active"
    elif active_score <= inactive_thr:
        active_state = "inactive"
    else:
        active_state = "uncertain"

    # inactive / uncertain → AE 건너뜀
    if active_state != "active":
        return ChannelAePredictResponse(
            channel=req.channel,
            active_state=active_state,
            active_score=round(active_score, 6),
        )

    # ── 3. 피처 정규화 ────────────────────────────────────────────────────────
    FEATURE_COLS = ["min", "max", "rms", "std", "var", "p2p",
                    "mean_abs", "peak_abs", "skewness", "kurtosis",
                    "crest", "shape", "impulse"]
    feat_vec = np.array([features[c] for c in FEATURE_COLS], dtype=np.float32)

    scaler_mean = np.array(ch_info.get("scaler_mean", [0.0] * 13), dtype=np.float32)
    scaler_std  = np.array(ch_info.get("scaler_std",  [1.0] * 13), dtype=np.float32)
    scaler_std  = np.where(scaler_std < 1e-8, 1.0, scaler_std)
    feat_norm   = (feat_vec - scaler_mean) / scaler_std
    feat_input  = feat_norm.reshape(1, -1)  # (1, 13)

    # ── 4. ONNX 추론 ──────────────────────────────────────────────────────────
    sess = _load_ch_ae_session(req.channel)
    if sess is None:
        raise HTTPException(status_code=404, detail=f"채널 모델 없음: {req.channel}")

    try:
        input_name = sess.get_inputs()[0].name
        recon      = sess.run(None, {input_name: feat_input})[0]  # (1, 13)
        recon_error = float(np.mean((feat_input - recon) ** 2))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AE 추론 실패: {e}")

    # ── 5. 이상 점수 정규화 (thr99 기준 — 오탐 1% 목표) ──────────────────────
    # thr95 는 정의상 정상의 5%가 초과 → 간헐적 오탐. thr99 로 판정해 ~1%로 낮춘다.
    # (thr99 미저장 구 모델은 thr95 로 폴백)
    thr = ch_info.get("recon_error_thr_99")
    if thr is None or thr < 1e-10:
        thr = ch_info.get("recon_error_thr_95", 1e-3)
    if thr < 1e-10:
        thr = 1e-3
    anomaly_score = recon_error / thr            # >1.0 이면 이상
    is_anomaly    = anomaly_score > 1.0

    return ChannelAePredictResponse(
        channel       = req.channel,
        active_state  = "active",
        active_score  = round(active_score,  6),
        is_anomaly    = is_anomaly,
        anomaly_score = round(anomaly_score, 6),
        threshold     = 1.0,
        recon_error   = round(recon_error,   8),
        raw_threshold = round(thr,           8),
    )


# ── 모델 재로드 시 채널 AE 캐시도 초기화 ──────────────────────────────────────
@app.get("/models/reload")
def reload_models_v2():
    global _ch_ae_meta_cache, _ch_ae_sessions
    _sessions.clear()
    _ch_ae_meta_cache = None
    _ch_ae_sessions   = {}
    return {"status": "reloaded", "message": "AE + 채널 AE 캐시가 초기화됩니다."}


# ── 엔트리포인트 ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print(f"[inference] 모델 루트: {MODELS_ROOT}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
