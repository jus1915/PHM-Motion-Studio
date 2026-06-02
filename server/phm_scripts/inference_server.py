"""
PHM 추론 서버 — FastAPI + ONNX Runtime
========================================
POST /predict         : 신호 윈도우 → 이상탐지(AE) 결과 반환
POST /predict/combined: 이상탐지(AE) + 결함진단(CLS) 동시 반환
GET  /health          : 로드된 모델 목록 반환
GET  /models/reload   : 모델 캐시 재로드
GET  /profiles         : 사용 가능한 프로파일 목록 + 현재 활성 프로파일
POST /profiles/activate: 활성 프로파일 전환 (모델 캐시 자동 클리어)

모델 구조:
  accel (가속도, 단일 센서):
    AE  → ae_accel.onnx       (axis=None, 단일 전역 모델)
    CLS → cls_accel.onnx      (axis=None, 단일 전역 모델)
  torque (토크, 축별):
    AE  → ae_torque_ax{n}.onnx  (axis=n, per-axis 모델)
    CLS → cls_torque_ax{n}.onnx (axis=n, per-axis 모델)
  combined (결합, 축별, CLS 전용):
    CLS → cls_combined_ax{n}.onnx

  accel 은 물리적으로 단일 센서이므로 axis 없이 항상 axis=None 으로 요청합니다.

환경변수:
  PHM__models_root() : ONNX 모델 루트 경로 (기본 /opt/phm/models)

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
_MODELS_BASE    = Path(os.getenv("PHM__models_root()", "/opt/phm/models"))
_active_profile: str = "default"

# ── 런타임 튜닝 가능 스코어링 파라미터 ────────────────────────────────────────
# GET /config 로 조회, POST /config 로 변경 가능
_RMS_WEIGHT:        float = 0.0   # RMS 기여 가중치 (0.0 ~ 1.0) — 진단용 0 (순수 AE 재구성 오차만 평가)
_ANOMALY_THRESHOLD: float = 1.0   # 이상 판정 기준값 (log₂ 스코어 ≥ 이 값 → 이상)

def _models_root() -> Path:
    """현재 활성 프로파일의 모델 디렉토리.
    profile 서브디렉토리가 있으면 그것을, 없으면 베이스 경로 반환 (하위호환)."""
    p = _MODELS_BASE / _active_profile
    return p if p.is_dir() else _MODELS_BASE

app = FastAPI(title="PHM Inference Server", version="2.0.0")

# ── 모델 캐시 {cache_key: (session, meta)} ────────────────────────────────────
# cache_key = "{sensor_type}" 또는 "{sensor_type}_ax{n}"
_sessions: dict = {}

# AE 전역 후보 파일 — axis=None 시 또는 per-axis 모델 없을 때 폴백
# accel: 단일 전역 모델 ae_accel.onnx (레거시 ae_fd.onnx 폴백)
# torque: 전역 폴백은 없음 — per-axis 모델만 사용
_FALLBACK_CANDIDATES = {
    "accel":  ["ae_accel.onnx",  "ae_fd.onnx",     "cnn1d_fd.onnx"],
    "torque": ["ae_torque.onnx", "cnn1d_torque.onnx"],
}

# CLS(결함진단) 전용 후보 파일 — 신규 cls_ 접두사 우선, cnn1d_ 레거시 폴백
_CLS_CANDIDATES = {
    "accel":    ["cls_accel.onnx",    "cls_fd.onnx",     "cnn1d_fd.onnx"],
    "torque":   ["cls_torque.onnx",   "cnn1d_torque.onnx"],
    "combined": ["cls_combined.onnx"],  # AE 없이 CLS 단독 운영
}

# 최대 지원 축 수 (health 엔드포인트에서 스캔용)
_MAX_AXIS_SCAN = 8


def _per_axis_candidates(sensor_type: str, axis: int) -> List[str]:
    """AE per-axis 모델 후보 파일명 목록 (우선순위 높은 순).

    accel: 단일 전역 모델만 사용하므로 per-axis 후보 없음 → []
    torque: 축별 AE 모델 ae_torque_ax{n}.onnx
    """
    if sensor_type == "accel":
        return []   # accel 은 단일 전역 모델 (axis=None 으로만 요청)
    if sensor_type == "torque":
        return [f"ae_torque_ax{axis}.onnx", f"cnn1d_torque_ax{axis}.onnx"]
    if sensor_type == "combined":
        return [f"ae_combined_ax{axis}.onnx"]
    return []


def _per_axis_cls_candidates(sensor_type: str, axis: int) -> List[str]:
    """CLS per-axis 모델 후보 파일명 목록 (신규 cls_ 접두사 우선, 레거시 폴백).

    accel: 단일 전역 CLS 모델만 사용하므로 per-axis 후보 없음 → []
    torque: 축별 CLS 모델 cls_torque_ax{n}.onnx
    combined: 축별 CLS 모델 cls_combined_ax{n}.onnx
    """
    if sensor_type == "accel":
        return []   # accel 은 단일 전역 모델 (axis=None 으로만 요청)
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
                if (_models_root() / _fname).exists():
                    candidates.append(_fname)
                    break   # 해당 축의 최우선 모델 1개만 추가

    for fname in candidates:
        model_path = _models_root() / fname
        if not model_path.exists():
            continue
        meta_path = _models_root() / (fname.replace(".onnx", "_meta.json"))
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
                if (_models_root() / _fname).exists():
                    candidates.append(_fname)
                    break

    for fname in candidates:
        model_path = _models_root() / fname
        if not model_path.exists():
            continue
        meta_path = _models_root() / (fname.replace(".onnx", "_meta.json"))
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


# ── 헬퍼 — Z-score 정규화 ────────────────────────────────────────────────────
def _zscore(arr: np.ndarray) -> np.ndarray:
    """(1, T, C) 배열을 채널별 z-score 정규화합니다."""
    mean = arr.mean(axis=1, keepdims=True)
    std  = arr.std(axis=1, keepdims=True)
    std  = np.where(std < 1e-8, 1.0, std)
    return (arr - mean) / std


def _zscore_ch(arr: np.ndarray) -> np.ndarray:
    """(T, C) 배열을 채널별 z-score 정규화합니다."""
    mean = arr.mean(axis=0, keepdims=True)
    std  = arr.std(axis=0, keepdims=True)
    std  = np.where(std < 1e-8, 1.0, std)
    return ((arr - mean) / std).astype(np.float32)


# ── 헬퍼 — mixed 채널 증강 (_add_mixed_feature_channels 와 동일 로직) ──────────
def _augment_mixed(raw: np.ndarray, meta: dict) -> np.ndarray:
    """
    augment_mode="mixed" 전처리 (normalize=False RAW 데이터용).

    가속도 채널(x/y/z): FFT magnitude(채널별 z-score) + derivative
    토크 채널(*_Trq%):  derivative + 통계 tile(mean/std/rms/min/max/ptp)
    공통:              abs (전채널)

    채널 식별은 meta["channels"] 로 수행합니다.
    """
    channels   = meta.get("channels", [])
    accel_idx  = [i for i, ch in enumerate(channels) if ch.strip().lower() in ("x", "y", "z")]
    torque_idx = [i for i, ch in enumerate(channels) if "trq" in ch.lower()]

    add_fft   = bool(meta.get("add_fft_channels",        True))
    add_deriv = bool(meta.get("add_derivative_channels", True))
    add_stats = bool(meta.get("add_torque_stats",        False))  # 기존 모델은 stats 없이 학습 → False
    add_abs   = bool(meta.get("add_abs_channels",        True))

    T, _ = raw.shape
    parts: List[np.ndarray] = [raw]

    if accel_idx:
        accel = raw[:, accel_idx]
        if add_fft:
            fft_raw   = np.abs(np.fft.rfft(accel, axis=0)).astype(np.float32)
            half      = fft_raw.shape[0]
            fft_tiled = np.tile(fft_raw, (math.ceil(T / half), 1))[:T]
            parts.append(_zscore_ch(fft_tiled))
        if add_deriv:
            parts.append(np.diff(accel, axis=0, prepend=accel[:1]).astype(np.float32))

    if torque_idx:
        torque = raw[:, torque_idx]
        if add_deriv:
            parts.append(np.diff(torque, axis=0, prepend=torque[:1]).astype(np.float32))
        if add_stats:
            stats_rows = np.array([
                torque.mean(axis=0),
                torque.std(axis=0),
                np.sqrt(np.mean(torque.astype(np.float64) ** 2, axis=0)),
                torque.min(axis=0),
                torque.max(axis=0),
                torque.ptp(axis=0),
            ], dtype=np.float32)                                     # (6, n_torque)
            parts.append(np.tile(stats_rows.T.reshape(1, -1), (T, 1)).astype(np.float32))

    if add_abs:
        parts.append(np.abs(raw).astype(np.float32))

    return np.concatenate(parts, axis=1).astype(np.float32)


# ── 헬퍼 — standard 채널 증강 (_add_feature_channels 와 동일 로직) ─────────────
def _augment_standard(raw: np.ndarray, meta: dict) -> np.ndarray:
    """
    augment_mode="standard" 전처리 — train_dl_model.py _extract_windows 와 동일한 순서/방식.

    채널 순서: [raw | abs(선택) | derivative(선택) | fft(선택)]
    ※ 학습(abs→deriv→fft 순)과 반드시 일치해야 norm_mean/norm_std 인덱스가 맞음.

    FFT 리사이징: np.interp 선형 보간 (학습과 동일, tile 아님).
    """
    add_fft   = bool(meta.get("add_fft_channels",        True))
    add_deriv = bool(meta.get("add_derivative_channels", True))
    add_abs   = bool(meta.get("add_abs_channels",        False))

    T, C = raw.shape
    parts: List[np.ndarray] = [raw]

    # ① abs — 학습 _extract_windows 와 동일한 첫 번째 extras
    if add_abs:
        parts.append(np.abs(raw).astype(np.float32))

    # ② derivative — 두 번째 extras
    if add_deriv:
        parts.append(np.diff(raw, axis=0, prepend=raw[:1]).astype(np.float32))

    # ③ fft — 세 번째 extras (마지막)
    #    선형 보간으로 T//2+1 → T 로 리사이즈 (학습과 동일, tile 아님)
    if add_fft:
        fft_mag = np.abs(np.fft.rfft(raw, axis=0)).astype(np.float32)  # (T//2+1, C)
        fft_len = fft_mag.shape[0]
        x_old   = np.linspace(0.0, 1.0, fft_len)
        x_new   = np.linspace(0.0, 1.0, T)
        fft_resized = np.stack(
            [np.interp(x_new, x_old, fft_mag[:, c]) for c in range(C)],
            axis=1,
        ).astype(np.float32)
        parts.append(fft_resized)

    return np.concatenate(parts, axis=1).astype(np.float32) if len(parts) > 1 else parts[0]


# ── 헬퍼 — 채널 증강 + 정규화 (학습과 동일한 전처리) ─────────────────────────
def _preprocess_window(
    window_flat,      # list[float] | flat np.ndarray
    window_size: int,
    n_channels:  int,
    meta:        dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    meta["augment_mode"] 에 따라 학습과 동일한 전처리를 적용합니다.

    "mixed"    → _augment_mixed()    — normalize=False (RAW 진폭 보존)
    "standard" → _augment_standard() — per-window z-score 후 증강

    Returns:
        raw_arr:  (1, T, n_raw)   — 원본 (RMS 계산용)
        proc_arr: (1, T, n_model) — 증강+정규화 완료 (모델 입력)
    """
    raw     = np.array(window_flat, dtype=np.float32).reshape(window_size, n_channels)
    raw_arr = raw[np.newaxis, :, :].astype(np.float32)

    # augment_mode: meta에 저장된 값 사용 (None이면 "standard" 기본)
    # None은 str() 시 "none"이 되므로 명시적으로 처리
    _aug_raw     = meta.get("augment_mode")
    augment_mode = str(_aug_raw).lower() if _aug_raw is not None else "standard"
    # do_augment: 개별 채널 플래그로 결정 (add_augmented_channels는 미사용)
    do_augment   = bool(
        meta.get("add_fft_channels",        False) or
        meta.get("add_derivative_channels", False) or
        meta.get("add_abs_channels",        False)
    )

    nm = meta.get("norm_mean")
    ns = meta.get("norm_std")

    if augment_mode == "mixed":
        aug = _augment_mixed(raw, meta) if do_augment else raw
        # ── 학습 시 global_norm 적용됨 → 추론도 동일하게 적용 ────────────────
        # 학습:  raw → global_norm(mean, std) → AE 학습
        # 기존:  raw → (norm 없음) → AE 입력  ← 스케일 불일치로 score 폭증
        # 수정:  raw → global_norm            → AE 입력 (학습과 동일)
        if nm is not None and ns is not None:
            m = np.array(nm, dtype=np.float32).reshape(1, 1, -1)
            s = np.where(np.array(ns, dtype=np.float32).reshape(1, 1, -1) < 1e-8,
                         1.0, np.array(ns, dtype=np.float32).reshape(1, 1, -1))
            proc_arr = ((aug[np.newaxis] - m) / s).astype(np.float32)
        else:
            proc_arr = aug[np.newaxis, :, :].astype(np.float32)
    else:
        # standard 모드 — global norm 유무에 따라 경로 분기
        if nm is not None and ns is not None:
            # ── AE 모드: raw → augment → global_norm (학습과 동일 순서) ──────
            # 주의: z-score를 먼저 적용하면 raw 스케일 기준 nm/ns가 맞지 않아
            #       재구성 오차가 수십 배 폭증함 (이중 정규화 오류)
            aug = _augment_standard(raw, meta) if do_augment else raw
            m = np.array(nm, dtype=np.float32).reshape(1, 1, -1)
            s = np.where(np.array(ns, dtype=np.float32).reshape(1, 1, -1) < 1e-8,
                         1.0, np.array(ns, dtype=np.float32).reshape(1, 1, -1))
            proc_arr = ((aug[np.newaxis] - m) / s).astype(np.float32)
        else:
            # ── CLS 모드: per-window z-score → augment ────────────────────────
            norm_raw = _zscore_ch(raw)
            aug      = _augment_standard(norm_raw, meta) if do_augment else norm_raw
            proc_arr = aug[np.newaxis, :, :].astype(np.float32)

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
                if (_models_root() / fname).exists():
                    files.append(fname)
                    break
        for fname in fallbacks:
            if (_models_root() / fname).exists():
                files.append(fname)
        available_ae[st] = files

    # 사용 가능한 CLS 모델 스캔 (accel / torque / combined)
    available_cls: dict = {}
    for st in list(_CLS_CANDIDATES.keys()):
        fallbacks = _CLS_CANDIDATES.get(st, [])
        files = []
        for ax in range(_MAX_AXIS_SCAN):
            for fname in _per_axis_cls_candidates(st, ax):
                if (_models_root() / fname).exists():
                    files.append(fname)
                    break
        for fname in fallbacks:
            if (_models_root() / fname).exists() and fname not in files:
                files.append(fname)
        if files:
            available_cls[st] = files

    return {
        "status":               "ok",
        "active_profile":       _active_profile,
        "loaded_models":        loaded,
        "available_ae_models":  available_ae,
        "available_cls_models": available_cls,
        "models_root":          str(_models_root()),
    }


@app.get("/models/reload")
def reload_models():
    _sessions.clear()
    return {"status": "reloaded", "message": "다음 /predict 호출 시 재로드됩니다."}


@app.get("/model_info")
def model_info():
    """각 sensor_type의 모델 메타 정보(window_size, activity_threshold 등)를 반환합니다.

    전역 모델 + per-axis 모델 모두 스캔해 **최대 window_size** 와
    **최솟값 activity_threshold** (0 = 게이팅 미적용)를 반환합니다.
    클라이언트는 window_size 만큼 활성 행을 버퍼링하고,
    activity_threshold 미만 행은 제외해야 학습 분포와 일치합니다.
    (ONNX 모델을 새로 로드하지 않으므로 빠르게 응답합니다.)
    """
    def _read_meta(fname: str) -> dict:
        """메타 파일을 읽어 dict 반환. 없으면 {}."""
        meta_path = _models_root() / fname.replace(".onnx", "_meta.json")
        if not meta_path.exists():
            return {}
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    result = {}
    all_sensor_types = list(_FALLBACK_CANDIDATES.keys())  # ["accel", "torque", ...]

    for sensor_type in all_sensor_types:
        max_ws: int   = 0
        # activity_threshold: 모델별로 다를 수 있으므로 최솟값(가장 엄격하지 않은 쪽) 사용.
        # 0.0 = 게이팅 미적용 모델이 하나라도 있으면 클라이언트도 필터 비활성화.
        min_act: float = float("inf")   # 아직 못 읽으면 inf
        source: str   = "default"

        def _update(meta: dict, src: str):
            nonlocal max_ws, min_act, source
            ws  = int(meta.get("window_size", 0))
            act = float(meta.get("activity_threshold", 0.0))
            if ws > max_ws:
                max_ws = ws
                source = src
            # activity_threshold=0 이면 게이팅 없음 → 클라이언트도 비활성화
            if act == 0.0:
                min_act = 0.0
            elif act < min_act:
                min_act = act

        # ① 캐시에 로드된 세션 스캔 (전역 + per-axis)
        for key, (_, meta) in list(_sessions.items()):
            if not key.startswith(sensor_type):
                continue
            _update(meta, "cache")

        # ② 메타 파일 스캔 — 전역 폴백 후보
        for fname in _FALLBACK_CANDIDATES.get(sensor_type, []):
            _update(_read_meta(fname), "meta_file")

        # ③ per-axis 메타 파일 스캔 (캐시 미로드 축 포함)
        for ax in range(_MAX_AXIS_SCAN):
            for fname in _per_axis_candidates(sensor_type, ax):
                _update(_read_meta(fname), "meta_file_peraxis")
                break   # 해당 축의 최우선 후보만

        result[sensor_type] = {
            "window_size":        max_ws  if max_ws  > 0             else 512,
            "activity_threshold": min_act if min_act < float("inf")  else 0.0,
            "source":             source  if max_ws  > 0             else "default",
        }

    return result


# ── 프로파일 관리 ────────────────────────────────────────────────────────────

class ProfileActivateRequest(BaseModel):
    profile: str


@app.get("/profiles")
def list_profiles():
    """사용 가능한 프로파일 목록을 반환합니다.

    프로파일 = _MODELS_BASE 하위 디렉토리.
    각 디렉토리에 profile_info.json 이 있으면 name/description 표시.
    없으면 디렉토리명 그대로 사용.
    """
    profiles = []

    # 베이스 경로 자체 (플랫 구조 레거시) 도 "default" 로 포함
    # onnx 파일이 1개 이상이거나 profile_info.json 이 있는 디렉토리만 프로파일로 인식
    subdirs = sorted(
        [d for d in _MODELS_BASE.iterdir()
         if d.is_dir()
         and not d.name.startswith(".")
         and (len(list(d.glob("*.onnx"))) > 0 or (d / "profile_info.json").exists())]
    ) if _MODELS_BASE.is_dir() else []

    # 서브디렉토리가 없으면 베이스 자체를 "default" 프로파일로 취급
    if not subdirs:
        onnx_count = len(list(_MODELS_BASE.glob("*.onnx"))) if _MODELS_BASE.is_dir() else 0
        profiles.append({
            "name": "default",
            "label": "default",
            "description": "기본 모델 (플랫 구조)",
            "model_count": onnx_count,
        })
    else:
        for d in subdirs:
            info_path = d / "profile_info.json"
            if info_path.exists():
                try:
                    info = json.loads(info_path.read_text(encoding="utf-8"))
                except Exception:
                    info = {}
            else:
                info = {}
            onnx_count = len(list(d.glob("*.onnx")))
            profiles.append({
                "name":        d.name,
                "label":       info.get("label",       d.name),
                "description": info.get("description", ""),
                "model_count": onnx_count,
                "created":     info.get("created",     ""),
            })

    return {"profiles": profiles, "active": _active_profile}


@app.post("/profiles/activate")
def activate_profile(req: ProfileActivateRequest):
    """활성 프로파일을 전환하고 모델 캐시를 클리어합니다.

    profile = "default" 는 항상 허용 (서브디렉토리 없어도 OK → 플랫 폴백 사용).
    그 외는 _MODELS_BASE/{profile} 디렉토리가 존재해야 합니다.
    """
    global _active_profile

    if req.profile != "default":
        target = _MODELS_BASE / req.profile
        if not target.is_dir():
            raise HTTPException(
                status_code=404,
                detail=f"프로파일 없음: '{req.profile}'. "
                       f"사용 가능: {[d.name for d in _MODELS_BASE.iterdir() if d.is_dir()]}",
            )

    old_profile     = _active_profile
    _active_profile = req.profile
    _sessions.clear()   # 캐시 클리어 → 다음 /predict 시 새 경로에서 로드

    print(f"[inference] 프로파일 전환: {old_profile!r} → {_active_profile!r}  "
          f"모델 경로: {_models_root()}", flush=True)

    return {
        "status":          "ok",
        "previous_profile": old_profile,
        "active_profile":  _active_profile,
        "models_root":     str(_models_root()),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    sess, meta = _load_model(req.sensor_type, req.axis)
    if sess is None:
        # 사용 가능한 모델 목록 수집
        avail: List[str] = []
        if req.axis is not None:
            for fname in _per_axis_candidates(req.sensor_type, req.axis):
                if (_models_root() / fname).exists():
                    avail.append(fname)
        for fname in _FALLBACK_CANDIDATES.get(req.sensor_type, []):
            if (_models_root() / fname).exists():
                avail.append(fname)
        raise HTTPException(
            status_code=404,
            detail=(
                f"sensor_type='{req.sensor_type}' axis={req.axis} 모델 없음. "
                f"사용 가능: {avail if avail else '없음 — Airflow 학습을 먼저 실행하세요.'}"
            ),
        )

    # ── 모델 기대 채널 수 검증 ────────────────────────────────────────────────
    # meta["n_channels"]: 증강 후 ONNX 모델 입력 채널 수 (add_fft/deriv 포함)
    # meta["channels"]:   학습에 사용된 raw CSV 채널 목록 → 클라이언트가 보내는 채널 수
    # 클라이언트는 항상 raw 채널을 전송하고 서버가 증강을 적용하므로
    # raw 채널 수 기준으로 검증한다.
    raw_channels     = meta.get("channels", [])
    raw_n_channels   = len(raw_channels) if raw_channels else meta.get("n_channels")
    if raw_n_channels and req.n_channels != raw_n_channels:
        # per-axis 요청인데 폴백 전역 모델(다채널)로 떨어진 경우:
        # 채널 수가 다른 더 적합한 모델을 재탐색하지 않고 에러 반환
        # (→ 해결책: Airflow per-axis 토크 학습 실행)
        raise HTTPException(
            status_code=400,
            detail=(
                f"채널 수 불일치: 모델 raw={raw_n_channels}ch, 요청={req.n_channels}ch. "
                f"모델 학습 채널: {raw_channels}. "
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
    thr = float(meta.get("threshold", 0.1))

    # ── 활동성 게이팅 ────────────────────────────────────────────────────────
    # 학습 시 적용한 활동성 임계값보다 낮으면 정지 상태로 간주하고 점수=0.
    # 학습 분포에 정지 구간이 포함되지 않았으므로(=구동만 학습) 추론 시에도
    # 같은 게이팅을 적용해 정지 중 false alarm 을 방지하고, 모델 호출 비용도 절감.
    activity_thr_val = float(meta.get("activity_threshold", 0.0))
    if activity_thr_val > 0.0:
        # raw_arr shape (1, T, C_raw) — 정규화 전 원본
        # 활동성 = max over channels of std(time-axis)  (학습 시와 동일 공식)
        activity = float(raw_arr[0].astype(np.float64).std(axis=0).max())
        if activity < activity_thr_val:
            # 정지 윈도우 — 정상, 점수 0
            return False, 0.0, 0.0, thr

    input_name = sess.get_inputs()[0].name
    recon  = sess.run(None, {input_name: proc_arr})[0]

    # ── 재구성 오차: 학습 임계값 캘리브레이션과 동일한 공식 사용 ────────────────
    # 학습 시: per_sample_mae = abs(model(x)-x).mean(dim=(1,2)) = 전체 (T,C) 평균
    # 추론 시: 동일하게 전체 (T,C) 평균 → 임계값 비교가 일관성 유지됨
    err_per_step = np.abs(proc_arr - recon).mean(axis=-1)[0]  # (T,) 시간축 오차
    mae_mean = float(err_per_step.mean())
    mae      = mae_mean

    rms       = float(np.sqrt(np.mean(raw_arr.astype(np.float64) ** 2)))
    rms_thr   = float(meta.get("rms_thr",  float("inf")))
    rms_mean  = float(meta.get("rms_mean", 0.0))
    rms_std_m = meta.get("rms_std")
    if rms_thr < 1e30:
        denom    = float(rms_std_m) if rms_std_m else max(rms_thr - rms_mean, 1e-8)
        rms_norm = max(0.0, (rms - rms_mean) / denom)
    else:
        rms_norm = 0.0

    # ── log₂(1 + ratio) 스코어 ───────────────────────────────────────────────
    # 선형 mae/thr 은 normalize=False 모델(토크 등)에서 mae_thr 이 매우 작아지면
    # 이상 시 수백~수천까지 폭증하여 시각화/비교가 불가능해짐.
    #
    # log₂(1 + ratio) 변환:
    #   ratio = mae / thr
    #   ratio = 0.0  → score_mae = 0.000  (완전 정상)
    #   ratio = 1.0  → score_mae = 1.000  ← 임계값 그대로 보존
    #   ratio = 10   → score_mae = 3.459
    #   ratio = 100  → score_mae = 6.658
    #   ratio = 1000 → score_mae = 9.967
    #
    # is_anomaly 판정은 변환 후 score_normed >= 1.0 (동일 기준 유지).
    import math
    ratio     = mae / max(thr, 1e-8)
    score_mae = math.log2(1.0 + ratio)

    # rms 기여도도 동일 스케일로 압축
    score_rms = math.log2(1.0 + rms_norm) if rms_norm > 0.0 else 0.0

    score_normed = score_mae + _RMS_WEIGHT * score_rms
    is_anomaly   = score_normed >= _ANOMALY_THRESHOLD
    return is_anomaly, score_normed, mae, thr


# ── /predict/combined ─────────────────────────────────────────────────────────
@app.post("/predict/combined", response_model=CombinedPredictResponse)
def predict_combined(req: CombinedPredictRequest):
    """AE 이상탐지 + CLS 결함진단을 한 번에 수행합니다.

    sensor_type="accel" | "torque":
      AE 모델 필수 + CLS 모델 선택 (없으면 cls_available=False).

    sensor_type="combined":
      CLS 전용 — AE 모델 없이 CLS 단독 운영.
      is_anomaly = (예측 클래스 != "normal").
      CLS도 없으면 404.

    window 전처리는 각 모델의 meta["augment_mode"] 에 따라 독립 수행.
    """
    expected = req.window_size * req.n_channels
    if len(req.window) != expected:
        raise HTTPException(
            status_code=400,
            detail=f"window 길이 {len(req.window)} ≠ {req.window_size}×{req.n_channels}={expected}",
        )

    # ══════════════════════════════════════════════════════════════════════════
    # "combined" 센서 타입: AE(있으면) + CLS 경로
    # ══════════════════════════════════════════════════════════════════════════
    if req.sensor_type == "combined":
        # ── AE 모델 로드 시도 (ae_combined_ax{n}.onnx) ──────────────────────
        ae_sess, ae_meta = _load_model(req.sensor_type, req.axis)

        # ── CLS 모델 로드 ────────────────────────────────────────────────────
        cls_sess, cls_meta = _load_cls_model(req.sensor_type, req.axis)

        if ae_sess is None and cls_sess is None:
            avail: List[str] = []
            if req.axis is not None:
                for fn in _per_axis_cls_candidates(req.sensor_type, req.axis) + \
                           _per_axis_candidates(req.sensor_type, req.axis):
                    if (_models_root() / fn).exists():
                        avail.append(fn)
            raise HTTPException(
                status_code=404,
                detail=(
                    f"combined AE/CLS 모델 없음 (axis={req.axis}). "
                    f"사용 가능: {avail if avail else '없음 — Airflow ae_combined/train_combined 을 먼저 실행하세요.'}"
                ),
            )

        # ── AE 이상탐지 ──────────────────────────────────────────────────────
        ae_model_file  = None
        is_anomaly_ae  = False
        anomaly_score  = 0.0
        raw_mae        = None
        raw_threshold  = None
        if ae_sess is not None:
            try:
                raw_arr, proc_arr = _preprocess_window(
                    req.window, req.window_size, req.n_channels, ae_meta)
                is_anomaly_ae, score_ae, mae, thr = _ae_score(
                    proc_arr, raw_arr, ae_sess, ae_meta, req.sensor_type)
                ae_model_file = ae_meta.get("source_file")
                is_anomaly    = is_anomaly_ae
                anomaly_score = round(score_ae, 6)
                raw_mae       = round(mae, 6)
                raw_threshold = round(thr, 6)
            except Exception as ex:
                ae_sess = None  # AE 실패 시 CLS 결과로 대체

        # ── CLS 결함진단 ─────────────────────────────────────────────────────
        cls_available  = False
        cls_model_file = None
        cls_class_name = None
        cls_confidence = None
        cls_is_fault   = None
        if cls_sess is not None:
            try:
                cls_ws = int(cls_meta.get("window_size", req.window_size))
                if cls_ws != req.window_size:
                    need = cls_ws * req.n_channels
                    if len(req.window) < need:
                        cls_sess = None
                    else:
                        eff_win, eff_ws = list(req.window[-need:]), cls_ws
                else:
                    eff_win, eff_ws = req.window, req.window_size

                if cls_sess is not None:
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

                    cls_available  = True
                    cls_model_file = cls_meta.get("source_file")
                    cls_class_name = class_name
                    cls_confidence = round(confidence, 6)
                    cls_is_fault   = is_fault

                    # AE 없을 때만 CLS confidence를 anomaly_score 대용으로 사용
                    if ae_sess is None:
                        anomaly_score = round(confidence if is_fault else (1.0 - confidence), 6)
                        is_anomaly    = is_fault
            except Exception:
                pass

        return CombinedPredictResponse(
            sensor_type    = req.sensor_type,
            axis           = req.axis,
            ae_model_file  = ae_model_file,
            is_anomaly     = is_anomaly,
            anomaly_score  = anomaly_score,
            threshold      = 1.0,
            raw_mae        = raw_mae,
            raw_threshold  = raw_threshold,
            cls_available  = cls_available,
            cls_model_file = cls_model_file,
            cls_class_name = cls_class_name,
            cls_confidence = cls_confidence,
            cls_is_fault   = cls_is_fault,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # "accel" | "torque": AE 필수 + CLS 선택
    # ══════════════════════════════════════════════════════════════════════════
    # ── AE 모델 로드 ─────────────────────────────────────────────
    ae_sess, ae_meta = _load_model(req.sensor_type, req.axis)
    if ae_sess is None:
        avail: List[str] = []
        if req.axis is not None:
            for fn in _per_axis_candidates(req.sensor_type, req.axis):
                if (_models_root() / fn).exists():
                    avail.append(fn)
        for fn in _FALLBACK_CANDIDATES.get(req.sensor_type, []):
            if (_models_root() / fn).exists():
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
        # ── AE 전처리 (채널 증강 + 정규화) ───────────────────────
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
                    cls_proc = ae_proc_arr  # 길이 부족 시 AE 전처리 결과 재사용
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


# ── /config  (런타임 스코어링 파라미터 조회/변경) ────────────────────────────

class ServerConfigRequest(BaseModel):
    rms_weight:        Optional[float] = None   # None 이면 변경 안 함
    anomaly_threshold: Optional[float] = None


@app.get("/config")
def get_config():
    """현재 스코어링 파라미터를 반환합니다."""
    return {
        "rms_weight":        _RMS_WEIGHT,
        "anomaly_threshold": _ANOMALY_THRESHOLD,
    }


@app.post("/config")
def set_config(req: ServerConfigRequest):
    """스코어링 파라미터를 변경합니다 (재시작 전까지 유효).

    - rms_weight:        RMS 기여 가중치 (0.0 ~ 1.0)
    - anomaly_threshold: 이상 판정 기준값 (log₂ 스코어 ≥ 이 값 → 이상)
    """
    global _RMS_WEIGHT, _ANOMALY_THRESHOLD
    changed = []
    if req.rms_weight is not None:
        _RMS_WEIGHT = float(req.rms_weight)
        changed.append(f"rms_weight={_RMS_WEIGHT}")
    if req.anomaly_threshold is not None:
        _ANOMALY_THRESHOLD = float(req.anomaly_threshold)
        changed.append(f"anomaly_threshold={_ANOMALY_THRESHOLD}")
    if changed:
        print(f"[config] 변경: {', '.join(changed)}", flush=True)
    return get_config()


# ── 엔트리포인트 ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print(f"[inference] 모델 루트: {_models_root()}  (프로파일: {_active_profile!r})", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
