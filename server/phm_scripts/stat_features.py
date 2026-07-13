"""
stat_features.py — 통계·주파수 기반 신호 특징 추출 (가속도 x/y/z + magnitude)

train_isoforest_accel.py(학습)와 inference_server.py(실시간 추론)가 이 모듈을
**공유 import** 합니다. 두 스크립트가 각자 특징 추출 코드를 복사해서 들고 있으면
한쪽만 수정됐을 때 학습/추론 전처리가 어긋나 스코어가 무의미해지므로,
반드시 이 파일 하나만 수정하고 양쪽 다 여기서 import 해서 씁니다.

원본 방법론: D:\\Dev\\hvs\\PycharmProjects\\csv_viewer\\train.py
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import kurtosis, skew

PEAK_COUNT = 4
MIN_FREQ_HZ = 1.0


def extract_frequency_peaks(
    v: np.ndarray,
    fs: float,
    peak_count: int = PEAK_COUNT,
    min_freq_hz: float = MIN_FREQ_HZ,
    max_freq_hz: float = None,
) -> List[Tuple[float, float]]:
    """DC 제거 + Hann window + 단측 FFT로 주요 주파수 피크를 추출한다."""
    v = np.asarray(v, dtype=float)
    n = len(v)

    if n < 4 or fs <= 0:
        return [(0.0, 0.0)] * peak_count

    centered = v - np.mean(v)
    window = np.hanning(n)
    window_sum = np.sum(window)

    if window_sum <= 1e-12:
        return [(0.0, 0.0)] * peak_count

    spectrum = np.fft.rfft(centered * window)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    # Hann window의 coherent gain을 보정한 단측 진폭
    amplitudes = 2.0 * np.abs(spectrum) / window_sum
    if len(amplitudes) > 0:
        amplitudes[0] = 0.0
    if n % 2 == 0 and len(amplitudes) > 1:
        amplitudes[-1] *= 0.5

    valid = freqs >= min_freq_hz
    if max_freq_hz is not None:
        valid &= freqs <= max_freq_hz

    candidate_indices = np.flatnonzero(valid)
    if len(candidate_indices) == 0:
        return [(0.0, 0.0)] * peak_count

    # 인접 bin이 같은 피크로 중복 선택되지 않도록 국부 최대점만 후보로 사용
    local_peaks = []
    for idx in candidate_indices:
        left = amplitudes[idx - 1] if idx > 0 else -np.inf
        right = amplitudes[idx + 1] if idx + 1 < len(amplitudes) else -np.inf
        if amplitudes[idx] >= left and amplitudes[idx] >= right:
            local_peaks.append(idx)

    # 진폭이 큰 국부 피크 순서로 선택한다. 실제 피크가 부족하면 0으로 채운다.
    ranked = sorted(local_peaks, key=lambda i: amplitudes[i], reverse=True)

    peaks = [(float(freqs[i]), float(amplitudes[i])) for i in ranked[:peak_count]]
    peaks.extend([(0.0, 0.0)] * (peak_count - len(peaks)))
    return peaks


_ORDINAL = {1: "st", 2: "nd", 3: "rd"}


def add_signal_features(
    row: Dict[str, float],
    prefix: str,
    v: np.ndarray,
    fs: float,
    peak_count: int = PEAK_COUNT,
    min_freq_hz: float = MIN_FREQ_HZ,
) -> None:
    v = np.asarray(v, dtype=float)
    abs_v = np.abs(v)
    abs_mean = np.mean(abs_v)
    abs_max = np.max(abs_v)
    rms = np.sqrt(np.mean(v ** 2))

    row[f"{prefix}_absmean"] = float(abs_mean)
    row[f"{prefix}_absmax"] = float(abs_max)
    row[f"{prefix}_rms"] = float(rms)
    row[f"{prefix}_p2p"] = float(np.ptp(v))
    if np.std(v) <= 1e-12:
        row[f"{prefix}_skewness"] = 0.0
        row[f"{prefix}_kurtosis"] = 0.0
    else:
        row[f"{prefix}_skewness"] = float(skew(v, bias=False))
        row[f"{prefix}_kurtosis"] = float(kurtosis(v, fisher=False, bias=False))
    row[f"{prefix}_crest"] = float(abs_max / (rms + 1e-12))
    row[f"{prefix}_impulse"] = float(abs_max / (abs_mean + 1e-12))
    row[f"{prefix}_shape"] = float(rms / (abs_mean + 1e-12))

    peaks = extract_frequency_peaks(v, fs, peak_count=peak_count, min_freq_hz=min_freq_hz)
    for rank, (freq, amp) in enumerate(peaks, start=1):
        ordinal = _ORDINAL.get(rank, "th")
        row[f"{prefix}_{rank}{ordinal}_freq"] = freq
        row[f"{prefix}_{rank}{ordinal}_amp"] = amp


def extract_features_from_segment(
    seg: np.ndarray,
    fs: float,
    axes: Sequence[str] = ("x", "y", "z"),
    peak_count: int = PEAK_COUNT,
    min_freq_hz: float = MIN_FREQ_HZ,
) -> Dict[str, float]:
    """seg: shape (T, len(axes)) — 각 축 + magnitude(=norm(axes)) 특징을 계산한다.

    반환 dict의 키 순서는 axes 순서 → "mag" 순으로 결정적이다. 학습/추론 양쪽
    모두 meta["feature_names"]로 이 순서를 명시적으로 저장/조회하므로, 이 함수의
    내부 순서가 바뀌어도 벡터 정렬 자체는 깨지지 않는다(각 키를 이름으로 조회).
    """
    row: Dict[str, float] = {}

    for i, axis in enumerate(axes):
        add_signal_features(row, axis, seg[:, i], fs, peak_count=peak_count, min_freq_hz=min_freq_hz)

    mag = np.linalg.norm(seg, axis=1)
    add_signal_features(row, "mag", mag, fs, peak_count=peak_count, min_freq_hz=min_freq_hz)

    for key, value in row.items():
        if not np.isfinite(value):
            row[key] = 0.0

    return row


def feature_vector(row: Dict[str, float], feature_names: Sequence[str]) -> List[float]:
    """저장된 feature_names 순서대로 row에서 값을 뽑아 벡터를 만든다.

    학습 시 저장한 순서와 추론 시 조회 순서를 이 함수 하나로 고정해
    두 시점의 특징 벡터 정렬이 어긋나는 것을 방지한다.
    """
    return [float(row[k]) for k in feature_names]
