"""
train_dl_model.py - PHM 대시보드용 1D-CNN 딥러닝 모델 학습 및 ONNX 변환 스크립트

사용법:
    python train_dl_model.py --params <params_json_path>

params JSON 구조:
{
    "data_dir": "C:/Data/PHM_Logs/raw",      // CSV 디렉터리 (재귀 탐색), OR
    "csv_files": [                            // 명시적 파일+레이블 목록
        {"path": "C:/Data/.../normal.csv", "label": "normal"}
    ],
    "output": "C:/Models/cnn1d_fault.onnx",
    "channels": ["X", "Y", "Z"],             // 입력 채널 컬럼명
    "label_column": "Label",                 // 레이블 컬럼명 (기본 "Label")
    "class_names": ["normal", "fault", "bearing_fault", "gear_fault", "imbalance", "looseness"],
    "window_size": 256,
    "stride": 128,
    "epochs": 50,
    "batch_size": 32,
    "lr": 0.001,
    "val_split": 0.2,
    "seed": 42,
    "mlflow_tracking_uri": "http://localhost:5000",  // 선택
    "mlflow_experiment": "PHM-DL"                    // 선택
}

결과: stdout에 JSON 출력 {"info": "...", "accuracy": 0.95, "epochs": 45}
      output 경로에 ONNX 파일 및 _meta.json 사이드카 생성

의존성: 첫 실행 시 자동 설치됨 (pip 필요)
    torch  numpy  onnx  scikit-learn
"""

import sys
import subprocess


def _ensure_packages() -> None:
    """필요한 패키지가 없으면 자동으로 pip 설치합니다.

    torch는 CPU 전용 wheel을 사용합니다. GPU 환경이라면 직접 설치하십시오.
    """
    # (import_name, pip_name, extra_args)
    REQUIRED = [
        ("numpy",      "numpy",       []),
        ("sklearn",    "scikit-learn",[]),
        ("onnx",       "onnx",        []),
        ("onnxscript", "onnxscript",  []),  # PyTorch 2.6+ ONNX exporter 의존성
    ]
    missing_general = []
    for import_name, pip_name, _ in REQUIRED:
        try:
            __import__(import_name)
        except ImportError:
            missing_general.append(pip_name)

    if missing_general:
        print(f"[setup] 패키지 자동 설치: {', '.join(missing_general)}", file=sys.stderr)
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet"] + missing_general,
                stdout=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as e:
            print(f"[setup] 설치 실패: {e}", file=sys.stderr)
            sys.exit(1)

    # torch는 별도 처리: CPU wheel 인덱스 사용
    try:
        __import__("torch")
    except ImportError:
        print(
            "[setup] torch가 없습니다. CPU 버전을 설치합니다 (수 분 소요될 수 있습니다)...",
            file=sys.stderr,
        )
        try:
            subprocess.check_call(
                [
                    sys.executable, "-m", "pip", "install", "--quiet",
                    "torch",
                    "--index-url", "https://download.pytorch.org/whl/cpu",
                ],
                stdout=subprocess.DEVNULL,
            )
            print("[setup] torch 설치 완료", file=sys.stderr)
        except subprocess.CalledProcessError as e:
            print(f"[setup] torch 설치 실패: {e}", file=sys.stderr)
            sys.exit(1)

    print("[setup] 모든 패키지 준비 완료", file=sys.stderr)


_ensure_packages()

# ── 이후 임포트는 패키지 설치 보장 후 ────────────────────────────────────────
import argparse
import json
import os
import math
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit


# ── 데이터 로딩 ──────────────────────────────────────────────────────────────

def load_params(params_path: str) -> dict:
    """UTF-8(BOM 포함/미포함 모두 처리) JSON 파라미터 파일을 읽습니다."""
    with open(params_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def _zscore_normalize(window: np.ndarray) -> np.ndarray:
    """윈도우를 채널별 z-score 정규화합니다 (C# ZScoreInPlace와 동일 로직).

    Args:
        window: shape (T, C) float32 배열

    Returns:
        정규화된 (T, C) float32 배열. 표준편차 0인 채널은 그대로 유지.
    """
    mean = window.mean(axis=0, keepdims=True)      # (1, C)
    std = window.std(axis=0, keepdims=True)        # (1, C)
    std = np.where(std < 1e-8, 1.0, std)
    return ((window - mean) / std).astype(np.float32)


def _extract_windows(
    signal: np.ndarray,
    label_int: int,
    window_size: int,
    stride: int,
    normalize: bool = True,
) -> List[Tuple[np.ndarray, int]]:
    """슬라이딩 윈도우로 (window_array, label_int) 튜플 목록을 생성합니다.

    Args:
        signal:     shape (N, C) float32 배열
        label_int:  정수 레이블
        window_size: 윈도우 샘플 수 T
        stride:     슬라이딩 스트라이드
        normalize:  True이면 윈도우별 z-score 정규화 (CLS 용).
                    AE는 False — 절대 진폭/에너지 정보를 보존해야 이상 탐지가 가능.

    Returns:
        [(window_np, label_int), ...] — 각 window_np shape (T, C)
    """
    results: List[Tuple[np.ndarray, int]] = []
    n_samples = signal.shape[0]
    for start in range(0, n_samples - window_size + 1, stride):
        window = signal[start : start + window_size].copy()
        if normalize:
            window = _zscore_normalize(window)
        results.append((window, label_int))
    return results


def _compute_global_stats(
    windows: List[Tuple[np.ndarray, int]]
) -> Tuple[np.ndarray, np.ndarray]:
    """윈도우 목록 전체로부터 채널별 전역 mean/std를 계산합니다.

    Returns:
        (mean, std) 각 shape (C,) float64
    """
    all_vals = np.concatenate([w for w, _ in windows], axis=0)  # (N*T, C)
    mean = all_vals.mean(axis=0)
    std  = all_vals.std(axis=0)
    std  = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float64), std.astype(np.float64)


def _apply_global_norm(
    windows: List[Tuple[np.ndarray, int]],
    mean: np.ndarray,
    std: np.ndarray,
) -> List[Tuple[np.ndarray, int]]:
    """채널별 전역 통계로 모든 윈도우를 정규화합니다."""
    return [
        (((w - mean) / std).astype(np.float32), lbl)
        for w, lbl in windows
    ]


def _detect_sensor_type_from_headers(csv_path: str) -> str:
    """CSV 헤더를 읽어 센서 타입을 추론합니다.

    Returns:
        "combined" — x/y/z 가속도 AND Trq 컬럼이 모두 있는 경우
        "torque"   — Trq 관련 컬럼이 있고 x/y/z 가속도 컬럼이 없는 경우
        "accel"    — x, y, z 컬럼만 있는 경우
        ""         — 판별 불가
    """
    import csv as _csv
    for enc in ("utf-8-sig", "cp949", "utf-8", "latin-1"):
        try:
            with open(csv_path, newline="", encoding=enc, errors="replace") as f:
                reader = _csv.reader(f)
                headers = [h.strip().lower() for h in (next(reader, []))]
            if not headers:
                continue
            has_trq = any("trq" in h or "vel(mm" in h or "pos(mm" in h for h in headers)
            has_xyz = any(h in ("x", "y", "z") for h in headers)
            if has_trq and has_xyz:
                return "combined"   # 통합 CSV (accel + torque 동시 수집)
            if has_trq:
                return "torque"
            if has_xyz:
                return "accel"
            return ""  # 헤더 읽기는 성공했지만 판별 불가
        except Exception:
            continue
    return ""


def _resolve_channels(headers: List[str], channels: List[str]) -> List[str]:
    """축 번호 없는 채널명을 실제 헤더 컬럼명으로 매핑합니다.

    규칙:
      - 헤더에 그대로 있으면 그대로 사용 (e.g. 'x', 'y', 'z')
      - 'Ax' 접두사가 없는 이름(e.g. 'Trq(%)')은 'Ax\\d+_{channel}' 패턴으로
        매칭되는 컬럼을 *모두* 확장합니다 → 연결 축 수에 따라 자동 다축 지원
      - 매칭 컬럼이 없으면 원래 이름을 유지 (이후 오류로 처리)
    """
    import re as _re
    resolved = []
    for ch in channels:
        if ch in headers:
            resolved.append(ch)
        else:
            pat = _re.compile(r"^Ax\d+_" + _re.escape(ch) + r"$", _re.IGNORECASE)
            matches = [h for h in headers if pat.match(h)]
            if matches:
                # 축 번호 순서대로 정렬 (Ax0, Ax1, ...)
                matches.sort(key=lambda h: int(_re.search(r"\d+", h).group()))
                resolved.extend(matches)
            else:
                resolved.append(ch)  # 못 찾으면 원래 이름 유지 (오류로 처리됨)
    return resolved


def _read_signal_csv(
    path: str,
    channels: List[str],
    label_column: str,
    filter_op_column: Optional[str] = None,
) -> Tuple[List[np.ndarray], Optional[str]]:
    """단일 CSV 파일에서 연속 Pos 구간별로 분리된 신호 세그먼트와 레이블을 반환합니다.

    Op 컬럼이 Idle로 바뀌는 시점에 구간을 끊어, 서로 다른 동작 구간이
    하나의 연속 신호로 이어붙여지는 경계 오염을 방지합니다.

    Args:
        path           : CSV 파일 경로
        channels       : 사용할 컬럼명 목록
        label_column   : 레이블 컬럼명
        filter_op_column: 특정 Op 컬럼 명시 시 해당 컬럼만 Idle 판별에 사용.
                          None 이면 Op_Ax* / Op 컬럼 자동 감지 → 모두 Idle일 때 구간 끊기.

    Returns:
        (segments, label_str_or_None)
        segments : 각 연속 Pos 구간의 np.ndarray (shape (N_i, C)) 목록.
                   길이 0인 구간은 포함되지 않음.
    """
    import csv as _csv

    segments: List[np.ndarray] = []
    current_seg: List[List[float]] = []
    label_value: Optional[str] = None

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = _csv.DictReader(f)
        if reader.fieldnames is None:
            return [], None

        actual_channels = _resolve_channels(list(reader.fieldnames), channels)
        has_label = label_column in (reader.fieldnames or [])
        fieldnames_lower = [f.strip().lower() for f in (reader.fieldnames or [])]

        if filter_op_column:
            _foc_lower = filter_op_column.strip().lower()
            _op_cols = [reader.fieldnames[i] for i, f in enumerate(fieldnames_lower)
                        if f == _foc_lower]
        else:
            _op_cols = [reader.fieldnames[i] for i, f in enumerate(fieldnames_lower)
                        if f.startswith("op_ax") or f == "op"]

        def _flush_seg() -> None:
            if current_seg:
                segments.append(np.array(current_seg, dtype=np.float32))
                current_seg.clear()

        for row in reader:
            # ── Op 상태 판별 ──────────────────────────────────────────────
            is_idle = False
            if _op_cols:
                values_idle = [(row.get(c) or "").strip().lower() == "idle" for c in _op_cols]
                if filter_op_column:
                    is_idle = bool(values_idle and values_idle[0])
                else:
                    is_idle = all(values_idle)

            if is_idle:
                _flush_seg()   # 구간 종료
                continue

            # ── 값 파싱 ───────────────────────────────────────────────────
            try:
                vals = [float(row[c]) for c in actual_channels]
            except (KeyError, ValueError, TypeError):
                _flush_seg()   # 파싱 오류도 구간 끊기
                continue

            if any(math.isnan(v) or math.isinf(v) for v in vals):
                _flush_seg()
                continue

            current_seg.append(vals)

            if has_label and label_value is None:
                lv = row.get(label_column, "").strip()
                if lv:
                    label_value = lv

        _flush_seg()  # 파일 끝 처리

    return segments, label_value


def load_windows_from_dir(
    data_dir: str,
    channels: List[str],
    label_column: str,
    class_names: List[str],
    window_size: int,
    stride: int,
    sensor_type: str = "",
    normalize: bool = True,
    filter_op_column: Optional[str] = None,
) -> List[Tuple[np.ndarray, int]]:
    """디렉터리를 재귀 탐색해 모든 CSV에서 윈도우를 추출합니다.

    레이블 우선순위:
        1. CSV 내 label_column 값
        2. 경로 컴포넌트 중 class_names와 일치하는 것 (예: .../normal/Accel/...)
        3. CSV 부모 디렉터리명

    Args:
        data_dir    : 루트 디렉터리
        channels    : 채널 컬럼명 목록
        label_column: CSV 내 레이블 컬럼명
        class_names : 클래스명 → 정수 인덱스 매핑 기준
        window_size : 윈도우 크기
        stride      : 슬라이딩 스트라이드
        sensor_type : "accel" 이면 경로에 /Accel/ 포함 파일만, "torque" 이면 /Torque/ 만 처리

    Returns:
        [(window_np, label_int), ...]
    """
    name_to_id = {n.lower(): i for i, n in enumerate(class_names)}
    all_windows: List[Tuple[np.ndarray, int]] = []
    skipped = 0

    csv_files = list(Path(data_dir).rglob("*.csv"))
    if not csv_files:
        print(f"[data] 경고: {data_dir} 에서 CSV 파일을 찾지 못했습니다.", file=sys.stderr)
        return []

    print(f"[data] rglob 결과: {len(csv_files)}개 CSV  (예: {csv_files[0] if csv_files else 'N/A'})", file=sys.stderr)

    # sensor_type 필터 — (1) 경로 컴포넌트 우선, (2) 헤더 기반, (3) 채널 존재 여부 fallback
    filter_kw = sensor_type.strip().lower()
    if filter_kw in ("accel", "torque", "combined"):
        # 1차: 경로에 "Accel" / "Torque" / "Combined" 폴더가 있는 구조적 데이터
        path_filtered = [f for f in csv_files
                         if any(p.lower() == filter_kw for p in f.parts)]
        print(f"[data] 경로필터({filter_kw}): {len(path_filtered)}/{len(csv_files)}  "
              f"부분목록={[str(f.parts[-2:]) for f in path_filtered[:3]]}", file=sys.stderr)
        if path_filtered:
            csv_files = path_filtered
            print(f"[data] sensor_type={filter_kw} 경로 필터 → {len(csv_files)}개 파일", file=sys.stderr)
        else:
            # 2차 fallback: CSV 헤더를 읽어 센서 타입 추론 (평탄한 폴더 구조 대응)
            # combined 파일(accel+torque)은 accel/torque 어느 쪽 요청에도 사용 가능
            def _header_match(f: "Path") -> bool:
                detected = _detect_sensor_type_from_headers(str(f))
                if detected == filter_kw:
                    return True
                # combined CSV는 accel / torque / combined 세 가지 학습에 모두 사용 가능
                if detected == "combined" and filter_kw in ("accel", "torque", "combined"):
                    return True
                return False

            header_filtered = [f for f in csv_files if _header_match(f)]
            if header_filtered:
                csv_files = header_filtered
                print(f"[data] sensor_type={filter_kw} 헤더 감지(경로 미매칭) → {len(csv_files)}개 파일", file=sys.stderr)
            else:
                # 3차 fallback: 요청 채널이 CSV 헤더에 실제로 존재하는지 확인
                import csv as _csv_mod
                def _has_channels(f: "Path") -> bool:
                    ch_lower = [c.strip().lower() for c in channels]
                    for enc in ("utf-8-sig", "cp949", "utf-8"):
                        try:
                            with open(str(f), newline="", encoding=enc, errors="replace") as fh:
                                hdrs = [h.strip().lower() for h in (next(_csv_mod.reader(fh), []))]
                            return all(c in hdrs for c in ch_lower)
                        except Exception:
                            continue
                    return False

                csv_files = [f for f in csv_files if _has_channels(f)]
                print(f"[data] sensor_type={filter_kw} 채널 존재 여부 기반 → {len(csv_files)}개 파일", file=sys.stderr)

    for csv_path in csv_files:
        segments, label_str = _read_signal_csv(str(csv_path), channels, label_column,
                                               filter_op_column=filter_op_column)

        # 레이블 결정: CSV 컬럼 → 경로 컴포넌트 → 부모 폴더명
        if label_str is None:
            for part in csv_path.parts:
                if part.lower() in name_to_id:
                    label_str = part
                    break
        if label_str is None:
            label_str = csv_path.parent.name

        label_int = name_to_id.get(label_str.lower())
        if label_int is None:
            skipped += 1
            continue

        # 구간별 윈도우 추출 (경계 오염 방지)
        file_windows: List[Tuple[np.ndarray, int]] = []
        short_segs = 0
        for seg in segments:
            if seg.shape[0] < window_size:
                short_segs += 1
                continue
            file_windows.extend(_extract_windows(seg, label_int, window_size, stride, normalize=normalize))

        if not file_windows:
            print(
                f"[data] 경고: 유효 구간 없음 (전체 {len(segments)}구간, "
                f"window_size={window_size} 미만 {short_segs}개) — {csv_path.name}",
                file=sys.stderr,
            )
            skipped += 1
            continue

        if short_segs:
            print(
                f"[data] {csv_path.name}: {len(segments)}구간 중 {short_segs}개 스킵 "
                f"(window_size={window_size} 미만), {len(file_windows)}개 윈도우 추출",
                file=sys.stderr,
            )
        all_windows.extend(file_windows)

    # 클래스별 윈도우 수 진단 출력
    cls_dist: Dict[str, int] = {}
    for _, lbl in all_windows:
        name = class_names[lbl] if lbl < len(class_names) else str(lbl)
        cls_dist[name] = cls_dist.get(name, 0) + 1
    dist_log = ", ".join(f"{k}={v}" for k, v in sorted(cls_dist.items()))
    print(
        f"[data] 디렉터리 로드 완료: {len(csv_files) - skipped}개 파일, "
        f"{len(all_windows)}개 윈도우 (건너뜀={skipped}) | 클래스별: {dist_log or '없음'}",
        file=sys.stderr,
    )
    if skipped > 0:
        print(
            f"[data] ⚠ {skipped}개 파일이 건너뛰어짐. "
            f"원인: Label 컬럼 없음, 경로에 클래스명 없음, 또는 샘플 수 부족. "
            f"class_names={class_names}, label_column 확인 필요.",
            file=sys.stderr,
        )
    return all_windows


def load_windows_from_file_list(
    csv_files: List[Dict[str, str]],
    channels: List[str],
    label_column: str,
    class_names: List[str],
    window_size: int,
    stride: int,
    normalize: bool = True,
    filter_op_column: Optional[str] = None,
) -> List[Tuple[np.ndarray, int]]:
    """명시적 파일 목록에서 윈도우를 추출합니다.

    Args:
        csv_files: [{"path": "...", "label": "..."}, ...] 목록
        channels: 채널 컬럼명 목록
        label_column: CSV 내 레이블 컬럼명 (파일 목록의 label보다 낮은 우선순위)
        class_names: 클래스명 → 정수 인덱스 매핑 기준
        window_size: 윈도우 크기
        stride: 슬라이딩 스트라이드

    Returns:
        [(window_np, label_int), ...]
    """
    name_to_id = {n.lower(): i for i, n in enumerate(class_names)}
    all_windows: List[Tuple[np.ndarray, int]] = []
    skipped = 0

    for entry in csv_files:
        path = entry.get("path", "")
        forced_label = entry.get("label", "").strip()

        if not os.path.isfile(path):
            print(f"[data] 경고: 파일 없음 — {path}", file=sys.stderr)
            skipped += 1
            continue

        segments, csv_label = _read_signal_csv(path, channels, label_column,
                                              filter_op_column=filter_op_column)

        # 레이블 우선순위: entry["label"] > CSV 내 label_column
        label_str = forced_label if forced_label else (csv_label or "")
        label_int = name_to_id.get(label_str.lower())
        if label_int is None:
            skipped += 1
            continue

        # 구간별 윈도우 추출 (경계 오염 방지)
        file_windows: List[Tuple[np.ndarray, int]] = []
        short_segs = 0
        for seg in segments:
            if seg.shape[0] < window_size:
                short_segs += 1
                continue
            file_windows.extend(_extract_windows(seg, label_int, window_size, stride, normalize=normalize))

        if not file_windows:
            print(
                f"[data] 경고: 유효 구간 없음 (전체 {len(segments)}구간, "
                f"window_size={window_size} 미만 {short_segs}개) — {Path(path).name}",
                file=sys.stderr,
            )
            skipped += 1
            continue

        all_windows.extend(file_windows)

    print(
        f"[data] 파일 목록 로드 완료: {len(csv_files) - skipped}개 파일, "
        f"{len(all_windows)}개 윈도우 (건너뜀={skipped})",
        file=sys.stderr,
    )
    return all_windows


# ── PyTorch Dataset ──────────────────────────────────────────────────────────

class WindowDataset(Dataset):
    """슬라이딩 윈도우 데이터셋.

    각 샘플은 (window_tensor, label_tensor) 형태이며,
    window_tensor shape은 (T, C) — channels last (C# 대시보드 텐서 포맷).
    """

    def __init__(self, windows: List[Tuple[np.ndarray, int]]) -> None:
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        arr, label = self.windows[idx]
        x = torch.from_numpy(arr)                       # (T, C) float32
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# ── 1D-CNN 모델 ──────────────────────────────────────────────────────────────

class CNN1DClassifier(nn.Module):
    """PHM 진동/토크 신호 분류용 1D-CNN.

    입력 포맷: (B, T, C) — channels last (C# 대시보드와 동일).
    내부적으로 (B, C, T)로 변환 후 Conv1d 블록을 통과합니다.

    아키텍처 (n_windows에 따라 자동 선택):
        tiny  (<300)  : Conv(C→32) → Conv(32→64)  → GAP → FC
        small (<1000) : Conv(C→32) → Conv(32→64)  → Conv(64→128) → GAP → FC
        full  (≥1000) : Conv(C→32) → ... → Conv(128→256) → GAP → FC
    """

    def __init__(self, n_channels: int, n_classes: int, n_windows: int = 9999) -> None:
        super().__init__()

        if n_windows < 300:
            # tiny: 2블록 — 소규모 데이터셋용
            layers = [
                nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            ]
            fc_in = 64
        elif n_windows < 1000:
            # small: 3블록 — 중규모 데이터셋용
            layers = [
                nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.MaxPool1d(2),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            ]
            fc_in = 128
        else:
            # full: 4블록 — 대규모 데이터셋용
            layers = [
                nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32), nn.ReLU(inplace=True), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.MaxPool1d(2),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.MaxPool1d(2),
                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            ]
            fc_in = 256

        self.conv_blocks = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(fc_in, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파.

        Args:
            x: (B, T, C) — channels last

        Returns:
            logits: (B, n_classes)
        """
        # (B, T, C) → (B, C, T)
        x = x.permute(0, 2, 1)
        x = self.conv_blocks(x)   # (B, 256, T')
        x = self.pool(x)          # (B, 256, 1)
        x = x.squeeze(-1)         # (B, 256)
        return self.classifier(x) # (B, n_classes)


# ── AE 전용 Dataset ──────────────────────────────────────────────────────────

class AEWindowDataset(Dataset):
    """AE 학습용 데이터셋 — 레이블 없이 윈도우 텐서만 반환합니다."""

    def __init__(self, windows: List[Tuple[np.ndarray, int]]) -> None:
        self.windows = [w for w, _ in windows]  # 레이블 무시

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.windows[idx])  # (T, C) float32


# ── AE 1D-CNN 모델 ───────────────────────────────────────────────────────────

class AE1DCNN(nn.Module):
    """1D-CNN 오토인코더.

    입력/출력 포맷: (B, T, C) — channels last, C# 대시보드와 동일.
    구조: Encoder(Conv × 3, MaxPool × 2) → interpolate(T 복원) → Decoder(Conv × 3)

    Args:
        n_channels:   입력/출력 채널 수
        base_filters: 첫 번째 인코더 레이어 필터 수 (기본 32).
                      낮출수록 모델 용량 감소 → 이상 패턴 재구성 실패 확률 상승
                      (이상탐지 민감도 ↑).  권장: 8 / 16 / 32 / 64
    """

    def __init__(self, n_channels: int, base_filters: int = 32) -> None:
        super().__init__()
        f = base_filters
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, f,   kernel_size=7, padding=3),
            nn.BatchNorm1d(f),   nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(f,   f*2, kernel_size=5, padding=2),
            nn.BatchNorm1d(f*2), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(f*2, f*4, kernel_size=3, padding=1),
            nn.BatchNorm1d(f*4), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(f*4, f*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(f*2), nn.ReLU(inplace=True),
            nn.Conv1d(f*2, f,   kernel_size=5, padding=2),
            nn.BatchNorm1d(f),   nn.ReLU(inplace=True),
            nn.Conv1d(f, n_channels, kernel_size=7, padding=3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파.

        Args:
            x: (B, T, C) — channels last

        Returns:
            recon: (B, T, C) — 복원 신호
        """
        T = x.size(1)
        z = self.encoder(x.permute(0, 2, 1))         # (B, C, T) → (B, f*4, T//4)
        z_up = torch.nn.functional.interpolate(       # (B, f*4, T//4) → (B, f*4, T)
            z, size=T, mode="linear", align_corners=False
        )
        return self.decoder(z_up).permute(0, 2, 1)   # (B, T, C)


# ── AE 학습 루프 ──────────────────────────────────────────────────────────────

def train_ae(
    params: dict,
    windows: List[Tuple[np.ndarray, int]],
    n_channels: int,
    mlflow_run=None,
    ae_threshold_percentile: float = 99,
) -> Tuple["AE1DCNN", float, int, float, float, float, float, np.ndarray, np.ndarray]:
    """AE-CNN1D 모델을 학습하고 (model, best_val_mae, epochs, mae_thr, rms_mean, rms_thr, rms_std, norm_mean, norm_std)를 반환합니다.

    정규화 전략:
        - 전역(global) 정규화: 훈련 데이터 전체의 채널별 mean/std 로 정규화.
          → 절대 진폭·에너지 정보가 보존되어 AE 재구성 오차로 이상 탐지 가능.
        - norm_mean / norm_std 는 _meta.json 에 저장되어 평가 시 동일하게 적용.
    """
    seed = int(params.get("seed", 42))
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

    epochs     = int(params.get("epochs", 50))
    batch_size = int(params.get("batch_size", 32))
    lr         = float(params.get("lr", 0.001))
    val_split  = float(params.get("val_split", 0.2))
    patience   = max(5, min(10, epochs // 8))

    # ── RAW 윈도우에서 RMS 통계 계산 (진폭 이상 감지용) ────────────────────
    # windows는 normalize=False로 로드된 RAW 데이터
    raw_rms_list: List[float] = []
    for w, _ in windows:
        # 전체 채널 RMS: sqrt(mean(x^2))
        rms = float(np.sqrt(np.mean(w.astype(np.float64) ** 2)))
        raw_rms_list.append(rms)
    rms_arr  = np.array(raw_rms_list, dtype=np.float64)
    rms_mean = float(rms_arr.mean())
    rms_std  = float(rms_arr.std()) if rms_arr.std() > 1e-8 else float(rms_arr.mean() * 0.1)
    rms_thr  = float(rms_mean + 2.0 * rms_std)
    print(
        f"[train_ae] RMS 통계 — mean={rms_mean:.4f}  std={rms_std:.4f}  thr={rms_thr:.4f}",
        file=sys.stderr,
    )

    # ── 전역 정규화 (진폭 보존) ──────────────────────────────────────────────
    norm_mean, norm_std = _compute_global_stats(windows)
    print(
        f"[train_ae] 전역 정규화 — mean={np.round(norm_mean, 4).tolist()}  "
        f"std={np.round(norm_std, 4).tolist()}",
        file=sys.stderr,
    )
    windows_norm = _apply_global_norm(windows, norm_mean, norm_std)

    # 랜덤 분할 (레이블 불필요)
    n = len(windows_norm)
    n_val = max(1, int(n * val_split))
    idx = list(range(n)); random.shuffle(idx)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    dataset      = AEWindowDataset(windows_norm)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)

    base_filters = int(params.get("ae_base_filters", 32))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = AE1DCNN(n_channels=n_channels, base_filters=base_filters).to(device)
    print(f"[train_ae] AE1DCNN base_filters={base_filters}  파라미터 수≈{sum(p.numel() for p in model.parameters()):,}", file=sys.stderr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )
    criterion = nn.L1Loss()  # MAE — C# MeanAbsoluteError와 동일

    print(f"[train_ae] 디바이스: {device} | 학습={len(train_idx)} 검증={len(val_idx)} | patience={patience}",
          file=sys.stderr)

    best_val_mae  = float("inf")
    best_state: dict = {}
    no_improve    = 0
    epochs_trained = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0; n_batches = 0
        for x_batch in train_loader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), x_batch)
            loss.backward(); optimizer.step()
            total_loss += loss.item(); n_batches += 1
        avg_loss = total_loss / n_batches if n_batches > 0 else float("nan")

        model.eval()
        val_sum = 0.0; val_n = 0
        with torch.no_grad():
            for x_batch in val_loader:
                x_batch = x_batch.to(device)
                val_sum += criterion(model(x_batch), x_batch).item() * x_batch.size(0)
                val_n   += x_batch.size(0)
        val_mae = val_sum / val_n if val_n > 0 else float("nan")
        epochs_trained = epoch

        scheduler.step(val_mae)
        print(json.dumps({"epoch": epoch, "loss": round(avg_loss, 6),
                          "val_mse": round(val_mae, 6)}), flush=True)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"[train_ae] Early stopping at epoch {epoch} (patience={patience})",
                  file=sys.stderr)
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    # ── MAE 임계값: val 세트 샘플별 MAE의 mean + 2σ ────────────────────────
    per_sample_maes: List[float] = []
    with torch.no_grad():
        for x_batch in val_loader:
            x_batch = x_batch.to(device)
            mae_per = (model(x_batch) - x_batch).abs().mean(dim=(1, 2))  # (B,)
            per_sample_maes.extend(mae_per.cpu().numpy().tolist())

    if per_sample_maes:
        err_arr = np.array(per_sample_maes, dtype=np.float64)
        # 퍼센타일 기반 임계값: mean+2σ보다 분포 형태에 덜 민감하고 해석이 명확함
        mae_thr = float(np.percentile(err_arr, ae_threshold_percentile))
        print(
            f"[train_ae] MAE 임계값 ({ae_threshold_percentile}th pct) = {mae_thr:.6f}  "
            f"(mean={err_arr.mean():.6f}  std={err_arr.std():.6f})",
            file=sys.stderr,
        )
    else:
        mae_thr = float(best_val_mae * 2.0)

    print(
        f"[train_ae] 완료 — best_val_mae={best_val_mae:.6f}  "
        f"mae_thr={mae_thr:.6f}  rms_mean={rms_mean:.4f}  rms_std={rms_std:.4f}  rms_thr={rms_thr:.4f}  epochs={epochs_trained}",
        file=sys.stderr,
    )
    return model, best_val_mae, epochs_trained, mae_thr, rms_mean, rms_thr, rms_std, norm_mean, norm_std


# ── AE ONNX 내보내기 ──────────────────────────────────────────────────────────

def export_onnx_ae(
    model: "AE1DCNN",
    output_path: str,
    window_size: int,
    n_channels: int,
) -> None:
    """AE 모델을 ONNX opset 17로 내보냅니다.

    입력: "input"  shape (1, T, C)
    출력: "recon"  shape (1, T, C)
    """
    import onnx

    model.eval(); model.cpu()
    dummy = torch.randn(1, window_size, n_channels, dtype=torch.float32)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    torch.onnx.export(
        model, dummy, output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["recon"],
        dynamic_axes={
            "input": {0: "batch", 1: "time"},
            "recon": {0: "batch", 1: "time"},
        },
        dynamo=False,
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"[export] AE ONNX 저장 완료: {output_path}", file=sys.stderr)


# ── 학습 루프 ────────────────────────────────────────────────────────────────

def train(
    params: dict,
    windows: List[Tuple[np.ndarray, int]],
    n_classes: int,
    n_channels: int,
    mlflow_run=None,
) -> Tuple[CNN1DClassifier, float, int]:
    """CNN1D 모델을 학습하고 최적 모델을 반환합니다.

    Args:
        params: 학습 파라미터 딕셔너리
        windows: [(window_np, label_int), ...] 전체 데이터
        n_classes: 분류 클래스 수
        n_channels: 입력 채널 수
        mlflow_run: 활성 mlflow run 객체 (None이면 MLflow 미사용)

    Returns:
        (best_model, best_val_accuracy, epochs_trained)
    """
    seed = int(params.get("seed", 42))
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    epochs = int(params.get("epochs", 50))
    batch_size = int(params.get("batch_size", 32))
    lr = float(params.get("lr", 0.001))
    val_split = float(params.get("val_split", 0.2))
    # 데이터 크기에 따라 patience 동적 조정 (최소 5, 최대 10)
    patience = max(5, min(10, epochs // 8))

    dataset = WindowDataset(windows)
    labels_arr = np.array([w[1] for w in windows])

    # 계층별 분할
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(labels_arr)), labels_arr))

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_windows = len(windows)

    # 모델 크기 결정 로그
    if n_windows < 300:
        arch = "tiny(2블록)"
    elif n_windows < 1000:
        arch = "small(3블록)"
    else:
        arch = "full(4블록)"
    print(
        f"[train] 디바이스: {device} | 아키텍처: {arch} | patience: {patience}",
        file=sys.stderr,
    )
    print(
        f"[train] 학습={len(train_idx)} 샘플, 검증={len(val_idx)} 샘플",
        file=sys.stderr,
    )

    model = CNN1DClassifier(n_channels=n_channels, n_classes=n_classes, n_windows=n_windows).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state: dict = {}
    no_improve = 0
    epochs_trained = 0

    for epoch in range(1, epochs + 1):
        # ── 학습 ─────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        n_batches = 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches if n_batches > 0 else float("nan")

        # ── 검증 ─────────────────────────────────────────────────────────────
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(x_batch)
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        val_acc = correct / total if total > 0 else 0.0
        epochs_trained = epoch

        # LR 스케줄러 업데이트 (val_acc 기준)
        scheduler.step(val_acc)

        # stdout JSON 로그 (C# 대시보드 파싱 용도)
        log_line = json.dumps({"epoch": epoch, "loss": round(avg_loss, 6), "val_acc": round(val_acc, 6)})
        print(log_line, flush=True)

        if mlflow_run is not None:
            try:
                import mlflow
                mlflow.log_metric("loss", avg_loss, step=epoch)
                mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            except Exception:
                pass

        # 최적 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        # 조기 종료
        if no_improve >= patience:
            print(
                f"[train] Early stopping at epoch {epoch} (patience={patience})",
                file=sys.stderr,
            )
            break

    # 최적 가중치 복원
    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    print(
        f"[train] 학습 완료 — best_val_acc={best_val_acc:.4f}, epochs={epochs_trained}",
        file=sys.stderr,
    )
    return model, best_val_acc, epochs_trained


# ── ONNX 내보내기 ─────────────────────────────────────────────────────────────

def export_onnx(
    model: CNN1DClassifier,
    output_path: str,
    window_size: int,
    n_channels: int,
) -> None:
    """학습된 모델을 ONNX opset 15로 내보냅니다.

    입력: "input"  shape (1, T, C) — dynamic T
    출력: "logits" shape (1, n_classes)

    Args:
        model: 학습된 CNN1DClassifier (CPU 모드)
        output_path: 저장 경로 (.onnx)
        window_size: 더미 입력 T 크기
        n_channels: 채널 수 C
    """
    import onnx

    model.eval()
    model.cpu()

    dummy = torch.randn(1, window_size, n_channels, dtype=torch.float32)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # dynamo=False: PyTorch 2.6+ 기본값(dynamo=True)에서 dynamic_axes 충돌 방지
    # 레거시 TorchScript 기반 exporter를 명시적으로 사용
    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input":  {0: "batch", 1: "time"},
            "logits": {0: "batch"},
        },
        dynamo=False,
    )

    # 모델 유효성 검사
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"[export] ONNX 저장 완료 및 검증 통과: {output_path}", file=sys.stderr)


# ── _meta.json 저장 ───────────────────────────────────────────────────────────

def save_meta(
    output_path: str,
    params: dict,
    class_names: List[str],
    channels: List[str],
    val_accuracy: float = 0.0,
    epochs_trained: int = 0,
    mlflow_run_id: Optional[str] = None,
    mlflow_tracking_uri: Optional[str] = None,
    val_mse: Optional[float] = None,
    threshold: Optional[float] = None,
    rms_mean: Optional[float] = None,
    rms_thr:  Optional[float] = None,
    rms_std:  Optional[float] = None,
    norm_mean: Optional[np.ndarray] = None,  # 전역 정규화 채널별 mean (AE 전용)
    norm_std:  Optional[np.ndarray] = None,  # 전역 정규화 채널별 std  (AE 전용)
    n_channels_override: Optional[int] = None,  # _resolve_channels 확장 후 실제 채널 수
) -> str:
    """ONNX 파일 옆에 _meta.json 사이드카를 저장합니다.

    session="AD" (AE) 인 경우 kind=AE-CNN1D, output_name=recon, threshold/norm_mean/norm_std 포함.
    session="FD" (CLS) 인 경우 kind=CNN1D, output_name=logits, val_accuracy 포함.

    Returns:
        저장된 _meta.json 경로
    """
    session     = params.get("session", "CLS").upper()
    # 구버전 호환: AD → AE, FD → CLS
    session     = {"AD": "AE", "FD": "CLS"}.get(session, session)
    is_ae       = session == "AE"           # AE: 이상탐지, CLS: 결함진단
    sensor_type = params.get("sensor_type", "accel")

    meta = {
        "kind":        "AE-CNN1D" if is_ae else "CNN1D-CLS",
        "session":     session,   # "AE" | "CLS"
        "sensor_type": sensor_type,
        "y_column":    channels[0] if channels else "",
        "channels":    channels,
        "n_channels":  n_channels_override if n_channels_override is not None else len(channels),
        "window_size": int(params.get("window_size", 1024)),
        "input_name":  "input",
        "output_name": "recon" if is_ae else "logits",
        # norm_mean이 있으면 전역 정규화 → per-sample 불필요
        "standardize_per_sample": norm_mean is None,
        "epochs_trained": epochs_trained,
    }

    # 전역 정규화 통계 저장 (추론 시 동일 정규화 적용)
    if norm_mean is not None:
        meta["norm_mean"] = [round(float(v), 8) for v in norm_mean]
    if norm_std is not None:
        meta["norm_std"]  = [round(float(v), 8) for v in norm_std]

    if is_ae:
        meta["val_mae"]   = round(val_mse or 0.0, 6)   # val_mse 인수가 실제로는 val_mae 값
        meta["threshold"] = round(threshold or 0.0, 6)  # MAE 임계값 (global norm 공간)
        normal_classes    = params.get("normal_classes", class_names)
        meta["normal_classes"] = normal_classes
        # RMS 진폭 이상 감지용 통계 (외력 등 진폭 변화 감지)
        if rms_mean is not None:
            meta["rms_mean"] = round(float(rms_mean), 6)
        if rms_thr is not None:
            meta["rms_thr"]  = round(float(rms_thr),  6)
        if rms_std is not None:
            meta["rms_std"]  = round(float(rms_std),  6)
    else:
        meta["class_names"] = class_names
        meta["n_classes"]   = len(class_names)
        meta["recon_output_name"] = None
        meta["val_accuracy"] = round(val_accuracy, 6)

    if mlflow_run_id:
        meta["mlflow_run_id"] = mlflow_run_id
    if mlflow_tracking_uri:
        meta["mlflow_tracking_uri"] = mlflow_tracking_uri

    meta_path = os.path.splitext(output_path)[0] + "_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[meta] _meta.json 저장: {meta_path}", file=sys.stderr)
    return meta_path


# ── MLflow 헬퍼 ──────────────────────────────────────────────────────────────

def _try_setup_mlflow(params: dict):
    """MLflow를 초기화하고 run을 시작합니다. 실패 시 None을 반환합니다."""
    uri = params.get("mlflow_tracking_uri") or os.environ.get("MLFLOW_TRACKING_URI", "")
    if not uri:
        return None, None

    try:
        import mlflow
        import mlflow.pytorch

        mlflow.set_tracking_uri(uri)
        experiment = params.get("mlflow_experiment", "PHM-DL")
        mlflow.set_experiment(experiment)
        run = mlflow.start_run()

        # 파라미터 로깅
        log_params = {
            "window_size": params.get("window_size", 1024),
            "stride": params.get("stride", 512),
            "epochs": params.get("epochs", 50),
            "batch_size": params.get("batch_size", 32),
            "lr": params.get("lr", 0.001),
            "n_channels": len(params.get("channels", [])),
            "n_classes": len(params.get("class_names", [])),
            "class_names": ",".join(params.get("class_names", [])),
        }
        mlflow.log_params(log_params)
        print(f"[mlflow] run 시작: {run.info.run_id} (experiment={experiment})", file=sys.stderr)
        return run, mlflow
    except Exception as e:
        print(f"[mlflow] 초기화 실패 (건너뜀): {e}", file=sys.stderr)
        return None, None


def _try_end_mlflow(
    mlflow_run,
    mlflow_mod,
    model: CNN1DClassifier,
    output_path: str,
    meta_path: str,
    best_val_acc: float,
    epochs_trained: int,
) -> Optional[str]:
    """MLflow run을 종료하고 아티팩트를 업로드합니다. 실패 시 None을 반환합니다."""
    if mlflow_run is None or mlflow_mod is None:
        return None
    try:
        mlflow_mod.log_metric("best_val_accuracy", best_val_acc)
        mlflow_mod.log_metric("epochs_trained", epochs_trained)
    except Exception as e:
        print(f"[mlflow] 메트릭 로깅 실패 (건너뜀): {e}", file=sys.stderr)

    # ONNX / meta 파일 아티팩트 업로드 (실패해도 계속)
    for path in [output_path, meta_path]:
        try:
            mlflow_mod.log_artifact(path)
        except Exception as e:
            print(f"[mlflow] 아티팩트 업로드 실패 (건너뜀): {os.path.basename(path)} — {e}", file=sys.stderr)

    # PyTorch 모델 로깅 — 구버전 MLflow 서버(2.x < 2.13) 에서 /logged-models API 없을 수 있음
    # 404 또는 인코딩 오류가 발생해도 run 종료에 영향을 주지 않도록 완전히 격리
    try:
        import importlib
        if importlib.util.find_spec("mlflow.pytorch") is not None:
            try:
                mlflow_mod.pytorch.log_model(model, name="pytorch_model")
            except TypeError:
                try:
                    mlflow_mod.pytorch.log_model(model, artifact_path="pytorch_model")
                except Exception:
                    pass  # 구버전 서버에서 완전히 무시
    except Exception as e:
        # 이모지/비ASCII 포함 오류 메시지가 cp949 stdout에 쓰이는 문제 방지 — ASCII safe 출력
        try:
            msg = e.args[0] if e.args else ""
            safe_msg = msg.encode("ascii", errors="replace").decode("ascii")
            print(f"[mlflow] pytorch 모델 로깅 실패 (건너뜀): {safe_msg}", file=sys.stderr)
        except Exception:
            print("[mlflow] pytorch 모델 로깅 실패 (건너뜀)", file=sys.stderr)

    # run 종료는 반드시 시도 — 위 오류와 완전히 분리
    run_id = None
    try:
        run_id = mlflow_run.info.run_id
    except Exception:
        pass
    try:
        mlflow_mod.end_run()
        print(f"[mlflow] run 종료: {run_id}", file=sys.stderr)
    except Exception as e:
        try:
            safe_msg = str(e).encode("ascii", errors="replace").decode("ascii")
            print(f"[mlflow] run 종료 실패 (건너뜀): {safe_msg}", file=sys.stderr)
        except Exception:
            print("[mlflow] run 종료 실패 (건너뜀)", file=sys.stderr)
    return run_id


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PHM 1D-CNN 딥러닝 모델 학습 및 ONNX 변환"
    )
    parser.add_argument("--params", required=True, help="파라미터 JSON 파일 경로")
    args = parser.parse_args()

    if not os.path.isfile(args.params):
        print(json.dumps({"error": f"params 파일을 찾을 수 없습니다: {args.params}"}))
        sys.exit(1)

    params = load_params(args.params)

    # 필수 파라미터 검증
    output_path: str = params.get("output", "")
    if not output_path:
        print(json.dumps({"error": "params에 'output' 경로가 없습니다."}))
        sys.exit(1)

    channels: List[str] = params.get("channels", [])
    if not channels:
        print(json.dumps({"error": "params에 'channels' 목록이 없습니다."}))
        sys.exit(1)

    class_names: List[str] = params.get(
        "class_names",
        ["normal", "fault", "bearing_fault", "gear_fault", "imbalance", "looseness"],
    )
    label_column: str = params.get("label_column", "Label")
    filter_op_column: Optional[str] = params.get("filter_op_column", None)
    window_size: int = int(params.get("window_size", 1024))
    stride: int = int(params.get("stride", 512))

    session_raw = params.get("session", "CLS").upper()
    # 구버전 호환: AD → AE, FD → CLS
    _SESSION_ALIAS = {"AD": "AE", "FD": "CLS"}
    session = _SESSION_ALIAS.get(session_raw, session_raw)
    if session != session_raw:
        print(f"[main] session 별칭 변환: {session_raw!r} → {session!r}  (AE=이상탐지, CLS=결함진단)", file=sys.stderr)
    is_ae = session == "AE"
    n_channels = len(channels)
    n_classes  = len(class_names)

    # AE 모드에서는 normal_classes 폴더만 로드
    load_class_names = params.get("normal_classes", class_names) if is_ae else class_names

    print(
        f"[main] session={session} | 채널={channels} | "
        f"{'정상클래스' if is_ae else '클래스'}={load_class_names} | "
        f"window_size={window_size}, stride={stride}",
        file=sys.stderr,
    )

    # ── 데이터 로드 ───────────────────────────────────────────────────────────
    windows: List[Tuple[np.ndarray, int]] = []

    # AE 학습: windows는 RAW로 전달 (train_ae 내부에서 RMS 통계 계산 후 z-score 적용)
    # CLS 학습: per-sample z-score 정규화 적용
    normalize_windows = not is_ae

    if "csv_files" in params and params["csv_files"]:
        windows = load_windows_from_file_list(
            csv_files=params["csv_files"],
            channels=channels,
            label_column=label_column,
            class_names=load_class_names,
            window_size=window_size,
            stride=stride,
            normalize=normalize_windows,
            filter_op_column=filter_op_column,
        )
    elif "data_dir" in params and params["data_dir"]:
        windows = load_windows_from_dir(
            data_dir=params["data_dir"],
            channels=channels,
            label_column=label_column,
            class_names=load_class_names,
            window_size=window_size,
            stride=stride,
            sensor_type=params.get("sensor_type", ""),
            normalize=normalize_windows,
            filter_op_column=filter_op_column,
        )
    else:
        print(json.dumps({"error": "params에 'data_dir' 또는 'csv_files' 중 하나가 필요합니다."}))
        sys.exit(1)

    if len(windows) == 0:
        print(json.dumps({"warning": "유효한 윈도우가 없어 학습을 건너뜁니다 (데이터 없음). 수집 후 재시도하세요."}), flush=True)
        print(f"[main] ⚠ 데이터 없음 — 학습 건너뜀 (sensor_type={params.get('sensor_type','?')}, "
              f"channels={channels}, filter_op={params.get('filter_op_column')})", file=sys.stderr)
        sys.exit(0)   # 데이터 없음은 오류가 아닌 정상 종료

    # _resolve_channels 가 채널을 확장했을 수 있으므로 실제 데이터 shape 으로 재설정
    # 예: channels=["Trq(%)"] → Ax0_Trq(%)/Ax1_Trq(%)/Ax2_Trq(%) 3채널로 확장된 경우
    actual_n_channels = windows[0][0].shape[1]
    if actual_n_channels != n_channels:
        print(
            f"[main] 채널 수 재설정: {n_channels} → {actual_n_channels} "
            f"(_resolve_channels 확장 결과)",
            file=sys.stderr,
        )
        n_channels = actual_n_channels

    # 클래스 분포 출력
    label_counts: Dict[int, int] = {}
    for _, lbl in windows:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    dist_str = ", ".join(
        f"{load_class_names[i] if i < len(load_class_names) else i}={c}"
        for i, c in sorted(label_counts.items())
    )
    print(f"[main] {'정상 데이터' if is_ae else '클래스'} 분포: {dist_str}", file=sys.stderr)

    # ── 단일 클래스 검사 ─────────────────────────────────────────────────────
    if not is_ae and len(label_counts) < 2:
        # 요청된 클래스가 2개 이상인데 데이터가 부족한 경우 → 진단 오류 (자동 전환 금지)
        # 요청된 클래스가 1개인 경우 → AE 이상탐지 모드로 자동 전환 (기존 동작 유지)
        if len(class_names) >= 2:
            found_names   = [class_names[i] for i in sorted(label_counts.keys())
                             if i < len(class_names)]
            missing_names = [c for c in class_names
                             if c.lower() not in {class_names[i].lower()
                                                  for i in label_counts.keys()
                                                  if i < len(class_names)}]
            diag = (
                f"CLS 학습 실패: 요청된 {len(class_names)}개 클래스 중 "
                f"{len(found_names)}개만 발견되었습니다. "
                f"발견={found_names}, 누락={missing_names}. "
                f"\n\n해결 방법:"
                f"\n  (1) 폴더 구조 확인: data_dir/{'/'.join(missing_names)}/Accel/*.csv 가 존재해야 합니다."
                f"\n  (2) CSV Label 컬럼 확인: label_column='{label_column}' 에 {missing_names} 값이 있어야 합니다."
                f"\n  (3) 해당 클래스의 학습 데이터를 먼저 수집 후 재시도하세요."
            )
            print(f"[main] ❌ {diag}", file=sys.stderr)
            print(json.dumps({"error": diag}), flush=True)
            sys.exit(1)

        # class_names가 1개인 경우: AE 이상탐지 모드로 자동 전환
        print(
            f"[main] ⚠ 클래스 수 {len(label_counts)}개 (분류 학습 불가) "
            f"→ AE 이상탐지 모드로 자동 전환합니다.",
            file=sys.stderr,
        )
        is_ae   = True
        session = "AE"
        params["session"] = "AE"

        # AE는 RAW 데이터 필요 (RMS 통계 계산) — normalize=False 로 재로드
        print("[main] AE 모드 데이터 재로드 (normalize=False)...", file=sys.stderr)
        if "csv_files" in params and params["csv_files"]:
            windows = load_windows_from_file_list(
                csv_files=params["csv_files"],
                channels=channels,
                label_column=label_column,
                class_names=load_class_names,
                window_size=window_size,
                stride=stride,
                normalize=False,
                filter_op_column=filter_op_column,
            )
        else:
            windows = load_windows_from_dir(
                data_dir=params["data_dir"],
                channels=channels,
                label_column=label_column,
                class_names=load_class_names,
                window_size=window_size,
                stride=stride,
                sensor_type=params.get("sensor_type", ""),
                normalize=False,
                filter_op_column=filter_op_column,
            )

        # 출력 파일명을 AE 용으로 변경
        # CLS 요청 파일명(cls_fd / cls_torque) 또는 레거시(cnn1d_fd / cnn1d_torque) 모두 처리
        for old, new in [
            ("cls_fd",       "ae_fd"),
            ("cls_torque",   "ae_torque"),
            ("cnn1d_fd",     "ae_fd"),
            ("cnn1d_torque", "ae_torque"),
        ]:
            if old in output_path:
                output_path = output_path.replace(old, new)
                params["output"] = output_path
                print(f"[main] 출력 경로 변경 (AE 자동전환): {output_path}", file=sys.stderr)
                break

    mlflow_tracking_uri = params.get("mlflow_tracking_uri") or os.environ.get("MLFLOW_TRACKING_URI", "")

    # ══════════════════════════════════════════════════════════════════════════
    # AE (이상탐지) 경로
    # ══════════════════════════════════════════════════════════════════════════
    if is_ae:
        print(f"[main] AE 학습 시작 — 정상 샘플 {len(windows)}개 윈도우", file=sys.stderr)
        mlflow_run, mlflow_mod = _try_setup_mlflow(params)

        ae_model, best_val_mse, epochs_trained, threshold, rms_mean, rms_thr, rms_std, norm_mean, norm_std = train_ae(
            params=params,
            windows=windows,
            n_channels=n_channels,
            mlflow_run=mlflow_run,
            ae_threshold_percentile=float(params.get("ae_threshold_percentile", 99)),
        )

        export_onnx_ae(
            model=ae_model,
            output_path=output_path,
            window_size=window_size,
            n_channels=n_channels,
        )

        meta_path = save_meta(
            output_path=output_path,
            params=params,
            class_names=load_class_names,
            channels=channels,
            epochs_trained=epochs_trained,
            mlflow_run_id=mlflow_run.info.run_id if mlflow_run else None,
            mlflow_tracking_uri=mlflow_tracking_uri or None,
            val_mse=best_val_mse,
            threshold=threshold,
            rms_mean=rms_mean,
            rms_thr=rms_thr,
            rms_std=rms_std,
            norm_mean=norm_mean,
            norm_std=norm_std,
            n_channels_override=n_channels,  # _resolve_channels 확장 후 실제 채널 수
        )

        _try_end_mlflow(
            mlflow_run=mlflow_run, mlflow_mod=mlflow_mod,
            model=ae_model, output_path=output_path, meta_path=meta_path,
            best_val_acc=0.0, epochs_trained=epochs_trained,
        )

        info_str = (
            f"AE-CNN1D 학습 완료 | 윈도우={len(windows)} | "
            f"val_mse={best_val_mse:.6f} | threshold={threshold:.6f} | epochs={epochs_trained}"
        )
        result = {
            "info":      info_str,
            "val_mse":   round(best_val_mse, 6),
            "threshold": round(threshold, 6),
            "epochs":    epochs_trained,
        }
        print(json.dumps(result), flush=True)
        sys.exit(0)

    # ══════════════════════════════════════════════════════════════════════════
    # CLS (분류) 경로 — 기존 로직
    # ══════════════════════════════════════════════════════════════════════════

    # ── MLflow 초기화 ─────────────────────────────────────────────────────────
    mlflow_run, mlflow_mod = _try_setup_mlflow(params)

    # ── 학습 ─────────────────────────────────────────────────────────────────
    model, best_val_acc, epochs_trained = train(
        params=params,
        windows=windows,
        n_classes=n_classes,
        n_channels=n_channels,
        mlflow_run=mlflow_run,
    )

    # ── ONNX 내보내기 ─────────────────────────────────────────────────────────
    export_onnx(
        model=model,
        output_path=output_path,
        window_size=window_size,
        n_channels=n_channels,
    )

    # ── _meta.json 저장 ───────────────────────────────────────────────────────
    meta_path = save_meta(
        output_path=output_path,
        params=params,
        class_names=class_names,
        channels=channels,
        val_accuracy=best_val_acc,
        epochs_trained=epochs_trained,
        mlflow_run_id=mlflow_run.info.run_id if mlflow_run else None,
        mlflow_tracking_uri=mlflow_tracking_uri or None,
        n_channels_override=n_channels,  # _resolve_channels 확장 후 실제 채널 수
    )

    # ── MLflow 종료 ───────────────────────────────────────────────────────────
    _try_end_mlflow(
        mlflow_run=mlflow_run,
        mlflow_mod=mlflow_mod,
        model=model,
        output_path=output_path,
        meta_path=meta_path,
        best_val_acc=best_val_acc,
        epochs_trained=epochs_trained,
    )

    # ── 최종 결과 출력 ────────────────────────────────────────────────────────
    info_str = (
        f"CNN1D 학습 완료 | 윈도우={len(windows)} | "
        f"클래스={n_classes} | val_acc={best_val_acc:.4f} | epochs={epochs_trained}"
    )
    result = {
        "info": info_str,
        "accuracy": round(best_val_acc, 6),
        "epochs": epochs_trained,
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
