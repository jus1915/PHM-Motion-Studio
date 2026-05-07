"""
train_all_models.py — PHM 전체 모델 통합 학습 스크립트

축별 × 센서 타입별 (가속도 / 토크 / combined) CLS 모델을 순차 학습하고
마지막에 그룹별 성능을 하나의 표로 출력합니다.

사용법:
    python train_all_models.py

아래 ═══ 설정 ═══ 섹션에서 경로·파라미터를 수정하세요.
"""

import os
import sys
import time
import traceback
from typing import Dict, List, Optional, Tuple

# train_dl_model.py 와 같은 디렉터리에 있다고 가정
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_dl_model import (
    load_windows_from_dir,
    _add_feature_channels,
    _add_mixed_feature_channels,
    train,
    export_onnx,
    save_meta,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 설정
# ═══════════════════════════════════════════════════════════════════════════════

# 학습 데이터 루트 (하위 폴더에 클래스명 포함: .../normal/..., .../looseness/... 등)
DATA_DIR   = r"D:\Dev\hvs\WorkingSource\PHM-Motion-Studio\server\phm_data"

# ONNX + _meta.json 출력 디렉터리
OUTPUT_DIR = r"D:\PHM_Models"

# 클래스 목록 (폴더명 / CSV Label 컬럼값과 일치해야 함)
CLASS_NAMES = ["normal", "overload", "looseness"]

# Combined CSV 필터 — *_Combined.csv 파일만 로드
SENSOR_TYPE = "combined"

# ── 윈도우 파라미터 ─────────────────────────────────────────────────────────
WINDOW_SIZE = 128
STRIDE      = 32

# ── 학습 하이퍼파라미터 ─────────────────────────────────────────────────────
EPOCHS      = 200
BATCH_SIZE  = 32
LR          = 0.0005
VAL_SPLIT   = 0.2
SEED        = 42

# ── 손실 함수 ───────────────────────────────────────────────────────────────
LABEL_SMOOTHING   = 0.0    # 0.0 이면 비활성
USE_CLASS_WEIGHTS = False   # 클래스 불균형 보정

# ── 채널 증강 ────────────────────────────────────────────────────────────────
# augment_mode:
#   "standard" → 전채널 균일 FFT+derivative+abs (normalize=True 와 함께)
#   "mixed"    → 가속도: FFT+deriv / 토크: deriv+stats×6 / 공통: abs
#                (normalize=False, RAW 진폭 보존)
#
# 아래 JOBS 에서 job 별로 override 가능 (6번째 원소: augment_mode, 7번째: normalize)
DEFAULT_AUGMENT_MODE = "mixed"     # combined/torque 작업용 기본값
DEFAULT_NORMALIZE    = False       # "mixed" 모드는 normalize=False

ADD_FFT_CHANNELS        = True
ADD_DERIVATIVE_CHANNELS = True
ADD_TORQUE_STATS        = True
ADD_ABS_CHANNELS        = True


# ═══════════════════════════════════════════════════════════════════════════════
# 학습 대상 정의
# (그룹명, 채널 목록, Op 필터 컬럼, 출력 파일명 스템, augment_mode, normalize)
# ═══════════════════════════════════════════════════════════════════════════════

JOBS: List[Tuple[str, List[str], str, str, str, bool]] = [
    # 가속도 — 축별 (standard 증강, normalize=True)
    ("Accel-Ax0", ["x", "y", "z"], "Op_Ax0", "cls_accel_ax0", "standard", True),
    ("Accel-Ax1", ["x", "y", "z"], "Op_Ax1", "cls_accel_ax1", "standard", True),
    ("Accel-Ax2", ["x", "y", "z"], "Op_Ax2", "cls_accel_ax2", "standard", True),
    # 토크 — 축별 (mixed 증강, normalize=False)
    ("Torque-Ax0", ["Ax0_Trq(%)"], "Op_Ax0", "cls_torque_ax0", "mixed", False),
    ("Torque-Ax1", ["Ax1_Trq(%)"], "Op_Ax1", "cls_torque_ax1", "mixed", False),
    ("Torque-Ax2", ["Ax2_Trq(%)"], "Op_Ax2", "cls_torque_ax2", "mixed", False),
    # 결합 — 축별 (mixed 증강, normalize=False)
    ("Combined-Ax0", ["x", "y", "z", "Ax0_Trq(%)"], "Op_Ax0", "cls_combined_ax0", "mixed", False),
    ("Combined-Ax1", ["x", "y", "z", "Ax1_Trq(%)"], "Op_Ax1", "cls_combined_ax1", "mixed", False),
    ("Combined-Ax2", ["x", "y", "z", "Ax2_Trq(%)"], "Op_Ax2", "cls_combined_ax2", "mixed", False),
]


# ═══════════════════════════════════════════════════════════════════════════════
# 단일 작업 실행
# ═══════════════════════════════════════════════════════════════════════════════

def _make_params(
    channels: List[str],
    filter_op_col: str,
    output_path: str,
    augment_mode: str,
    normalize: bool,
) -> dict:
    return {
        "session":               "CLS",
        "sensor_type":           SENSOR_TYPE,
        "channels":              channels,
        "class_names":           CLASS_NAMES,
        "window_size":           WINDOW_SIZE,
        "stride":                STRIDE,
        "epochs":                EPOCHS,
        "batch_size":            BATCH_SIZE,
        "lr":                    LR,
        "val_split":             VAL_SPLIT,
        "seed":                  SEED,
        "label_smoothing":       LABEL_SMOOTHING,
        "use_class_weights":     USE_CLASS_WEIGHTS,
        "filter_op_column":      filter_op_col,
        "output":                output_path,
        "normalize":             normalize,
        "augment_mode":          augment_mode,
        "add_augmented_channels":True,
        "add_fft_channels":      ADD_FFT_CHANNELS,
        "add_derivative_channels":ADD_DERIVATIVE_CHANNELS,
        "add_torque_stats":      ADD_TORQUE_STATS,
        "add_abs_channels":      ADD_ABS_CHANNELS,
    }


def run_job(
    name: str,
    channels: List[str],
    filter_op_col: str,
    output_stem: str,
    augment_mode: str = DEFAULT_AUGMENT_MODE,
    normalize: bool   = DEFAULT_NORMALIZE,
) -> dict:
    """단일 (그룹, 센서, 축) 모델을 학습하고 결과 dict를 반환합니다."""

    output_path = os.path.join(OUTPUT_DIR, output_stem + ".onnx")
    params      = _make_params(channels, filter_op_col, output_path, augment_mode, normalize)

    _banner(f"[{name}] 시작  채널={channels}  필터={filter_op_col}  augment={augment_mode}  normalize={normalize}")
    t0 = time.time()

    try:
        # ── 1. 데이터 로드 ───────────────────────────────────────────────
        windows, seg_ids = load_windows_from_dir(
            data_dir         = DATA_DIR,
            channels         = channels,
            label_column     = "Label",
            class_names      = CLASS_NAMES,
            window_size      = WINDOW_SIZE,
            stride           = STRIDE,
            sensor_type      = SENSOR_TYPE,
            normalize        = normalize,
            filter_op_column = filter_op_col,
        )

        if not windows:
            print(f"[{name}] ❌ 유효한 윈도우 없음 — 건너뜀", file=sys.stderr)
            return {"name": name, "error": "유효 윈도우 없음"}

        # 클래스 분포 (증강 전 원본 기준)
        cls_dist: Dict[str, int] = {c: 0 for c in CLASS_NAMES}
        for _, lbl in windows:
            cls_dist[CLASS_NAMES[lbl]] += 1

        n_channels = windows[0][0].shape[1]

        # ── 2. 채널 증강 ─────────────────────────────────────────────────
        pre_aug = n_channels
        if augment_mode == "mixed":
            windows = _add_mixed_feature_channels(
                windows,
                channels         = channels,
                add_accel_fft        = ADD_FFT_CHANNELS,
                add_accel_derivative = ADD_DERIVATIVE_CHANNELS,
                add_torque_derivative= ADD_DERIVATIVE_CHANNELS,
                add_torque_stats     = ADD_TORQUE_STATS,
                add_abs              = ADD_ABS_CHANNELS,
            )
        else:
            windows = _add_feature_channels(
                windows,
                add_fft        = ADD_FFT_CHANNELS,
                add_derivative = ADD_DERIVATIVE_CHANNELS,
                add_abs        = ADD_ABS_CHANNELS,
            )
        n_channels = windows[0][0].shape[1]
        print(
            f"[{name}] 채널 증강({augment_mode}): {pre_aug} → {n_channels}채널",
            file=sys.stderr,
        )

        # ── 3. 학습 ──────────────────────────────────────────────────────
        model, win_acc, seg_acc, epochs_done = train(
            params     = params,
            windows    = windows,
            n_classes  = len(CLASS_NAMES),
            n_channels = n_channels,
            seg_ids    = seg_ids if seg_ids else None,
        )

        # ── 4. ONNX 내보내기 + 메타 저장 ─────────────────────────────────
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        export_onnx(model, output_path, WINDOW_SIZE, n_channels)
        save_meta(
            output_path        = output_path,
            params             = params,
            class_names        = CLASS_NAMES,
            channels           = channels,
            val_accuracy       = win_acc,
            epochs_trained     = epochs_done,
            n_channels_override= n_channels,
        )

        elapsed = time.time() - t0
        _banner(
            f"[{name}] 완료  Win={win_acc*100:.2f}%"
            + (f"  Seg={seg_acc*100:.2f}%" if seg_acc is not None else "")
            + f"  {int(elapsed//60)}m{int(elapsed%60):02d}s"
        )

        return {
            "name":     name,
            "windows":  len(windows),
            "cls_dist": cls_dist,
            "win_acc":  win_acc,
            "seg_acc":  seg_acc,
            "epochs":   epochs_done,
            "elapsed":  elapsed,
            "output":   output_stem + ".onnx",
        }

    except Exception as exc:
        traceback.print_exc()
        return {"name": name, "error": str(exc)}


# ═══════════════════════════════════════════════════════════════════════════════
# 결과 표 출력
# ═══════════════════════════════════════════════════════════════════════════════

_GW  = 14   # 그룹명
_WW  =  8   # 윈도우 수
_CW  =  9   # 클래스별 샘플 수
_AW  =  8   # 정확도
_EW  =  6   # 에폭
_TW  =  7   # 경과 시간

def _table_width() -> int:
    return _GW + _WW + len(CLASS_NAMES) * (_CW + 2) + _AW * 2 + _EW + _TW + 20

def _header_row() -> str:
    cls_hdr = "  ".join(f"{c[:_CW]:>{_CW}}" for c in CLASS_NAMES)
    return (
        f"  {'그룹':<{_GW}} {'윈도우':>{_WW}}  {cls_hdr}"
        f"  {'Win Acc':>{_AW}}  {'Seg Acc':>{_AW}}  {'Epochs':>{_EW}}  {'경과시간':>{_TW}}"
    )

def _data_row(r: dict) -> str:
    cls_str = "  ".join(f"{r['cls_dist'].get(c, 0):>{_CW}}" for c in CLASS_NAMES)
    t = r.get("elapsed", 0)
    t_str = f"{int(t//60)}m{int(t%60):02d}s"
    win_str = f"{r['win_acc']*100:.2f}%" if r.get("win_acc") is not None else "  N/A  "
    seg_str = f"{r['seg_acc']*100:.2f}%" if r.get("seg_acc") is not None else "  N/A  "
    return (
        f"  {r['name']:<{_GW}} {r['windows']:>{_WW}}  {cls_str}"
        f"  {win_str:>{_AW}}  {seg_str:>{_AW}}  {r['epochs']:>{_EW}}  {t_str:>{_TW}}"
    )

def _avg_row(label: str, rows: List[dict]) -> str:
    if not rows:
        return ""
    avg_win = sum(r["win_acc"] for r in rows) / len(rows)
    seg_vals = [r["seg_acc"] for r in rows if r.get("seg_acc") is not None]
    avg_seg  = sum(seg_vals) / len(seg_vals) if seg_vals else None
    total_w  = sum(r["windows"] for r in rows)
    blank_cls = "  ".join(" " * _CW for _ in CLASS_NAMES)
    seg_str  = f"{avg_seg*100:.2f}%" if avg_seg is not None else "  N/A  "
    return (
        f"  {label:<{_GW}} {total_w:>{_WW}}  {blank_cls}"
        f"  {avg_win*100:.2f}%  {seg_str:>{_AW}}"
    )

def _banner(msg: str) -> None:
    print(f"\n{'─'*72}", file=sys.stderr)
    print(f"  {msg}", file=sys.stderr)
    print(f"{'─'*72}", file=sys.stderr)


def print_summary(results: List[dict]) -> None:
    W   = _table_width()
    SEP = "═" * W
    DIV = "─" * W

    ok      = [r for r in results if "error" not in r]
    failed  = [r for r in results if "error" in r]
    accel_ok    = [r for r in ok if "accel"    in r["name"].lower()]
    torque_ok   = [r for r in ok if "torque"   in r["name"].lower()]
    combined_ok = [r for r in ok if "combined" in r["name"].lower()]

    print(f"\n{SEP}")
    print(f"  PHM 통합 학습 결과   ({len(ok)}/{len(results)} 성공)")
    print(SEP)
    print(_header_row())
    print(f"  {DIV}")

    # 가속도 그룹
    if accel_ok:
        for r in accel_ok:
            print(_data_row(r))
        print(f"  {DIV}")
        print(_avg_row("가속도 평균", accel_ok))
        print(f"  {DIV}")

    # 토크 그룹
    if torque_ok:
        for r in torque_ok:
            print(_data_row(r))
        print(f"  {DIV}")
        print(_avg_row("토크 평균", torque_ok))
        print(f"  {DIV}")

    # Combined 그룹
    if combined_ok:
        for r in combined_ok:
            print(_data_row(r))
        print(f"  {DIV}")
        print(_avg_row("Combined 평균", combined_ok))
        print(f"  {DIV}")

    # 전체 평균
    if ok:
        print(_avg_row("전체 평균", ok))
    print(SEP)

    # 실패 목록
    if failed:
        print(f"\n  ❌ 실패한 작업 ({len(failed)}개):")
        for r in failed:
            print(f"     {r['name']}: {r['error']}")
        print()

    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    t_total = time.time()

    print(f"\n{'═'*72}", file=sys.stderr)
    print(f"  PHM 통합 학습 시작  ({len(JOBS)}개 모델)", file=sys.stderr)
    print(f"  DATA_DIR   = {DATA_DIR}", file=sys.stderr)
    print(f"  OUTPUT_DIR = {OUTPUT_DIR}", file=sys.stderr)
    print(f"  window={WINDOW_SIZE}  stride={STRIDE}  epochs={EPOCHS}  "
          f"lr={LR}  batch={BATCH_SIZE}", file=sys.stderr)
    print(f"{'═'*72}\n", file=sys.stderr)

    results: List[dict] = []
    for idx, job in enumerate(JOBS, 1):
        name, channels, filter_op, output_stem = job[0], job[1], job[2], job[3]
        augment_mode = job[4] if len(job) > 4 else DEFAULT_AUGMENT_MODE
        normalize    = job[5] if len(job) > 5 else DEFAULT_NORMALIZE

        print(f"\n[{idx}/{len(JOBS)}]", file=sys.stderr)
        result = run_job(name, channels, filter_op, output_stem, augment_mode, normalize)
        results.append(result)

    total_elapsed = time.time() - t_total
    print_summary(results)
    print(f"  전체 소요 시간: {int(total_elapsed//60)}분 {int(total_elapsed%60):02d}초\n")


if __name__ == "__main__":
    main()
