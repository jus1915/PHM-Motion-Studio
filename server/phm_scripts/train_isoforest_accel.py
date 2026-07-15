"""
train_isoforest_accel.py — 가속도(x/y/z) RobustScaler + IsolationForest 이상탐지 학습

DashboardForm의 AE-CNN1D(ae_accel.onnx)와 "선택 가능한 대안 모델"로 나란히 쓰기
위한 스크립트. CSV 파싱/구간 분리는 train_dl_model.py의 로더를 그대로 재사용해
기존 AE-Accel 모델과 동일한 학습 데이터 모집단을 보장하고, 특징 추출은
stat_features.py를 inference_server.py와 공유해서 학습/추론 전처리가 절대
어긋나지 않게 한다.

사용법:
    python train_isoforest_accel.py --params <params_json_path>

params JSON 구조:
{
    "data_dir": "C:/Data/PHM_Logs/raw",       // CSV 디렉터리 (재귀 탐색), OR
    "csv_files": [                             // 명시적 파일 목록
        {"path": "C:/Data/.../normal.csv", "label": "normal"}
    ],
    "output": "C:/Models/isoforest_accel/ae_accel.onnx",
    "channels": ["x", "y", "z"],
    "window_size": 500,                        // train.py 기준: WIN_S=0.5s @ fs=1000Hz
    "stride": 250,                              // train.py 기준: STEP_S=0.25s @ fs=1000Hz
    "n_estimators": 400,
    "contamination": 0.01,
    "fs": null,                                // null이면 CSV time_s에서 자동 감지
    "seed": 42,
    "mlflow_tracking_uri": "http://localhost:5000",  // 선택
    "mlflow_experiment": "PHM-IsoForest"             // 선택
}

결과: stdout에 JSON 출력 {"info": "...", "threshold_99": ..., "threshold_995": ...}
      output 경로에 ONNX 파일 및 _meta.json 사이드카 생성 (inference_server.py가
      kind="IF-STAT"로 인식)

의존성 (자동 설치): numpy scipy scikit-learn skl2onnx onnx
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def _ensure_packages() -> None:
    REQUIRED = [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("sklearn", "scikit-learn"),
        ("skl2onnx", "skl2onnx"),
        ("onnx", "onnx"),
    ]
    missing = []
    for import_name, pip_name in REQUIRED:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        print(f"[setup] 패키지 자동 설치: {', '.join(missing)}", file=sys.stderr)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing,
            stdout=subprocess.DEVNULL,
        )
        print("[setup] 설치 완료", file=sys.stderr)


_ensure_packages()

import numpy as np  # noqa: E402

# train_dl_model.py / stat_features.py 와 같은 디렉터리에서 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stat_features import extract_features_from_segment  # noqa: E402
from train_dl_model import load_windows_from_dir, load_windows_from_file_list  # noqa: E402


def load_params(params_path: str) -> dict:
    with open(params_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def _detect_fs(csv_path: str, fallback: float = 1000.0) -> float:
    """CSV의 time_s 컬럼 간격 중앙값으로 샘플링 주파수를 추정한다 (train.py와 동일 방식)."""
    import csv as _csv

    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = _csv.DictReader(f)
            if not reader.fieldnames or "time_s" not in reader.fieldnames:
                return fallback
            times = []
            for i, row in enumerate(reader):
                try:
                    times.append(float(row["time_s"]))
                except (KeyError, ValueError, TypeError):
                    continue
                if i > 500:
                    break
        if len(times) < 2:
            return fallback
        diffs = np.diff(np.asarray(times, dtype=np.float64))
        diffs = diffs[diffs > 0]
        if len(diffs) == 0:
            return fallback
        return float(1.0 / np.median(diffs))
    except Exception:
        return fallback


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="가속도 IsolationForest 이상탐지 모델 학습")
    parser.add_argument("--params", required=True, help="파라미터 JSON 파일 경로")
    args = parser.parse_args()

    params = load_params(args.params)

    channels = params.get("channels", ["x", "y", "z"])
    # 기본값은 train.py 기준(WIN_S=0.5s, STEP_S=0.25s @ fs=1000Hz) — 500/250샘플
    window_size = int(params.get("window_size", 500))
    stride = int(params.get("stride", 250))
    output_path = params["output"]
    n_estimators = int(params.get("n_estimators", 400))
    contamination = params.get("contamination", 0.01)
    seed = int(params.get("seed", 42))

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    # train_dl_model.py의 로더를 그대로 써서 기존 ae_accel(AE-CNN1D)와
    # 동일한 CSV 파싱 규칙을 공유한다.
    # normalize=False: 원본 진폭 보존 (RMS/AbsMax 등 진폭 기반 특징에 필수)
    # use_op_filter=False: Op_Ax*/Op 컬럼의 Idle 상태로 구간을 끊지 않는다 — 이 프로젝트의
    # 다른 accel/torque AE 학습(run_training_ae_torque 등)도 전부 이렇게 하며, train.py
    # 원본도 애초에 구간 분리 없이 파일 전체를 그대로 슬라이딩한다. 이걸 안 끄면 실제
    # 연속수집 데이터(정지-구동이 잦음)가 Idle마다 잘게 쪼개져서 window_size 이상인
    # 구간이 거의 안 남는다 (실측: 2906구간 중 2898개가 500샘플 미만으로 스킵됨).
    first_csv = None
    if params.get("csv_files"):
        windows, _ = load_windows_from_file_list(
            csv_files=params["csv_files"],
            channels=channels,
            label_column="Label",
            class_names=["normal"],
            window_size=window_size,
            stride=stride,
            normalize=False,
            filter_op_column=None,
            use_op_filter=False,
        )
        if params["csv_files"]:
            first_csv = params["csv_files"][0].get("path")
    else:
        data_dir = params["data_dir"]
        windows, _ = load_windows_from_dir(
            data_dir=data_dir,
            channels=channels,
            label_column="Label",
            class_names=["normal"],
            window_size=window_size,
            stride=stride,
            sensor_type="accel",
            normalize=False,
            filter_op_column=None,
            use_op_filter=False,
        )
        found = list(Path(data_dir).rglob("*.csv"))
        first_csv = str(found[0]) if found else None

    if not windows:
        print(json.dumps({"error": "유효한 윈도우가 없습니다. data_dir/csv_files와 채널명을 확인하세요."}))
        sys.exit(1)

    fs = params.get("fs")
    fs = float(fs) if fs else (_detect_fs(first_csv) if first_csv else 1000.0)
    print(
        f"[train_isoforest] fs={fs:.3f} Hz  windows={len(windows)}  "
        f"window_size={window_size}  stride={stride}  channels={channels}",
        file=sys.stderr,
    )

    # ── 특징 추출 (stat_features.py — inference_server.py와 동일 함수) ─────────
    rows = []
    for w, _ in windows:
        seg = w.astype(np.float64)
        rows.append(extract_features_from_segment(seg, fs, axes=channels))

    feature_names = list(rows[0].keys())
    X = np.array([[r[k] for k in feature_names] for r in rows], dtype=np.float64)
    print(f"[train_isoforest] 특징 행렬: {X.shape} ({len(feature_names)}개 특징/윈도우)", file=sys.stderr)

    # ── RobustScaler + IsolationForest 학습 ─────────────────────────────────
    from sklearn.ensemble import IsolationForest
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler

    pipeline = Pipeline(
        [
            ("scaler", RobustScaler()),
            (
                "model",
                IsolationForest(
                    n_estimators=n_estimators,
                    contamination=contamination,
                    random_state=seed,
                ),
            ),
        ]
    )
    pipeline.fit(X)

    # ── 중요: score_samples가 아니라 decision_function으로 임계값을 계산한다 ──
    # skl2onnx가 IsolationForest를 변환할 때 "scores" 출력은 decision_function
    # (= score_samples - offset_)이지 score_samples 자체가 아니다. inference_server.py의
    # _if_stat_score()는 ONNX "scores" 출력을 그대로 읽으므로, 여기서 threshold를
    # score_samples 기준으로 계산하면 offset_ 만큼 스케일이 어긋나 추론 시 정상 구간의
    # raw_score가 threshold 근처에도 못 미치는(=탐지 불능) 상태가 된다.
    # (실측: 622개 정상 윈도우에서 score_samples 기준 p99≈0.53, decision_function
    #  기준 p99≈0.05 — 약 10배 스케일 차이가 남을 확인함.)
    scores = -pipeline.decision_function(X)
    threshold_99 = float(np.quantile(scores, 0.99))
    threshold_995 = float(np.quantile(scores, 0.995))
    print(
        f"[train_isoforest] Normal 99% threshold={threshold_99:.4f}  "
        f"99.5% threshold={threshold_995:.4f}",
        file=sys.stderr,
    )

    # ── ONNX 변환 (skl2onnx — train_model.py와 동일 변환 규칙) ──────────────
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    n_features = X.shape[1]
    onnx_model = convert_sklearn(
        pipeline,
        initial_types=[("float_input", FloatTensorType([None, n_features]))],
        target_opset={"": 15, "ai.onnx.ml": 3},
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"[train_isoforest] ONNX 저장 완료: {output_path}", file=sys.stderr)

    # ── _meta.json 사이드카 ──────────────────────────────────────────────────
    # inference_server.py는 kind == "IF-STAT" 를 보고 _if_stat_score() 로 라우팅한다.
    # window_size/activity_threshold는 /model_info 에서 기존 AE 모델과 동일한
    # 방식으로 스캔되므로 클라이언트(C#) 변경 없이 프로파일 전환만으로 교체된다.
    meta = {
        "kind": "IF-STAT",
        "sensor_type": "accel",
        "channels": channels,
        "window_size": window_size,
        "fs": fs,
        "feature_names": feature_names,
        "n_features": n_features,
        "input_name": "float_input",
        "threshold_99": round(threshold_99, 6),
        "threshold_995": round(threshold_995, 6),
        "contamination": contamination,
        "n_estimators": n_estimators,
        "n_train_windows": len(windows),
        "activity_threshold": 0.0,   # accel AE와 동일하게 게이팅 미적용 (Idle 포함 전체 구동 분포 학습)
        "source_file": os.path.basename(output_path),
    }
    meta_path = os.path.splitext(output_path)[0] + "_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[train_isoforest] meta 저장 완료: {meta_path}", file=sys.stderr)

    _try_log_mlflow(params, feature_names, n_features, threshold_99, threshold_995, output_path, meta_path)

    result = {
        "info": f"학습 {len(windows)}개 윈도우, {n_features}개 특징 (fs={fs:.1f}Hz)",
        "threshold_99": threshold_99,
        "threshold_995": threshold_995,
    }
    print(json.dumps(result, ensure_ascii=False))


def _try_log_mlflow(params, feature_names, n_features, threshold_99, threshold_995, output_path, meta_path):
    """MLflow 실험 로깅. mlflow 미설치 또는 tracking URI 미설정이면 조용히 건너뜀."""
    tracking_uri = params.get("mlflow_tracking_uri", "") or os.environ.get("MLFLOW_TRACKING_URI", "")
    if not tracking_uri:
        return
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(params.get("mlflow_experiment", "PHM-IsoForest"))

        with mlflow.start_run(run_name="isoforest-accel"):
            mlflow.log_params(
                {
                    "n_features": n_features,
                    "n_estimators": params.get("n_estimators", 400),
                    "contamination": params.get("contamination", 0.01),
                    "window_size": params.get("window_size", 500),
                    "stride": params.get("stride", 250),
                    "features": ",".join(feature_names),
                }
            )
            mlflow.log_metric("threshold_99", threshold_99)
            mlflow.log_metric("threshold_995", threshold_995)
            mlflow.log_artifact(output_path, artifact_path="model")
            if os.path.exists(meta_path):
                mlflow.log_artifact(meta_path, artifact_path="model")

            print(f"[MLflow] run logged → {tracking_uri}", file=sys.stderr)
    except Exception as e:
        print(f"[MLflow] 로깅 건너뜀: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
