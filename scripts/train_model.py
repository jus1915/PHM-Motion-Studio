"""
train_model.py - PHM 대시보드용 ML 모델 학습 및 ONNX 변환 스크립트

사용법:
    python train_model.py --params <params_json_path>

params JSON 구조:
{
    "csv": "features.csv 경로",
    "output": "model.onnx 출력 경로",
    "session": "AD" | "FD",
    "model": "knn" | "isoforest" | "ocsvm" | "svm" | "rf" | "gbm" | "mlp",
    "k": 5,
    "standardize": true,
    "threshold": 0.0,       // AD 전용 임계값
    "features": ["feat1", "feat2", ...],
    "class_names": ["Normal", "Anomaly"]  // or FD 클래스명
}

결과: stdout에 JSON 출력 (accuracy, info 필드)
      output 경로에 ONNX 파일 생성

의존성: 첫 실행 시 자동 설치됨 (pip 필요)
    scikit-learn skl2onnx onnx numpy
"""

import sys
import subprocess


def _ensure_packages():
    """필요한 패키지가 없으면 자동으로 pip 설치합니다."""
    REQUIRED = [
        ("numpy", "numpy"),
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
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet"] + missing,
                stdout=subprocess.DEVNULL,
            )
            print("[setup] 설치 완료", file=sys.stderr)
        except subprocess.CalledProcessError as e:
            print(f"[setup] 설치 실패: {e}", file=sys.stderr)
            sys.exit(1)


_ensure_packages()

import argparse
import json
import os
import numpy as np
import csv


def load_params(params_path: str) -> dict:
    # utf-8-sig: BOM 포함/미포함 모두 처리
    with open(params_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_csv(csv_path: str, feature_keys: list):
    """CSV에서 특징 데이터와 레이블을 읽습니다."""
    X, labels = [], []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = [float(row[k]) for k in feature_keys]
                labels.append(row.get("Label", "").strip())
                X.append(x)
            except (KeyError, ValueError):
                continue
    return np.array(X, dtype=np.float32), labels


def encode_labels_ad(labels):
    """AD: Anomaly=1, 그 외=0"""
    return np.array([1 if l.lower() == "anomaly" else 0 for l in labels], dtype=np.int64)


def encode_labels_fd(labels, class_names: list):
    """FD: 클래스명 → 정수 인덱스"""
    name_to_id = {n.lower(): i for i, n in enumerate(class_names)}
    result = []
    for l in labels:
        idx = name_to_id.get(l.lower(), len(class_names))
        result.append(idx)
    return np.array(result, dtype=np.int64)


def build_pipeline(model_key: str, session: str, params: dict):
    """sklearn 파이프라인을 구성합니다."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler() if params.get("standardize", True) else None
    k = int(params.get("k", 5))

    estimator = None
    if session == "AD":
        if model_key == "knn":
            from sklearn.neighbors import LocalOutlierFactor
            # LOF는 novelty=True로 AD 사용 (score_samples 지원)
            estimator = LocalOutlierFactor(n_neighbors=k, novelty=True)
        elif model_key == "isoforest":
            from sklearn.ensemble import IsolationForest
            estimator = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
        elif model_key == "ocsvm":
            from sklearn.svm import OneClassSVM
            estimator = OneClassSVM(nu=0.1, kernel="rbf", gamma="scale")
        else:
            from sklearn.ensemble import IsolationForest
            estimator = IsolationForest(n_estimators=100, random_state=42)
    else:  # FD
        if model_key == "knn":
            from sklearn.neighbors import KNeighborsClassifier
            estimator = KNeighborsClassifier(n_neighbors=k)
        elif model_key == "svm":
            from sklearn.svm import SVC
            estimator = SVC(kernel="rbf", probability=True, random_state=42)
        elif model_key == "rf":
            from sklearn.ensemble import RandomForestClassifier
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_key == "gbm":
            from sklearn.ensemble import GradientBoostingClassifier
            estimator = GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif model_key == "mlp":
            from sklearn.neural_network import MLPClassifier
            estimator = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
        else:
            from sklearn.ensemble import RandomForestClassifier
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)

    if scaler is not None:
        return Pipeline([("scaler", scaler), ("model", estimator)])
    else:
        return Pipeline([("model", estimator)])


def to_onnx(pipeline, n_features: int, output_path: str, session: str, class_names: list):
    """sklearn 파이프라인을 ONNX로 변환합니다."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_type = [("float_input", FloatTensorType([None, n_features]))]

    # AD 모델은 이진 분류 래퍼 적용
    if session == "AD":
        model_to_convert = pipeline
    else:
        model_to_convert = pipeline

    options = {}
    # SVC, RF 등 확률 출력 옵션
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    final_est = pipeline.steps[-1][1]
    if hasattr(final_est, "predict_proba"):
        options = {type(final_est): {"zipmap": False}}

    onnx_model = convert_sklearn(
        model_to_convert,
        initial_types=initial_type,
        options=options,
        target_opset=15,
    )

    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())


def main():
    parser = argparse.ArgumentParser(description="PHM 모델 학습 및 ONNX 변환")
    parser.add_argument("--params", required=True, help="파라미터 JSON 파일 경로")
    args = parser.parse_args()

    params = load_params(args.params)

    csv_path = params["csv"]
    output_path = params["output"]
    session = params.get("session", "AD").upper()
    model_key = params.get("model", "knn").lower()
    feature_keys = params.get("features", [])
    class_names = params.get("class_names", ["Normal", "Anomaly"])

    if not feature_keys:
        print(json.dumps({"error": "features 목록이 비어 있습니다."}))
        sys.exit(1)

    # 데이터 로드
    X, labels = load_csv(csv_path, feature_keys)
    if len(X) == 0:
        print(json.dumps({"error": "CSV에서 유효한 샘플을 읽지 못했습니다."}))
        sys.exit(1)

    # 레이블 인코딩
    if session == "AD":
        y = encode_labels_ad(labels)
        # AD: 정상(0)만 학습
        normal_mask = y == 0
        X_train = X[normal_mask]
        if len(X_train) == 0:
            print(json.dumps({"error": "정상 레이블 샘플이 없습니다."}))
            sys.exit(1)
        y_train = y[normal_mask]
    else:
        y = encode_labels_fd(labels, class_names)
        X_train = X
        y_train = y

    n_features = X_train.shape[1]

    # 파이프라인 구성 및 학습
    pipeline = build_pipeline(model_key, session, params)

    if session == "AD":
        # AD 모델은 정상 데이터만으로 학습
        pipeline.fit(X_train)
        # predict() → 1=정상, -1=이상 (IsolationForest/OneClassSVM/LOF 공통)
        # 학습 데이터의 정상 분류율을 표시 (임계값 무관)
        try:
            preds = pipeline.predict(X_train)  # 학습셋 기준
            n_normal = int(np.sum(preds == 1))
            n_total = len(X_train)
            normal_rate = n_normal / n_total if n_total > 0 else 0.0
            accuracy = normal_rate  # 정상샘플 중 정상으로 판정된 비율
            info = (f"학습={n_total} 정상샘플 | "
                    f"정상 분류율={n_normal}/{n_total} ({100*normal_rate:.1f}%)")
        except Exception as ex:
            accuracy = None
            info = f"학습={len(X_train)} 정상샘플"
    else:
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_train)
        accuracy = float(np.mean(pred == y_train))
        info = f"학습={len(X_train)} 샘플, 클래스={len(set(y_train))}"

    # ONNX 변환
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    to_onnx(pipeline, n_features, output_path, session, class_names)

    # _meta.json 사이드카 저장 (DashboardForm에서 SKL 모델 메타데이터 읽기용)
    meta = {
        "session": session,
        "model_type": model_key,
        "features": feature_keys,
        "class_names": class_names,
        "y_column": params.get("y_column", ""),
        "threshold": params.get("threshold", 0.0),
        "n_features": n_features,
    }
    meta_path = os.path.splitext(output_path)[0] + "_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    result = {"info": info}
    if accuracy is not None:
        result["accuracy"] = accuracy
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
