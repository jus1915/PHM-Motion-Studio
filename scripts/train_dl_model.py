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
    "window_size": 1024,
    "stride": 512,
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
        ("numpy", "numpy", []),
        ("sklearn", "scikit-learn", []),
        ("onnx", "onnx", []),
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
) -> List[Tuple[np.ndarray, int]]:
    """슬라이딩 윈도우로 (window_array, label_int) 튜플 목록을 생성합니다.

    Args:
        signal: shape (N, C) float32 배열
        label_int: 정수 레이블
        window_size: 윈도우 샘플 수 T
        stride: 슬라이딩 스트라이드

    Returns:
        [(window_np, label_int), ...] — 각 window_np shape (T, C)
    """
    results: List[Tuple[np.ndarray, int]] = []
    n_samples = signal.shape[0]
    for start in range(0, n_samples - window_size + 1, stride):
        window = signal[start : start + window_size].copy()
        window = _zscore_normalize(window)
        results.append((window, label_int))
    return results


def _read_signal_csv(
    path: str,
    channels: List[str],
    label_column: str,
) -> Tuple[np.ndarray, Optional[str]]:
    """단일 CSV 파일에서 채널 신호와 레이블 열을 읽습니다.

    Args:
        path: CSV 파일 경로
        channels: 사용할 컬럼명 목록
        label_column: 레이블 컬럼명

    Returns:
        (signal_array, label_str_or_None)
        signal_array shape (N, len(channels)), NaN 행 제거됨
    """
    import csv as _csv

    rows: List[List[float]] = []
    label_value: Optional[str] = None

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = _csv.DictReader(f)
        if reader.fieldnames is None:
            return np.empty((0, len(channels)), dtype=np.float32), None

        has_label = label_column in (reader.fieldnames or [])

        for row in reader:
            try:
                vals = [float(row[c]) for c in channels]
            except (KeyError, ValueError, TypeError):
                continue
            if any(math.isnan(v) or math.isinf(v) for v in vals):
                continue
            rows.append(vals)

            # 첫 번째 유효 레이블 값 사용 (파일 전체가 동일 레이블이라고 가정)
            if has_label and label_value is None:
                lv = row.get(label_column, "").strip()
                if lv:
                    label_value = lv

    signal = np.array(rows, dtype=np.float32) if rows else np.empty((0, len(channels)), dtype=np.float32)
    return signal, label_value


def load_windows_from_dir(
    data_dir: str,
    channels: List[str],
    label_column: str,
    class_names: List[str],
    window_size: int,
    stride: int,
) -> List[Tuple[np.ndarray, int]]:
    """디렉터리를 재귀 탐색해 모든 CSV에서 윈도우를 추출합니다.

    레이블 우선순위:
        1. CSV 내 label_column 값
        2. CSV 파일의 부모 디렉터리명

    Args:
        data_dir: 루트 디렉터리
        channels: 채널 컬럼명 목록
        label_column: CSV 내 레이블 컬럼명
        class_names: 클래스명 → 정수 인덱스 매핑 기준
        window_size: 윈도우 크기
        stride: 슬라이딩 스트라이드

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

    for csv_path in csv_files:
        signal, label_str = _read_signal_csv(str(csv_path), channels, label_column)

        # 레이블 결정
        if label_str is None:
            label_str = csv_path.parent.name

        label_int = name_to_id.get(label_str.lower())
        if label_int is None:
            print(
                f"[data] 경고: '{label_str}' 은 class_names에 없어 건너뜁니다 — {csv_path.name}",
                file=sys.stderr,
            )
            skipped += 1
            continue

        if signal.shape[0] < window_size:
            print(
                f"[data] 경고: 샘플 수 {signal.shape[0]} < window_size {window_size}, 건너뜁니다 — {csv_path.name}",
                file=sys.stderr,
            )
            skipped += 1
            continue

        windows = _extract_windows(signal, label_int, window_size, stride)
        all_windows.extend(windows)

    print(
        f"[data] 디렉터리 로드 완료: {len(csv_files) - skipped}개 파일, "
        f"{len(all_windows)}개 윈도우 (건너뜀={skipped})",
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

        signal, csv_label = _read_signal_csv(path, channels, label_column)

        # 레이블 우선순위: entry["label"] > CSV 내 label_column
        label_str = forced_label if forced_label else (csv_label or "")
        label_int = name_to_id.get(label_str.lower())
        if label_int is None:
            print(
                f"[data] 경고: '{label_str}' 은 class_names에 없어 건너뜁니다 — {Path(path).name}",
                file=sys.stderr,
            )
            skipped += 1
            continue

        if signal.shape[0] < window_size:
            print(
                f"[data] 경고: 샘플 수 {signal.shape[0]} < window_size {window_size}, 건너뜁니다 — {Path(path).name}",
                file=sys.stderr,
            )
            skipped += 1
            continue

        windows = _extract_windows(signal, label_int, window_size, stride)
        all_windows.extend(windows)

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

    아키텍처:
        Conv1d(C→32,  k=7, pad=3) → BN → ReLU → MaxPool(2)
        Conv1d(32→64, k=5, pad=2) → BN → ReLU → MaxPool(2)
        Conv1d(64→128,k=3, pad=1) → BN → ReLU → MaxPool(2)
        Conv1d(128→256,k=3,pad=1) → BN → ReLU
        AdaptiveAvgPool1d(1) → Flatten → Linear(256 → n_classes)
    """

    def __init__(self, n_channels: int, n_classes: int) -> None:
        super().__init__()
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            # Block 4
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(256, n_classes)

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
    patience = 10

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
    print(f"[train] 디바이스: {device}", file=sys.stderr)
    print(
        f"[train] 학습={len(train_idx)} 샘플, 검증={len(val_idx)} 샘플",
        file=sys.stderr,
    )

    model = CNN1DClassifier(n_channels=n_channels, n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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

    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch", 1: "time"},
            "logits": {0: "batch"},
        },
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
    val_accuracy: float,
    epochs_trained: int,
    mlflow_run_id: Optional[str] = None,
    mlflow_tracking_uri: Optional[str] = None,
) -> str:
    """ONNX 파일 옆에 _meta.json 사이드카를 저장합니다.

    Returns:
        저장된 _meta.json 경로
    """
    meta = {
        "kind": "CNN1D",
        "session": "FD",
        "y_column": channels[0] if channels else "",
        "channels": channels,
        "n_channels": len(channels),
        "class_names": class_names,
        "n_classes": len(class_names),
        "window_size": int(params.get("window_size", 1024)),
        "input_name": "input",
        "output_name": "logits",
        "standardize_per_sample": True,
        "val_accuracy": round(val_accuracy, 6),
        "epochs_trained": epochs_trained,
    }
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
        mlflow_mod.log_artifact(output_path)
        mlflow_mod.log_artifact(meta_path)
        mlflow_mod.pytorch.log_model(model, artifact_path="pytorch_model")
        mlflow_mod.end_run()
        run_id = mlflow_run.info.run_id
        print(f"[mlflow] run 종료: {run_id}", file=sys.stderr)
        return run_id
    except Exception as e:
        print(f"[mlflow] 아티팩트 업로드 실패 (건너뜀): {e}", file=sys.stderr)
        try:
            mlflow_mod.end_run()
        except Exception:
            pass
        return None


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
    window_size: int = int(params.get("window_size", 1024))
    stride: int = int(params.get("stride", 512))

    n_channels = len(channels)
    n_classes = len(class_names)

    print(
        f"[main] 채널={channels}, 클래스={class_names}, "
        f"window_size={window_size}, stride={stride}",
        file=sys.stderr,
    )

    # ── 데이터 로드 ───────────────────────────────────────────────────────────
    windows: List[Tuple[np.ndarray, int]] = []

    if "csv_files" in params and params["csv_files"]:
        windows = load_windows_from_file_list(
            csv_files=params["csv_files"],
            channels=channels,
            label_column=label_column,
            class_names=class_names,
            window_size=window_size,
            stride=stride,
        )
    elif "data_dir" in params and params["data_dir"]:
        windows = load_windows_from_dir(
            data_dir=params["data_dir"],
            channels=channels,
            label_column=label_column,
            class_names=class_names,
            window_size=window_size,
            stride=stride,
        )
    else:
        print(json.dumps({"error": "params에 'data_dir' 또는 'csv_files' 중 하나가 필요합니다."}))
        sys.exit(1)

    if len(windows) == 0:
        print(json.dumps({"error": "유효한 윈도우를 하나도 추출하지 못했습니다. 데이터 경로와 채널 설정을 확인하십시오."}))
        sys.exit(1)

    # 클래스 분포 출력
    label_counts: Dict[int, int] = {}
    for _, lbl in windows:
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    dist_str = ", ".join(
        f"{class_names[i] if i < len(class_names) else i}={c}"
        for i, c in sorted(label_counts.items())
    )
    print(f"[main] 클래스 분포: {dist_str}", file=sys.stderr)

    if len(label_counts) < 2:
        print(json.dumps({"error": "학습에 필요한 클래스가 2개 미만입니다."}))
        sys.exit(1)

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
    mlflow_tracking_uri = params.get("mlflow_tracking_uri") or os.environ.get("MLFLOW_TRACKING_URI", "")
    meta_path = save_meta(
        output_path=output_path,
        params=params,
        class_names=class_names,
        channels=channels,
        val_accuracy=best_val_acc,
        epochs_trained=epochs_trained,
        mlflow_run_id=mlflow_run.info.run_id if mlflow_run else None,
        mlflow_tracking_uri=mlflow_tracking_uri or None,
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
