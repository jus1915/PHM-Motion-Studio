"""
PHM 모션 스튜디오 — 주기적 재학습 Airflow DAG
=================================================
기본 스케줄: 매일 새벽 2시 (schedule_interval='0 2 * * *')
  - 환경변수 PHM_RETRAIN_SCHEDULE 로 재정의 가능
  - None 으로 설정하면 수동 트리거 전용

C# AIForm 에서 POST /api/v1/dags/phm_retrain/dagRuns 로 즉시 트리거,
  dag_run.conf 에 train_dl_model.py 파라미터를 포함해 전달합니다.

환경변수:
  PHM_SCRIPTS_DIR      : train_dl_model.py 위치 (기본: /opt/phm/scripts)
  PHM_DATA_ROOT        : 수집 데이터 루트 경로   (기본: /opt/phm/data)
                         C# 앱이 Windows 경로로 보내도 이 값으로 대체됩니다.
  PHM_MODELS_ROOT      : 모델 출력 루트 경로     (기본: /opt/phm/models)
  PHM_RETRAIN_SCHEDULE : cron 식                 (기본: 0 2 * * *)

docker-compose 볼륨 예시:
  - ./phm_scripts:/opt/phm/scripts   # train_dl_model.py
  - ./phm_data:/opt/phm/data         # 수집 CSV
  - ./phm_models:/opt/phm/models     # 출력 ONNX
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

# ── 기본 설정 ────────────────────────────────────────────────────────────────
_SCHEDULE = os.getenv("PHM_RETRAIN_SCHEDULE", "0 2 * * *")   # 매일 새벽 2시
_SCRIPTS_DIR = Path(os.getenv(
    "PHM_SCRIPTS_DIR",
    Path(__file__).resolve().parents[1],  # dags/ 의 부모 = scripts/
))
_SCRIPT_PATH = _SCRIPTS_DIR / "train_dl_model.py"

_DATA_ROOT   = os.getenv("PHM_DATA_ROOT",   "/opt/phm/data")
_MODELS_ROOT = os.getenv("PHM_MODELS_ROOT", "/opt/phm/models")

# C# 앱이 conf 를 전달하지 않을 때 사용하는 기본값
# Windows 절대 경로 대신 Docker 볼륨 마운트 경로를 기본으로 사용합니다.
_DEFAULT_CONF: dict = {
    "data_dir":     _DATA_ROOT,
    "output":       str(Path(_MODELS_ROOT) / "cnn1d_fd.onnx"),
    "channels":     ["x", "y", "z"],
    "sensor_type":  "accel",
    "label_column": "",
    "class_names":  ["normal", "fault"],
    "window_size":  1024,
    "stride":       512,
    "epochs":       30,
    "batch_size":   32,
    "lr":           0.001,
    "val_split":    0.2,
    "seed":         42,
    # "FD" = 분류(2개 이상 클래스 필요), "AD" = AE 이상탐지(단일 클래스 가능)
    # train_dl_model.py가 클래스 부족 시 자동으로 AD로 전환하므로 FD로 시작해도 무방
    "session":      "FD",
}

# Windows 드라이브 패턴 (예: C:\, D:\)
_WIN_DRIVE_RE = __import__("re").compile(r"^[A-Za-z]:[/\\]")


def _normalize_path(value: object) -> object:
    """
    C# 앱에서 전달된 Windows 절대 경로를 Linux 경로로 변환합니다.

    규칙:
      data_dir  → PHM_DATA_ROOT  하위 경로로 재매핑
      output    → PHM_MODELS_ROOT 하위 경로로 재매핑
    단순히 마지막 구성 요소(파일명 또는 마지막 폴더명)만 보존합니다.
    """
    if not isinstance(value, str):
        return value
    if not _WIN_DRIVE_RE.match(value):
        return value  # 이미 Linux 경로 or 상대 경로

    # 경로 마지막 요소만 유지 (예: cnn1d_fd.onnx, Signals)
    tail = Path(value.replace("\\", "/")).name
    return tail  # 호출 측에서 루트와 결합

# ── 기본 인수 ─────────────────────────────────────────────────────────────────
_default_args = {
    "owner":            "phm",
    "depends_on_past":  False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=10),
    "email_on_failure": False,
}


# ── 태스크 함수 ───────────────────────────────────────────────────────────────
def run_training(**context) -> None:
    """
    dag_run.conf 의 params 를 파일로 저장한 뒤
    train_dl_model.py 를 서브프로세스로 실행합니다.

    C# 앱이 Windows 절대 경로를 conf 로 보낼 경우 자동으로 Linux 경로로 변환합니다.
    """
    conf: dict = context["dag_run"].conf or {}

    # C# 가 전달한 conf 를 기본값 위에 덮어씀
    params = {**_DEFAULT_CONF, **conf}

    # ── Windows 경로 → Linux 경로 변환 ─────────────────────────────────────
    raw_data = params.get("data_dir", "")
    if isinstance(raw_data, str) and _WIN_DRIVE_RE.match(raw_data):
        # Windows 절대 경로 → PHM_DATA_ROOT 로 대체
        # (Docker 볼륨이 phm_data 전체를 /opt/phm/data 로 마운트하므로
        #  하위 폴더명을 붙이지 않고 루트를 그대로 사용)
        params["data_dir"] = _DATA_ROOT
        print(f"[PHM] data_dir 변환: {raw_data!r} → {_DATA_ROOT!r}", flush=True)

    raw_out = params.get("output", "")
    if isinstance(raw_out, str) and _WIN_DRIVE_RE.match(raw_out):
        fname = Path(raw_out.replace("\\", "/")).name
        params["output"] = str(Path(_MODELS_ROOT) / fname)
        print(f"[PHM] output 변환: {raw_out!r} → {params['output']!r}", flush=True)

    # 출력 폴더 생성
    output_path = Path(params["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # params JSON 임시 파일 저장
    run_id = context.get("run_id", "manual")
    # run_id 에 슬래시 등이 포함될 수 있으므로 파일명에 안전한 문자만 사용
    safe_run_id = str(run_id).replace("/", "_").replace(":", "-")
    params_file = Path(tempfile.gettempdir()) / f"phm_airflow_{safe_run_id}.json"
    params_file.write_text(
        json.dumps(params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[PHM] 파라미터 파일: {params_file}", flush=True)
    print(f"[PHM] 파라미터 내용:\n{params_file.read_text(encoding='utf-8')}", flush=True)

    if not _SCRIPT_PATH.exists():
        raise FileNotFoundError(f"train_dl_model.py 없음: {_SCRIPT_PATH}")

    # 가상환경 python 우선 사용 (없으면 현재 인터프리터)
    venv_python = _SCRIPTS_DIR / ".venv" / "bin" / "python"   # Linux venv
    venv_python_win = _SCRIPTS_DIR / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        python = str(venv_python)
    elif venv_python_win.exists():
        python = str(venv_python_win)
    else:
        python = sys.executable

    cmd = [python, str(_SCRIPT_PATH), "--params", str(params_file)]
    print(f"[PHM] 학습 명령: {' '.join(cmd)}", flush=True)

    # stdout/stderr 를 파이프로 받아 Airflow 로그에 실시간 출력
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # stderr → stdout 합류
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)

    returncode = proc.wait()

    if returncode != 0:
        raise RuntimeError(
            f"train_dl_model.py 실패 (exit={returncode}). "
            f"위 출력 내용을 확인하세요."
        )

    print(f"[PHM] 학습 완료 → {output_path}", flush=True)


def run_training_accel(**context) -> None:
    """가속도 전용 학습 태스크.

    C# conf 에 channels/output 이 들어와도 accel 고유값으로 강제 덮어씁니다.
    """
    conf = dict(context["dag_run"].conf or {})
    conf["sensor_type"] = "accel"
    # 가속도 채널·출력 경로는 항상 accel 기준으로 강제 (C# conf 무시)
    conf["channels"] = ["x", "y", "z"]
    conf["output"]   = str(Path(_MODELS_ROOT) / "cnn1d_fd.onnx")
    context["dag_run"].conf = conf
    run_training(**context)


def run_training_torque(**context) -> None:
    """토크 전용 학습 태스크.

    C# conf 에 channels/output 이 들어와도 torque 고유값으로 강제 덮어씁니다.
    channels 는 "Trq(%)" 하나만 지정 — train_dl_model.py 의 _resolve_channels 가
    CSV 헤더를 읽어 Ax0_Trq(%)~AxN_Trq(%) 전체로 자동 확장합니다.
    """
    conf = dict(context["dag_run"].conf or {})
    conf["sensor_type"] = "torque"
    # 토크 채널·출력 경로는 항상 torque 기준으로 강제 (C# conf 무시)
    conf["channels"] = ["Trq(%)"]
    conf["output"]   = str(Path(_MODELS_ROOT) / "cnn1d_torque.onnx")
    context["dag_run"].conf = conf
    run_training(**context)


# ── DAG 정의 ──────────────────────────────────────────────────────────────────
with DAG(
    dag_id="phm_retrain",
    description="PHM 모션 스튜디오 — DL 모델 주기적 재학습",
    default_args=_default_args,
    schedule_interval=_SCHEDULE,
    start_date=days_ago(1),
    catchup=False,
    tags=["phm", "ml", "retrain"],
    doc_md=__doc__,
) as dag:

    # C# 에서 conf 로 sensor_type 을 명시하면 단일 task 로 처리
    # sensor_type 미지정 시 가속도 → 토크 순서로 순차 실행
    t_accel = PythonOperator(
        task_id="train_accel",
        python_callable=run_training_accel,
        doc_md="가속도 신호(x/y/z) CNN1D / AE 학습",
    )

    t_torque = PythonOperator(
        task_id="train_torque",
        python_callable=run_training_torque,
        doc_md="토크 신호 CNN1D / AE 학습",
    )

    # 가속도 → 토크 순차 실행 (독립 실행도 가능)
    t_accel >> t_torque
