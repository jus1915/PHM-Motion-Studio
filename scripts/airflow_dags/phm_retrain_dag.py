"""
PHM 모션 스튜디오 — 주기적 재학습 Airflow DAG
=================================================
기본 스케줄: 매일 새벽 2시 (schedule_interval='0 2 * * *')
  - 환경변수 PHM_RETRAIN_SCHEDULE 로 재정의 가능
  - None 으로 설정하면 수동 트리거 전용

C# AIForm 에서 POST /api/v1/dags/phm_retrain/dagRuns 로 즉시 트리거,
  dag_run.conf 에 train_dl_model.py 파라미터를 포함해 전달합니다.

기대 디렉터리 구조 (기본값):
  C:/Data/PHM_Logs/Signals/{label}/Accel/*.csv
  C:/Data/PHM_Logs/Signals/{label}/Torque/*.csv

사용법:
  1. 이 파일을 Airflow dags/ 폴더에 복사 (또는 심볼릭 링크)
  2. PHM_SCRIPTS_DIR 환경변수를 scripts/ 폴더 경로로 설정
  3. Airflow 웹 UI 에서 phm_retrain DAG 활성화
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

# C# 앱이 conf 를 전달하지 않을 때 사용하는 기본값
_DEFAULT_CONF: dict = {
    "data_dir":    r"C:\Data\PHM_Logs\Signals",
    "output":      r"C:\Data\PHM_Logs\models\cnn1d_fd.onnx",
    "channels":    ["x", "y", "z"],
    "sensor_type": "accel",
    "label_column": "",
    "class_names": ["normal", "fault"],
    "window_size": 1024,
    "stride":      512,
    "epochs":      30,
    "batch_size":  32,
    "lr":          0.001,
    "val_split":   0.2,
    "seed":        42,
    "session":     "FD",
}

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
    """
    conf: dict = context["dag_run"].conf or {}

    # C# 가 전달한 conf 를 기본값 위에 덮어씀
    params = {**_DEFAULT_CONF, **conf}

    # 출력 폴더 생성
    output_path = Path(params.get("output", _DEFAULT_CONF["output"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # params JSON 임시 파일 저장
    run_id = context.get("run_id", "manual")
    params_file = Path(tempfile.gettempdir()) / f"phm_airflow_{run_id}.json"
    params_file.write_text(
        json.dumps(params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if not _SCRIPT_PATH.exists():
        raise FileNotFoundError(f"train_dl_model.py 없음: {_SCRIPT_PATH}")

    # 가상환경 python 우선 사용 (없으면 현재 인터프리터)
    venv_python = _SCRIPTS_DIR / ".venv" / "Scripts" / "python.exe"
    python = str(venv_python) if venv_python.exists() else sys.executable

    cmd = [python, str(_SCRIPT_PATH), "--params", str(params_file)]
    print(f"[PHM] 학습 시작: {' '.join(cmd)}", flush=True)
    print(f"[PHM] 파라미터: {params_file}", flush=True)

    result = subprocess.run(
        cmd,
        capture_output=False,
        text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"train_dl_model.py 실패 (exit={result.returncode}). "
            f"로그를 확인하세요."
        )

    print(f"[PHM] 학습 완료 → {output_path}", flush=True)


def run_training_accel(**context) -> None:
    """가속도 전용 학습 태스크 (기본 conf 에서 sensor_type=accel 강제)."""
    if context["dag_run"].conf:
        context["dag_run"].conf["sensor_type"] = "accel"
    run_training(**context)


def run_training_torque(**context) -> None:
    """토크 전용 학습 태스크 (기본 conf 에서 sensor_type=torque 강제)."""
    conf = context["dag_run"].conf or {}
    conf["sensor_type"] = "torque"
    conf.setdefault("channels", ["Trq(%)"])
    conf.setdefault("output", r"C:\Data\PHM_Logs\models\cnn1d_torque.onnx")
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
