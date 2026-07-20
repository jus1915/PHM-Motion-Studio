"""
PHM 모션 스튜디오 — 자동 주기 재학습 Airflow DAG
=================================================
phm_retrain(C# AIForm "지금 트리거" 버튼 전용, 수동 트리거만)과 별개로, 정해진
스케줄에 따라 자동으로 이상탐지 모델을 재학습합니다.

On/Off:
  C# AIForm의 "자동 재학습" 체크박스가 이 DAG의 일시정지(is_paused) 상태를
  PATCH /api/v1/dags/phm_auto_retrain 로 전환합니다 (AirflowClient.SetDagPausedAsync).
  On/Off는 "정해진 주기대로 실제로 돌게 둘지"만 결정합니다 — Airflow 스케줄러가
  서버에서 계속 관리하므로 C# 앱이 꺼져 있어도 On 상태면 그대로 실행됩니다.
  배포 직후에는 is_paused_upon_creation=True 로 항상 꺼진 상태로 시작하며,
  사용자가 AIForm에서 명시적으로 켜야 실행됩니다.

주기(스케줄): Airflow Variable "phm_auto_retrain_schedule" 값(cron 표현식 또는
  "@daily" 같은 매크로)을 사용합니다. C# AIForm의 "주기" 입력 + "적용" 버튼이
  PATCH/POST /api/v1/variables/phm_auto_retrain_schedule 로 이 값을 갱신합니다
  (AirflowClient.SetVariableAsync). Variable이 아직 없으면 환경변수
  PHM_AUTO_RETRAIN_SCHEDULE(기본 "0 2 * * *" = 매일 새벽 2시)을 기본값으로 씁니다.

  ⚠ Airflow는 DAG 파일을 주기적으로만 재파싱하므로(기본 수 분 간격), 주기를
  바꿔도 즉시 반영되지 않고 다음 재스캔 때 반영됩니다 — AIForm에도 이 점을
  안내합니다.

기본 학습 대상:
  CLS(결함진단 분류)는 레이블이 있는 데이터가 필요해 자동으로 확보할 수 없으므로
  제외하고, "연속 수집"으로 무레이블 데이터가 계속 쌓이는 것만으로 재학습 가능한
  이상탐지(AE-CNN + IsolationForest) 모델만 자동 재학습 대상으로 삼습니다.
    - ae_accel, ae_torque_global, ae_torque, ae_combined_global, ae_combined
    - isoforest_accel
  환경변수 PHM_AUTO_RETRAIN_MODES (콤마 구분 문자열)로 재정의할 수 있습니다.

실제 학습 로직은 phm_retrain_dag.py 의 태스크 함수를 그대로 재사용합니다
(축 감지, 프로파일 디렉토리, 학습 스크립트 실행 등 ~800줄 로직 중복 방지).
수동으로 conf를 지정해 이 DAG를 트리거하면(예: 테스트 목적) 그 값이 항상
우선이며, 스케줄 실행처럼 conf가 비어 있을 때만 위 기본 train_modes가 채워집니다.

환경변수 (phm_retrain_dag.py 와 공유):
  PHM_SCRIPTS_DIR, PHM_DATA_ROOT, PHM_MODELS_ROOT, PHM_INFERENCE_URL
  PHM_AUTO_RETRAIN_SCHEDULE : Variable "phm_auto_retrain_schedule" 미설정 시 기본값 (기본: 0 2 * * *)
  PHM_AUTO_RETRAIN_MODES    : 콤마 구분 train_modes (기본: 위 6개 이상탐지 모델)

Airflow Variable:
  phm_auto_retrain_schedule : cron 식 또는 "@daily" 등 매크로. AIForm에서 갱신.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator

# phm_retrain_dag.py 를 같은 dags/ 폴더의 모듈로 임포트하기 위해 sys.path에 추가
# (Airflow 버전에 따라 dags 폴더가 항상 sys.path에 자동으로 잡히지는 않으므로 명시적으로 보강)
_DAGS_DIR = Path(__file__).resolve().parent
if str(_DAGS_DIR) not in sys.path:
    sys.path.insert(0, str(_DAGS_DIR))

from phm_retrain_dag import (  # noqa: E402  (sys.path 보강 이후 임포트)
    run_training_ae_accel,
    run_training_ae_torque,
    run_training_ae_torque_global,
    run_training_ae_combined_global,
    run_training_ae_combined,
    run_training_isoforest_accel,
    reload_inference_cache,
)

# ── 기본 설정 ────────────────────────────────────────────────────────────────
# 스케줄은 Airflow Variable로 관리 — AIForm의 "주기" 입력이 이 Variable을 갱신하면
# Airflow가 DAG 파일을 다음에 재파싱할 때 새 스케줄이 반영된다(코드 배포/재시작 불필요).
# Variable이 아직 없으면(최초 배포 등) 환경변수 → 하드코드 기본값 순으로 폴백한다.
# 주: DAG 최상위 코드에서 Variable.get()을 호출하면 매 파싱마다 DB 조회가 발생하지만,
# 이 DAG 하나뿐이고 파싱 주기가 짧지 않아(기본 수십 초~수 분) 이 규모에선 무리 없다.
_AUTO_RETRAIN_SCHEDULE_VAR = "phm_auto_retrain_schedule"
_SCHEDULE = Variable.get(
    _AUTO_RETRAIN_SCHEDULE_VAR,
    default_var=os.getenv("PHM_AUTO_RETRAIN_SCHEDULE", "0 2 * * *"),
)

_DEFAULT_AUTO_MODES = [
    "ae_accel",
    "ae_torque_global", "ae_torque",
    "ae_combined_global", "ae_combined",
    "isoforest_accel",
]
_env_modes = os.getenv("PHM_AUTO_RETRAIN_MODES")
_AUTO_MODES = (
    [m.strip() for m in _env_modes.split(",") if m.strip()]
    if _env_modes else _DEFAULT_AUTO_MODES
)


def _with_auto_defaults(func):
    """
    conf에 train_modes가 이미 있으면(수동 트리거) 그대로 두고, 스케줄 실행처럼
    conf가 비어 있을 때만 자동 재학습 기본 train_modes(_AUTO_MODES)를 채워 넣는
    래퍼입니다. phm_retrain_dag.py의 각 태스크 함수는 매번 새로
    dict(context["dag_run"].conf or {}) 로 conf를 읽으므로, 같은 task 실행
    안에서 context["dag_run"].conf 를 먼저 덮어써 두면 아래 원본 함수 호출에
    그대로 반영됩니다.
    """
    def _wrapped(**context):
        dag_run = context["dag_run"]
        conf = dict(dag_run.conf or {})
        if "train_modes" not in conf:
            conf["train_modes"] = _AUTO_MODES
            dag_run.conf = conf
            print(f"[PHM][auto_retrain] conf에 train_modes 없음(스케줄 실행) → "
                  f"기본값 적용: {_AUTO_MODES}", flush=True)
        return func(**context)
    _wrapped.__name__ = getattr(func, "__name__", "wrapped")
    return _wrapped


_default_args = {
    "owner":            "phm",
    "depends_on_past":  False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=10),
    "email_on_failure": False,
}

with DAG(
    dag_id="phm_auto_retrain",
    description="연속 수집 데이터로 이상탐지(AE-CNN+IsolationForest) 모델을 주기적으로 자동 재학습 "
                "— On/Off는 AIForm의 '자동 재학습' 체크박스(DAG 일시정지 전환)로 제어",
    default_args=_default_args,
    schedule_interval=_SCHEDULE,
    # 고정된 정적 날짜를 사용해야 함 — days_ago(1) 같은 동적 start_date는 DAG 파일이
    # 재파싱될 때마다(수십 초~수 분 간격) 값이 계속 바뀌어, 실제 스케줄이 있는 DAG에서는
    # 스케줄러의 "다음 실행" 계산이 갱신되지 않고 과거 시각에 멈춰버리는 문제가 있다
    # (Airflow 공식 문서에서 명시적으로 경고하는 안티패턴). catchup=False이므로 과거로
    # 밀리지 않고 "지금 이후"부터 정상적으로 스케줄링된다.
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=True,   # 배포 직후엔 항상 꺼진 상태 — 사용자가 AIForm에서 명시적으로 켜야 실행
    tags=["phm", "ml", "retrain", "auto"],
    doc_md=__doc__,
) as dag:

    t_ae_accel = PythonOperator(
        task_id="ae_accel_auto",
        python_callable=_with_auto_defaults(run_training_ae_accel),
    )
    t_ae_torque_global = PythonOperator(
        task_id="ae_torque_global_auto",
        python_callable=_with_auto_defaults(run_training_ae_torque_global),
    )
    t_ae_torque = PythonOperator(
        task_id="ae_torque_auto",
        python_callable=_with_auto_defaults(run_training_ae_torque),
    )
    t_ae_combined_global = PythonOperator(
        task_id="ae_combined_global_auto",
        python_callable=_with_auto_defaults(run_training_ae_combined_global),
    )
    t_ae_combined = PythonOperator(
        task_id="ae_combined_auto",
        python_callable=_with_auto_defaults(run_training_ae_combined),
    )
    t_isoforest_accel = PythonOperator(
        task_id="isoforest_accel_auto",
        python_callable=_with_auto_defaults(run_training_isoforest_accel),
    )
    t_reload = PythonOperator(
        task_id="reload_inference_cache",
        python_callable=reload_inference_cache,
    )

    [
        t_ae_accel, t_ae_torque_global, t_ae_torque,
        t_ae_combined_global, t_ae_combined, t_isoforest_accel,
    ] >> t_reload
