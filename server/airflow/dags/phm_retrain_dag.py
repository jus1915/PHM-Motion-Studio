"""
PHM 모션 스튜디오 — 주기적 재학습 Airflow DAG
=================================================
기본 스케줄: 매일 새벽 2시 (schedule_interval='0 2 * * *')
  - 환경변수 PHM_RETRAIN_SCHEDULE 로 재정의 가능
  - None 으로 설정하면 수동 트리거 전용

C# AIForm 에서 POST /api/v1/dags/phm_retrain/dagRuns 로 즉시 트리거,
  dag_run.conf 에 train_dl_model.py 파라미터를 포함해 전달합니다.

dag_run.conf 주요 파라미터:
  axis_count      : 가속도 센서 학습 대상 축 수 (기본 1)
                    per-axis 모델: ae_fd_ax0.onnx, ae_fd_ax1.onnx ...
  session         : "AD" (AE 이상탐지, 기본) | "FD" (분류)
  data_dir        : 수집 CSV 루트 (Windows 경로도 자동 변환)
  window_size     : 윈도우 크기 (기본 1024)
  epochs          : 학습 에폭 (기본 30)

환경변수:
  PHM_SCRIPTS_DIR      : train_dl_model.py 위치 (기본: /opt/phm/scripts)
  PHM_DATA_ROOT        : 수집 데이터 루트 경로   (기본: /opt/phm/data)
  PHM_MODELS_ROOT      : 모델 출력 루트 경로     (기본: /opt/phm/models)
  PHM_RETRAIN_SCHEDULE : cron 식                 (기본: 0 2 * * *)
  PHM_INFERENCE_URL    : 추론 서버 URL           (기본: http://phm-inference:8000)

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

_DATA_ROOT       = os.getenv("PHM_DATA_ROOT",       "/opt/phm/data")
_MODELS_ROOT     = os.getenv("PHM_MODELS_ROOT",     "/opt/phm/models")
_INFERENCE_URL   = os.getenv("PHM_INFERENCE_URL",   "http://phm-inference:8000")

# C# 앱이 conf 를 전달하지 않을 때 사용하는 기본값
_DEFAULT_CONF: dict = {
    "data_dir":       _DATA_ROOT,
    "channels":       ["x", "y", "z"],
    "sensor_type":    "accel",
    "label_column":   "",
    # AE 기본: 정상 데이터 폴더명. CLS 사용 시 conf 로 class_names 를 재정의하세요.
    # 예: {"session": "FD", "class_names": ["normal", "looseness"]}
    "class_names":    ["normal", "looseness"],
    "normal_classes": ["normal"],       # AE 모드: 정상으로 취급할 클래스 (학습 대상)
    "window_size":    256,              # 1024 → 256 (학습·추론 동일 윈도우)
    "stride":         128,              # 512  → 128 (오버랩 50%)
    "epochs":         30,
    "batch_size":     32,
    "lr":             0.001,
    "val_split":      0.2,
    "seed":           42,
    # "AE" = AE 이상탐지(normal_classes 만 학습), "CLS" = 분류(class_names 전체)
    # 구버전 호환: "AD"→"AE", "FD"→"CLS" (train_dl_model.py 내부에서 자동 변환)
    "session":        "AE",
}

# Windows 드라이브 패턴 (예: C:\, D:\)
_WIN_DRIVE_RE = __import__("re").compile(r"^[A-Za-z]:[/\\]")


def _normalize_data_dir(raw: str) -> str:
    """Windows 절대 경로이면 PHM_DATA_ROOT 로 대체합니다."""
    if isinstance(raw, str) and _WIN_DRIVE_RE.match(raw):
        print(f"[PHM] data_dir 변환: {raw!r} → {_DATA_ROOT!r}", flush=True)
        return _DATA_ROOT
    return raw


def _normalize_output(raw: str) -> str:
    """Windows 절대 경로이면 PHM_MODELS_ROOT/{파일명} 으로 대체합니다."""
    if isinstance(raw, str) and _WIN_DRIVE_RE.match(raw):
        fname = Path(raw.replace("\\", "/")).name
        result = str(Path(_MODELS_ROOT) / fname)
        print(f"[PHM] output 변환: {raw!r} → {result!r}", flush=True)
        return result
    return raw


# ── 축 수 자동 감지 ───────────────────────────────────────────────────────────
def _detect_axis_count(data_root: str) -> int:
    """
    data_root 하위 CSV 파일의 헤더에서 Op_Ax{n} 컬럼을 스캔해
    최대 축 인덱스 + 1 을 반환합니다.

    발견된 컬럼이 없으면 1 을 반환합니다 (Ax0 단일 축 가정).
    """
    import re as _re
    _pat = _re.compile(r'Op_Ax(\d+)', _re.IGNORECASE)
    max_ax = -1
    try:
        for csv_path in Path(data_root).rglob("*.csv"):
            try:
                with open(csv_path, "r", encoding="utf-8", errors="replace") as _f:
                    header_line = _f.readline()
                for _m in _pat.finditer(header_line):
                    ax = int(_m.group(1))
                    if ax > max_ax:
                        max_ax = ax
                if max_ax >= 0:
                    break   # 첫 번째 발견 CSV 로 충분
            except Exception:
                continue
    except Exception:
        pass
    result = max_ax + 1 if max_ax >= 0 else 1
    print(f"[PHM] axis_count 자동 감지: {result}개 축 (max Op_Ax index={max_ax})", flush=True)
    return result


# ── 핵심 실행 헬퍼 ─────────────────────────────────────────────────────────────
def _execute_training(params: dict, run_id: str) -> None:
    """
    params dict 를 JSON 파일로 저장한 뒤 train_dl_model.py 를 실행합니다.
    학습 결과(로그)를 Airflow 로그에 실시간 출력합니다.
    """
    # Windows 경로 정규화
    params["data_dir"] = _normalize_data_dir(params.get("data_dir", _DATA_ROOT))
    if "output" in params:
        params["output"] = _normalize_output(params["output"])

    # 출력 폴더 생성
    output_path = Path(params["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # params JSON 임시 파일 저장
    safe_id = str(run_id).replace("/", "_").replace(":", "-")
    params_file = Path(tempfile.gettempdir()) / f"phm_airflow_{safe_id}.json"
    params_file.write_text(
        json.dumps(params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[PHM] 파라미터 파일: {params_file}", flush=True)
    print(f"[PHM] 파라미터 내용:\n{params_file.read_text(encoding='utf-8')}", flush=True)

    if not _SCRIPT_PATH.exists():
        raise FileNotFoundError(f"train_dl_model.py 없음: {_SCRIPT_PATH}")

    # 가상환경 python 우선 사용
    venv_python     = _SCRIPTS_DIR / ".venv" / "bin" / "python"
    venv_python_win = _SCRIPTS_DIR / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        python = str(venv_python)
    elif venv_python_win.exists():
        python = str(venv_python_win)
    else:
        python = sys.executable

    cmd = [python, str(_SCRIPT_PATH), "--params", str(params_file)]
    print(f"[PHM] 학습 명령: {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
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
            f"train_dl_model.py 실패 (exit={returncode}). 위 출력을 확인하세요."
        )
    print(f"[PHM] 학습 완료 → {output_path}", flush=True)


# ── 기본 인수 ─────────────────────────────────────────────────────────────────
_default_args = {
    "owner":            "phm",
    "depends_on_past":  False,
    "retries":          1,
    "retry_delay":      timedelta(minutes=10),
    "email_on_failure": False,
}


# ── 태스크 함수 ───────────────────────────────────────────────────────────────
def _resolve_session(params: dict) -> str:
    """session 값을 정규화합니다 (AD→AE, FD→CLS)."""
    raw = str(params.get("session", "AE")).upper()
    return {"AD": "AE", "FD": "CLS"}.get(raw, raw)


def _output_prefix(session: str, sensor: str) -> str:
    """session·sensor_type 에 따른 모델 파일명 prefix를 반환합니다.

    AE  + accel  → ae_fd
    AE  + torque → ae_torque
    CLS + accel  → cls_fd
    CLS + torque → cls_torque
    """
    if session == "CLS":
        return "cls_fd" if sensor == "accel" else "cls_torque"
    else:
        return "ae_fd"  if sensor == "accel" else "ae_torque"


def run_training(**context) -> None:
    """
    dag_run.conf 의 params 를 그대로 train_dl_model.py 에 전달합니다.
    sensor_type / output 을 직접 지정할 때 사용하는 범용 태스크입니다.

    output 미지정 시 session·sensor_type 에 따라 자동 결정:
      AE  + accel  → ae_fd.onnx
      AE  + torque → ae_torque.onnx
      CLS + accel  → cls_fd.onnx
      CLS + torque → cls_torque.onnx
    """
    conf: dict = context["dag_run"].conf or {}
    params = {**_DEFAULT_CONF, **conf}
    if "output" not in params:
        session = _resolve_session(params)
        sensor  = params.get("sensor_type", "accel")
        prefix  = _output_prefix(session, sensor)
        params["output"] = str(Path(_MODELS_ROOT) / f"{prefix}.onnx")
    _execute_training(params, context.get("run_id", "manual"))


def run_training_accel(**context) -> None:
    """
    가속도 전용 학습 태스크 — 축별 per-axis 모델 학습.

    conf 파라미터:
      axis_count  : 학습할 축 수 (기본 0 → CSV 헤더 자동 감지).
      session     : "AE" (기본, AE 이상탐지) | "CLS" (결함진단 분류)
      그 외 _DEFAULT_CONF 참조.

    출력 파일 (session에 따라 자동 결정):
      AE  → ae_fd_ax0.onnx,  ae_fd_ax1.onnx  ...
      CLS → cls_fd_ax0.onnx, cls_fd_ax1.onnx ...
    """
    conf = dict(context["dag_run"].conf or {})
    _raw_ax = int(conf.pop("axis_count", 0))
    run_id  = str(context.get("run_id", "manual"))
    if _raw_ax <= 0:
        _data_dir = _normalize_data_dir(conf.get("data_dir", _DATA_ROOT))
        axis_count = _detect_axis_count(_data_dir)
    else:
        axis_count = _raw_ax
        print(f"[PHM] axis_count conf 지정: {axis_count}개 축", flush=True)

    for ax in range(axis_count):
        print(f"\n[PHM] ━━━ 가속도 Ax{ax} 학습 시작 ({ax+1}/{axis_count}) ━━━", flush=True)
        params = {**_DEFAULT_CONF, **conf}
        params["sensor_type"]      = "accel"
        params["channels"]         = ["x", "y", "z"]
        params["filter_op_column"] = f"Op_Ax{ax}"
        params.setdefault("session", "AE")
        # session에 따라 출력 파일명 결정 (AE→ae_fd, CLS→cls_fd)
        session = _resolve_session(params)
        prefix  = _output_prefix(session, "accel")
        params["output"] = str(Path(_MODELS_ROOT) / f"{prefix}_ax{ax}.onnx")
        print(f"[PHM] 출력 파일: {params['output']}  (session={session})", flush=True)
        _execute_training(params, f"{run_id}_ax{ax}")

    print(f"\n[PHM] 가속도 축별 학습 완료 (총 {axis_count}개 축)", flush=True)


def run_training_torque(**context) -> None:
    """
    토크 전용 학습 태스크 — 축별 per-axis 모델 학습.

    conf 파라미터:
      axis_count : 학습할 축 수 (0 또는 미지정 → CSV 스캔으로 자동 감지).
      session    : "AE" (기본, AE 이상탐지) | "CLS" (결함진단 분류)

    각 축별로:
      channels         = ["Ax{n}_Trq(%)"]   ← 해당 축 토크만
      filter_op_column = "Op_Ax{n}"         ← 해당 축이 움직인 행만

    출력 파일 (session에 따라 자동 결정):
      AE  → ae_torque_ax0.onnx,  ae_torque_ax1.onnx  ...
      CLS → cls_torque_ax0.onnx, cls_torque_ax1.onnx ...
    """
    conf = dict(context["dag_run"].conf or {})
    _raw_ax = int(conf.pop("axis_count", 0))
    run_id  = str(context.get("run_id", "manual"))

    if _raw_ax <= 0:
        _data_dir  = _normalize_data_dir(conf.get("data_dir", _DATA_ROOT))
        axis_count = _detect_axis_count(_data_dir)
    else:
        axis_count = _raw_ax
        print(f"[PHM] torque axis_count conf 지정: {axis_count}개 축", flush=True)

    for ax in range(axis_count):
        print(f"\n[PHM] ━━━ 토크 Ax{ax} 학습 시작 ({ax+1}/{axis_count}) ━━━", flush=True)
        params = {**_DEFAULT_CONF, **conf}
        params["sensor_type"]      = "torque"
        params["channels"]         = [f"Ax{ax}_Trq(%)"]
        params["filter_op_column"] = f"Op_Ax{ax}"
        params.setdefault("session", "AE")
        # session에 따라 출력 파일명 결정 (AE→ae_torque, CLS→cls_torque)
        session = _resolve_session(params)
        prefix  = _output_prefix(session, "torque")
        params["output"] = str(Path(_MODELS_ROOT) / f"{prefix}_ax{ax}.onnx")
        print(f"[PHM] 출력 파일: {params['output']}  (session={session})", flush=True)
        _execute_training(params, f"{run_id}_torque_ax{ax}")

    print(f"\n[PHM] 토크 축별 학습 완료 (총 {axis_count}개 축)", flush=True)


def reload_inference_cache(**context) -> None:
    """
    학습 완료 후 추론 서버의 모델 캐시를 재로드합니다.
    서버가 없거나 응답하지 않으면 경고만 출력하고 성공으로 처리합니다.
    """
    import urllib.request
    import urllib.error

    url = f"{_INFERENCE_URL}/models/reload"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode()
            print(f"[PHM] 추론 서버 캐시 재로드 완료: {body}", flush=True)
    except urllib.error.URLError as e:
        print(f"[PHM] 추론 서버 캐시 재로드 실패 (무시): {e}", flush=True)
    except Exception as e:
        print(f"[PHM] 추론 서버 캐시 재로드 오류 (무시): {e}", flush=True)


# ── DAG 정의 ──────────────────────────────────────────────────────────────────
with DAG(
    dag_id="phm_retrain",
    description="PHM 모션 스튜디오 — DL 모델 주기적 재학습 (per-axis 지원)",
    default_args=_default_args,
    schedule_interval=_SCHEDULE,
    start_date=days_ago(1),
    catchup=False,
    tags=["phm", "ml", "retrain"],
    doc_md=__doc__,
) as dag:

    t_accel = PythonOperator(
        task_id="train_accel",
        python_callable=run_training_accel,
        doc_md=(
            "가속도 신호(x/y/z) AE 이상탐지 모델 학습. "
            "conf.axis_count 만큼 축별(ae_fd_ax{n}.onnx) 순차 학습."
        ),
    )

    t_torque = PythonOperator(
        task_id="train_torque",
        python_callable=run_training_torque,
        doc_md="토크 신호 AE 이상탐지 모델 학습 (ae_torque.onnx, 전축 단일).",
    )

    t_reload = PythonOperator(
        task_id="reload_inference_cache",
        python_callable=reload_inference_cache,
        doc_md="학습 완료 후 추론 서버(phm-inference:8000)의 모델 캐시를 재로드.",
    )

    # 가속도 → 토크 → 추론 서버 캐시 재로드
    t_accel >> t_torque >> t_reload
