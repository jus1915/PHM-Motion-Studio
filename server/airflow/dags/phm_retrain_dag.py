"""
PHM 모션 스튜디오 — 주기적 재학습 Airflow DAG
=================================================
기본 스케줄: 매일 새벽 2시 (schedule_interval='0 2 * * *')
  - 환경변수 PHM_RETRAIN_SCHEDULE 로 재정의 가능
  - None 으로 설정하면 수동 트리거 전용

C# AIForm 에서 POST /api/v1/dags/phm_retrain/dagRuns 로 즉시 트리거,
  dag_run.conf 에 학습 파라미터를 포함해 전달합니다.

dag_run.conf 주요 파라미터:
  train_modes     : 실행할 학습 모드 목록 (기본 ["accel","torque","combined"])
                    예) ["combined"] 이면 결합 모델만 학습
  axis_count      : 학습 대상 축 수 (기본 0 → CSV 헤더 자동 감지)
  session         : "CLS" (결함진단 분류, 기본) | "AE" (이상탐지)
  data_dir        : 수집 CSV 루트 (Windows 경로는 PHM_DATA_ROOT 로 자동 변환)
  window_size     : 윈도우 크기 (기본 128)
  epochs          : 학습 에폭 (기본 150)
  class_names     : 분류 클래스 목록 (기본 ["normal","overload","looseness"])

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
_SCHEDULE    = os.getenv("PHM_RETRAIN_SCHEDULE", "0 2 * * *")   # 매일 새벽 2시
_SCRIPTS_DIR = Path(os.getenv(
    "PHM_SCRIPTS_DIR",
    Path(__file__).resolve().parents[1],  # dags/ 의 부모 = scripts/
))
_SCRIPT_PATH = _SCRIPTS_DIR / "train_dl_model.py"

_DATA_ROOT     = os.getenv("PHM_DATA_ROOT",     "/opt/phm/data")
_MODELS_ROOT   = os.getenv("PHM_MODELS_ROOT",   "/opt/phm/models")
_INFERENCE_URL = os.getenv("PHM_INFERENCE_URL", "http://phm-inference:8000")

# C# 앱이 conf 를 전달하지 않을 때 사용하는 기본값
_DEFAULT_CONF: dict = {
    "data_dir":         _DATA_ROOT,
    "label_column":     "Label",
    "class_names":      ["normal", "overload", "looseness"],
    "window_size":      128,
    "stride":           32,
    "epochs":           150,
    "batch_size":       64,
    "lr":               0.0005,
    "val_split":        0.2,
    "seed":             42,
    "label_smoothing":  0.0,
    "session":          "CLS",
    # 기본 학습 모드: accel / torque / combined 모두 실행
    # CLS 분류: accel / torque / combined
    # AE 이상탐지: ae_accel (단일 전역) / ae_torque (축별)
    "train_modes":      ["accel", "torque", "combined", "ae_accel", "ae_torque"],
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


# ── session / output prefix 헬퍼 ─────────────────────────────────────────────
def _resolve_session(params: dict) -> str:
    """session 값을 정규화합니다 (AD→AE, FD→CLS)."""
    raw = str(params.get("session", "CLS")).upper()
    return {"AD": "AE", "FD": "CLS"}.get(raw, raw)


def _output_prefix(session: str, sensor: str) -> str:
    """session·sensor 에 따른 모델 파일명 prefix를 반환합니다.

    CLS + accel    → cls_accel
    CLS + torque   → cls_torque
    CLS + combined → cls_combined
    AE  + accel    → ae_fd
    AE  + torque   → ae_torque
    AE  + combined → ae_combined
    """
    if session == "CLS":
        if sensor == "accel":
            return "cls_accel"
        if sensor == "torque":
            return "cls_torque"
        return "cls_combined"
    else:
        if sensor == "accel":
            return "ae_accel"   # 단일 전역 모델 (per-axis 없음)
        if sensor == "torque":
            return "ae_torque"
        return "ae_combined"


def _get_axis_count(conf: dict) -> int:
    """conf 에서 axis_count 를 꺼내거나 CSV 스캔으로 자동 감지합니다."""
    raw = int(conf.pop("axis_count", 0))
    if raw > 0:
        print(f"[PHM] axis_count conf 지정: {raw}개 축", flush=True)
        return raw
    data_dir = _normalize_data_dir(conf.get("data_dir", _DATA_ROOT))
    return _detect_axis_count(data_dir)


def _is_mode_enabled(conf: dict, mode: str) -> bool:
    """train_modes 목록에 mode 가 포함되어 있으면 True."""
    modes = conf.get("train_modes", _DEFAULT_CONF["train_modes"])
    if isinstance(modes, str):
        modes = [m.strip() for m in modes.split(",")]
    return mode in [str(m).lower() for m in modes]


# ── 태스크 함수 ───────────────────────────────────────────────────────────────

def run_training_accel(**context) -> None:
    """
    가속도 전용 CLS 학습 태스크 — 축별 per-axis 모델 학습.

    channels     = ["x", "y", "z"]
    augment_mode = "standard"  (FFT + derivative + z-score 정규화)
    normalize    = True

    출력: cls_accel_ax0.onnx, cls_accel_ax1.onnx, ...
    """
    conf = dict(context["dag_run"].conf or {})

    # train_modes 필터
    if not _is_mode_enabled(conf, "accel"):
        print("[PHM] train_modes 에 'accel' 없음 → 가속도 학습 건너뜀", flush=True)
        return

    axis_count = _get_axis_count(conf)
    run_id     = str(context.get("run_id", "manual"))

    for ax in range(axis_count):
        print(f"\n[PHM] ━━━ 가속도 Ax{ax} 학습 시작 ({ax+1}/{axis_count}) ━━━", flush=True)
        params = {**_DEFAULT_CONF, **conf}
        params["sensor_type"]      = "accel"
        params["channels"]         = ["x", "y", "z"]
        params["filter_op_column"] = f"Op_Ax{ax}"
        params["augment_mode"]     = "standard"
        params["normalize"]        = True
        params.setdefault("session", "CLS")
        session = _resolve_session(params)
        prefix  = _output_prefix(session, "accel")
        params["output"] = str(Path(_MODELS_ROOT) / f"{prefix}_ax{ax}.onnx")
        print(f"[PHM] 출력 파일: {params['output']}  (session={session})", flush=True)
        _execute_training(params, f"{run_id}_accel_ax{ax}")

    print(f"\n[PHM] 가속도 축별 학습 완료 (총 {axis_count}개 축)", flush=True)


def run_training_torque(**context) -> None:
    """
    토크 전용 CLS 학습 태스크 — 축별 per-axis 모델 학습.

    channels     = ["Ax{n}_Trq(%)"]
    augment_mode = "mixed"  (derivative + stats, 정규화 없음)
    normalize    = False

    출력: cls_torque_ax0.onnx, cls_torque_ax1.onnx, ...
    """
    conf = dict(context["dag_run"].conf or {})

    # train_modes 필터
    if not _is_mode_enabled(conf, "torque"):
        print("[PHM] train_modes 에 'torque' 없음 → 토크 학습 건너뜀", flush=True)
        return

    axis_count = _get_axis_count(conf)
    run_id     = str(context.get("run_id", "manual"))

    for ax in range(axis_count):
        print(f"\n[PHM] ━━━ 토크 Ax{ax} 학습 시작 ({ax+1}/{axis_count}) ━━━", flush=True)
        params = {**_DEFAULT_CONF, **conf}
        params["sensor_type"]      = "torque"
        params["channels"]         = [f"Ax{ax}_Trq(%)"]
        params["filter_op_column"] = f"Op_Ax{ax}"
        params["augment_mode"]     = "mixed"
        params["normalize"]        = False
        params.setdefault("session", "CLS")
        session = _resolve_session(params)
        prefix  = _output_prefix(session, "torque")
        params["output"] = str(Path(_MODELS_ROOT) / f"{prefix}_ax{ax}.onnx")
        print(f"[PHM] 출력 파일: {params['output']}  (session={session})", flush=True)
        _execute_training(params, f"{run_id}_torque_ax{ax}")

    print(f"\n[PHM] 토크 축별 학습 완료 (총 {axis_count}개 축)", flush=True)


def run_training_combined(**context) -> None:
    """
    결합(가속도 + 토크) CLS 학습 태스크 — 축별 per-axis 모델 학습.

    channels     = ["x", "y", "z", "Ax{n}_Trq(%)"]
    augment_mode = "mixed"  (accel→FFT+deriv / torque→deriv+stats, 정규화 없음)
    normalize    = False

    출력: cls_combined_ax0.onnx, cls_combined_ax1.onnx, ...
    """
    conf = dict(context["dag_run"].conf or {})

    # train_modes 필터
    if not _is_mode_enabled(conf, "combined"):
        print("[PHM] train_modes 에 'combined' 없음 → 결합 학습 건너뜀", flush=True)
        return

    axis_count = _get_axis_count(conf)
    run_id     = str(context.get("run_id", "manual"))

    for ax in range(axis_count):
        print(f"\n[PHM] ━━━ 결합 Ax{ax} 학습 시작 ({ax+1}/{axis_count}) ━━━", flush=True)
        params = {**_DEFAULT_CONF, **conf}
        params["sensor_type"]      = "combined"
        params["channels"]         = ["x", "y", "z", f"Ax{ax}_Trq(%)"]
        params["filter_op_column"] = f"Op_Ax{ax}"
        params["augment_mode"]     = "mixed"
        params["normalize"]        = False
        params.setdefault("session", "CLS")
        session = _resolve_session(params)
        prefix  = _output_prefix(session, "combined")
        params["output"] = str(Path(_MODELS_ROOT) / f"{prefix}_ax{ax}.onnx")
        print(f"[PHM] 출력 파일: {params['output']}  (session={session})", flush=True)
        _execute_training(params, f"{run_id}_combined_ax{ax}")

    print(f"\n[PHM] 결합 축별 학습 완료 (총 {axis_count}개 축)", flush=True)


def run_training_ae_accel(**context) -> None:
    """
    AE 이상탐지 — 가속도 단일 전역 모델 학습 (축 구분 없음).

    · channels          = ["x", "y", "z"]
    · filter_op_column  = None  (Idle/Pos 구분 없이 전체 학습)
    · augment_mode      = "standard"
    · normalize         = True
    · 출력: ae_accel.onnx  (단일, 축 suffix 없음)
    """
    conf = dict(context["dag_run"].conf or {})

    if not _is_mode_enabled(conf, "ae_accel"):
        print("[PHM] train_modes 에 'ae_accel' 없음 → AE 가속도 학습 건너뜀", flush=True)
        return

    run_id = str(context.get("run_id", "manual"))
    params = {**_DEFAULT_CONF, **conf}
    params["session"]           = "AE"
    params["sensor_type"]       = "accel"
    params["channels"]          = ["x", "y", "z"]
    params["filter_op_column"]  = None   # Op 컬럼 없는 CSV 도 허용, 전체 행 학습
    params["augment_mode"]      = "standard"
    params["normalize"]         = True
    # 단일 모델: 축 suffix 없음
    params["output"] = str(Path(_MODELS_ROOT) / "ae_accel.onnx")
    print(f"[PHM] AE 가속도 단일 모델 출력: {params['output']}", flush=True)
    _execute_training(params, f"{run_id}_ae_accel")
    print("[PHM] AE 가속도 학습 완료", flush=True)


def run_training_ae_torque(**context) -> None:
    """
    AE 이상탐지 — 토크 축별 모델 학습.

    · channels          = ["Ax{n}_Trq(%)"]
    · filter_op_column  = None  (Idle/Pos 구분 없이 전체 학습)
    · augment_mode      = "mixed"
    · normalize         = False
    · 출력: ae_torque_ax0.onnx, ae_torque_ax1.onnx, ...
    """
    conf = dict(context["dag_run"].conf or {})

    if not _is_mode_enabled(conf, "ae_torque"):
        print("[PHM] train_modes 에 'ae_torque' 없음 → AE 토크 학습 건너뜀", flush=True)
        return

    axis_count = _get_axis_count(conf)
    run_id     = str(context.get("run_id", "manual"))

    for ax in range(axis_count):
        print(f"\n[PHM] ━━━ AE 토크 Ax{ax} 학습 시작 ({ax+1}/{axis_count}) ━━━", flush=True)
        params = {**_DEFAULT_CONF, **conf}
        params["session"]           = "AE"
        params["sensor_type"]       = "torque"
        params["channels"]          = [f"Ax{ax}_Trq(%)"]
        params["filter_op_column"]  = None   # Op 컬럼 없는 CSV 도 허용, 전체 행 학습
        params["augment_mode"]      = "mixed"
        params["normalize"]         = False
        params["output"] = str(Path(_MODELS_ROOT) / f"ae_torque_ax{ax}.onnx")
        print(f"[PHM] 출력 파일: {params['output']}", flush=True)
        _execute_training(params, f"{run_id}_ae_torque_ax{ax}")

    print(f"\n[PHM] AE 토크 축별 학습 완료 (총 {axis_count}개 축)", flush=True)


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
    description="PHM 모션 스튜디오 — DL 모델 주기적 재학습 (accel / torque / combined 병렬)",
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
            "가속도 신호(x/y/z) CLS 분류 모델 학습. "
            "augment_mode=standard. 출력: cls_accel_ax{n}.onnx"
        ),
    )

    t_torque = PythonOperator(
        task_id="train_torque",
        python_callable=run_training_torque,
        doc_md=(
            "토크 신호 CLS 분류 모델 학습. "
            "augment_mode=mixed. 출력: cls_torque_ax{n}.onnx"
        ),
    )

    t_combined = PythonOperator(
        task_id="train_combined",
        python_callable=run_training_combined,
        doc_md=(
            "가속도+토크 결합 CLS 분류 모델 학습. "
            "augment_mode=mixed. 출력: cls_combined_ax{n}.onnx"
        ),
    )

    t_ae_accel = PythonOperator(
        task_id="train_ae_accel",
        python_callable=run_training_ae_accel,
        doc_md=(
            "가속도 AE 이상탐지 모델 학습 (단일 전역, 축 구분 없음). "
            "augment_mode=standard, filter_op=None. 출력: ae_accel.onnx"
        ),
    )

    t_ae_torque = PythonOperator(
        task_id="train_ae_torque",
        python_callable=run_training_ae_torque,
        doc_md=(
            "토크 AE 이상탐지 모델 학습 (축별). "
            "augment_mode=mixed, filter_op=None. 출력: ae_torque_ax{n}.onnx"
        ),
    )

    t_reload = PythonOperator(
        task_id="reload_inference_cache",
        python_callable=reload_inference_cache,
        doc_md="학습 완료 후 추론 서버(phm-inference:8000)의 모델 캐시를 재로드.",
    )

    # CLS(accel/torque/combined) + AE(ae_accel/ae_torque) 병렬 학습 → 완료 후 캐시 재로드
    [t_accel, t_torque, t_combined, t_ae_accel, t_ae_torque] >> t_reload
