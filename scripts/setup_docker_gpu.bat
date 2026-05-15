@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ============================================================
echo  PHM-Motion-Studio 서버 GPU Docker 빌드
echo  대상: airflow-scheduler (Dockerfile.airflow-gpu)
echo ============================================================
echo.

REM ── 설정 ────────────────────────────────────────────────────
set "DEFAULT_IP=10.100.17.127"
set "DEFAULT_USER=ubuntu"
set "DEFAULT_COMPOSE_DIR=~/PHM-Motion-Studio/server"

set /p SERVER_IP="서버 IP [기본: %DEFAULT_IP%]: "
if "!SERVER_IP!"=="" set "SERVER_IP=%DEFAULT_IP%"

set /p SERVER_USER="SSH 사용자명 [기본: %DEFAULT_USER%]: "
if "!SERVER_USER!"=="" set "SERVER_USER=%DEFAULT_USER%"

set /p COMPOSE_DIR="docker-compose 디렉토리 [기본: %DEFAULT_COMPOSE_DIR%]: "
if "!COMPOSE_DIR!"=="" set "COMPOSE_DIR=%DEFAULT_COMPOSE_DIR%"

echo.
echo [대상] !SERVER_USER!@!SERVER_IP!:!COMPOSE_DIR!
echo.

REM ── ssh.exe 확인 ────────────────────────────────────────────
where ssh >nul 2>&1
if errorlevel 1 (
    echo [오류] ssh.exe 를 찾을 수 없습니다.
    echo        Windows 설정 > 앱 > 선택적 기능 > OpenSSH 클라이언트를 설치하세요.
    pause & exit /b 1
)

REM ── [1/4] 서버 연결 확인 ────────────────────────────────────
echo [1/4] 서버 연결 확인 중...
ssh -o ConnectTimeout=10 -o BatchMode=yes ^
    !SERVER_USER!@!SERVER_IP! "echo connected" >nul 2>&1
if errorlevel 1 (
    echo [오류] SSH 연결 실패: !SERVER_USER!@!SERVER_IP!
    echo.
    echo  해결 방법:
    echo    1. 서버가 켜져 있는지 확인
    echo    2. SSH 키 등록:  ssh-keygen -t ed25519
    echo                     ssh-copy-id !SERVER_USER!@!SERVER_IP!
    echo    3. 또는 패스워드 인증 허용 시 BatchMode 없이 재시도
    pause & exit /b 1
)
echo        연결 성공.

REM ── [2/4] NVIDIA Container Toolkit 확인 ────────────────────
echo [2/4] nvidia-container-toolkit 확인 중...
ssh !SERVER_USER!@!SERVER_IP! ^
    "which nvidia-container-toolkit 2>/dev/null || dpkg -l nvidia-container-toolkit 2>/dev/null | grep -q 'ii' && echo FOUND || echo NOT_FOUND" > "%TEMP%\phm_nct_check.txt" 2>&1
set /p NCT_RESULT=<"%TEMP%\phm_nct_check.txt"
if "!NCT_RESULT!"=="NOT_FOUND" (
    echo [경고] nvidia-container-toolkit 이 설치되지 않았습니다.
    echo        GPU 학습이 불가능할 수 있습니다.
    echo        설치: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
    echo.
    set /p CONT="계속 진행하시겠습니까? (y/N): "
    if /i not "!CONT!"=="y" ( pause & exit /b 1 )
) else (
    echo        nvidia-container-toolkit 확인됨.
)

REM ── [3/4] Docker 이미지 빌드 ────────────────────────────────
echo.
echo [3/4] airflow-scheduler GPU 이미지 빌드 중...
echo        (CUDA 12.1 PyTorch ~2.5 GB — 첫 빌드 시 수 분 소요)
echo.
ssh -t !SERVER_USER!@!SERVER_IP! ^
    "cd !COMPOSE_DIR! && docker compose build airflow-scheduler"
if errorlevel 1 (
    echo.
    echo [오류] Docker 빌드 실패.
    pause & exit /b 1
)

REM ── [4/4] 컨테이너 재시작 ────────────────────────────────────
echo.
echo [4/4] airflow-scheduler 컨테이너 재시작 중...
ssh -t !SERVER_USER!@!SERVER_IP! ^
    "cd !COMPOSE_DIR! && docker compose up -d airflow-scheduler"
if errorlevel 1 (
    echo.
    echo [오류] 컨테이너 시작 실패.
    pause & exit /b 1
)

REM ── GPU 검증 ────────────────────────────────────────────────
echo.
echo [확인] GPU 사용 가능 여부 (약 30초 대기)...
timeout /t 20 /nobreak >nul
ssh !SERVER_USER!@!SERVER_IP! ^
    "docker exec phm_airflow_scheduler python -c ""import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"" 2>&1"

echo.
echo ============================================================
echo  빌드 완료!
echo  Airflow 스케줄러가 GPU 학습 환경으로 실행 중입니다.
echo  Airflow UI: http://!SERVER_IP!:8080
echo ============================================================
echo.
pause
endlocal
