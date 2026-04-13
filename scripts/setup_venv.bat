@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo  PHM-Motion-Studio DL 가상환경 설정
echo  위치: %~dp0.venv
echo ============================================================
echo.

REM Python 확인
where python >nul 2>&1
if errorlevel 1 (
    echo [오류] Python이 PATH에 없습니다.
    echo        python.org에서 설치 후 'Add python.exe to PATH' 를 체크하세요.
    pause & exit /b 1
)

for /f "delims=" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [1/5] Python 확인: %PYVER%

REM 가상환경 생성
if exist ".venv\Scripts\python.exe" (
    echo [2/5] 기존 가상환경 사용: .venv
) else (
    echo [2/5] 가상환경 생성 중: .venv
    python -m venv .venv
    if errorlevel 1 ( echo [오류] venv 생성 실패 & pause & exit /b 1 )
)

REM pip 업그레이드
echo [3/5] pip 업그레이드...
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet

REM 기본 ML 패키지
echo [4/5] 패키지 설치: numpy scikit-learn skl2onnx onnx onnxscript...
.venv\Scripts\pip.exe install --quiet ^
    numpy ^
    scikit-learn ^
    skl2onnx ^
    onnx ^
    onnxscript
if errorlevel 1 ( echo [오류] 패키지 설치 실패 & pause & exit /b 1 )

REM PyTorch CPU
echo [5/5] PyTorch CPU 설치 중 (수 분 소요)...
.venv\Scripts\pip.exe install --quiet torch ^
    --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 ( echo [오류] PyTorch 설치 실패 & pause & exit /b 1 )

REM 선택: MLflow
set /p INSTALL_MLFLOW="MLflow도 설치할까요? (y/N): "
if /i "%INSTALL_MLFLOW%"=="y" (
    echo [+] MLflow 설치 중...
    .venv\Scripts\pip.exe install --quiet mlflow
)

echo.
echo ============================================================
echo  설치 완료!
echo  AI Form > DL 학습 탭에서 자동으로 이 가상환경을 사용합니다.
echo ============================================================
echo.
pause
endlocal
