@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ============================================================
echo  PHM-Motion-Studio DL 가상환경 설정
echo  위치: %~dp0.venv
echo ============================================================
echo.

REM ── [1/5] Python 탐색 ───────────────────────────────────────────
set "PYTHON_EXE="

REM 1) PATH에 등록된 python (공백 없이 & 앞에 바로 붙여야 trailing space 방지)
where python >nul 2>&1
if not errorlevel 1 (set "PYTHON_EXE=python"& goto :found_python)

REM 2) python3
where python3 >nul 2>&1
if not errorlevel 1 (set "PYTHON_EXE=python3"& goto :found_python)

REM 3) py launcher → 전체 경로로 resolve (따옴표 문제 방지)
where py >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%P in ('where py') do (set "PYTHON_EXE=%%P"& goto :found_python)
)

REM 4) 일반 설치 폴더 직접 탐색
for %%R in (
    "%LOCALAPPDATA%\Programs\Python"
    "%APPDATA%\Programs\Python"
    "C:\Python3"
    "C:\Python"
    "%ProgramFiles%\Python"
    "%ProgramFiles(x86)%\Python"
) do (
    if exist "%%~R" (
        for /d %%D in ("%%~R\Python3*") do (
            if exist "%%D\python.exe" (
                set "PYTHON_EXE=%%D\python.exe"
                goto :found_python
            )
        )
    )
)

REM 5) 레지스트리에서 탐색 (HKCU 우선, HKLM 차선)
for %%H in (HKCU HKLM) do (
    for /f "tokens=2*" %%A in (
        'reg query "%%H\SOFTWARE\Python\PythonCore" /s /v ExecutablePath 2^>nul'
    ) do (
        if exist "%%B" (
            set "PYTHON_EXE=%%B"
            goto :found_python
        )
    )
)

echo [오류] Python을 찾을 수 없습니다.
echo        python.org 에서 설치 후 'Add python.exe to PATH' 를 체크하세요.
pause & exit /b 1

:found_python
for /f "delims=" %%v in ('"!PYTHON_EXE!" --version 2^>^&1') do set "PYVER=%%v"
echo [1/5] Python 확인: !PYVER!
echo        경로: !PYTHON_EXE!

REM ── [2/5] 가상환경 생성 ─────────────────────────────────────────
if exist ".venv\Scripts\python.exe" (
    echo [2/5] 기존 가상환경 사용: .venv
) else (
    echo [2/5] 가상환경 생성 중: .venv
    "!PYTHON_EXE!" -m venv .venv
    if errorlevel 1 ( echo [오류] venv 생성 실패 & pause & exit /b 1 )
)

REM ── [3/5] pip 업그레이드 ────────────────────────────────────────
echo [3/5] pip 업그레이드...
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
if errorlevel 1 ( echo [경고] pip 업그레이드 실패 — 계속 진행합니다. )

REM ── [4/5] 기본 ML 패키지 ────────────────────────────────────────
echo [4/5] 패키지 설치: numpy scikit-learn skl2onnx onnx onnxscript...
.venv\Scripts\pip.exe install --quiet ^
    numpy ^
    scikit-learn ^
    skl2onnx ^
    onnx ^
    onnxscript
if errorlevel 1 ( echo [오류] 패키지 설치 실패 & pause & exit /b 1 )

REM ── [5/5] PyTorch CPU ───────────────────────────────────────────
echo [5/5] PyTorch CPU 설치 중 (수 분 소요)...
.venv\Scripts\pip.exe install --quiet torch ^
    --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 ( echo [오류] PyTorch 설치 실패 & pause & exit /b 1 )

REM ── 선택: MLflow ────────────────────────────────────────────────
set /p INSTALL_MLFLOW="MLflow도 설치할까요? (y/N): "
if /i "!INSTALL_MLFLOW!"=="y" (
    echo [+] MLflow 설치 중...
    .venv\Scripts\pip.exe install --quiet mlflow
)

echo.
echo ============================================================
echo  설치 완료!
echo  AI Form > DL 학습 탭에서 자동으로 이 가상환경을 사용합니다.
echo  가상환경 Python: %~dp0.venv\Scripts\python.exe
echo ============================================================
echo.
pause
endlocal
