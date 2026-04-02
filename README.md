# PHM-Motion-Studio

> 산업용 모션 제어 + PHM(Prognostics and Health Management) 데이터 수집·분석 통합 플랫폼

WMX3(SoftServo) 또는 Ajin AMP 모션 컨트롤러를 제어하면서, NI-DAQ 기반 가속도 센서와 토크 데이터를 실시간으로 수집·로깅하고 이상 탐지·AI 추론까지 수행하는 Windows 데스크탑 애플리케이션입니다.

---

## 주요 기능

| 카테고리 | 내용 |
|---|---|
| **모션 제어** | WMX3 / Ajin AMP / 시뮬레이션 3가지 제어기 선택 지원 |
| **데이터 수집** | NI-DAQ 가속도(CSV 저장) + WMX3 토크 동시 로깅 |
| **실시간 스트리밍** | HTTP로 가속도 프레임 전송 (Python 추론 서버 연동) |
| **데이터 분석** | 전처리 / 이상 탐지 / AI 추론 / PHM 파이프라인 마법사 |
| **레이아웃 유지** | DockPanel 기반 다중 창, 창 위치 자동 저장·복원 |

---

## 빠른 시작

```bash
git clone https://github.com/jus1915/PHM-Motion-Studio.git
cd PHM-Motion-Studio
```

Visual Studio 2022에서 `PHM_Project_DockPanel.sln`을 열고 **x64 / Debug** 구성으로 빌드합니다.

> SDK DLL(`lib/`)과 NuGet 패키지는 별도 설치 없이 자동으로 복원됩니다.

---

## 아키텍처

```
PHM-Motion-Studio
├── lib/                           # 서드파티 SDK DLL (Git 관리)
│   ├── WMX3/                      # SoftServo WMX3 CLR 라이브러리
│   ├── Ajin/                      # Ajin AXL.dll
│   └── NationalInstruments/       # NI-DAQmx
│
├── Controller/
│   ├── IMotionController.cs       # 제어기 공통 인터페이스
│   ├── ControllerManager.cs       # IMotionController 주입·관리
│   ├── WMX3Controller.cs          # SoftServo WMX3 구현
│   ├── AjinController.cs          # Ajin AMP (AXL.dll) 구현
│   ├── SimulationController.cs    # 하드웨어 없는 시뮬레이션 모드
│   └── ControllerSelectDialog.cs  # 시작 시 제어기 선택 다이얼로그
│
├── Services/
│   ├── WMX/
│   │   └── WmxTorqueLogger.cs     # WMX3 토크 로거
│   └── DAQ/
│       ├── DaqAccelCsvLogger.cs   # NI-DAQ 가속도 → CSV 저장
│       └── DaqAccelHttpSender.cs  # NI-DAQ 가속도 → HTTP 실시간 전송
│
├── Windows/                       # DockContent 폼들
│   ├── AxisInfoForm
│   ├── TeachingForm
│   ├── SimulatorForm
│   ├── LogWriterForm
│   └── LogGraphForm
│
└── UI/
    ├── DataAnalysis/              # 전처리, 이상탐지, AI, 파이프라인 마법사
    └── Dashboard/                 # 실시간 추론 대시보드
```

### 제어기 추상화

하드웨어 종류에 관계없이 `IMotionController` 인터페이스 하나로 동작합니다. 앱 시작 시 다이얼로그에서 선택하면 `ControllerManager`에 주입됩니다.

```
앱 시작
   └─► ControllerSelectDialog
            ├─► WMX3Controller       (SoftServo WMX3)
            ├─► AjinController       (Ajin AXL.dll)
            └─► SimulationController (하드웨어 없음)
                      │
                      ▼
              ControllerManager
                      │
                      ▼
                 PHM_Motion
```

---

## 개발 환경

| 항목 | 내용 |
|---|---|
| .NET Framework | 4.8 |
| Visual Studio | 2022 |
| Target Platform | x64, Windows 10/11 |
| UI 프레임워크 | WinForms + [WeifenLuo.WinFormsUI.Docking](https://github.com/dockpanelsuite/dockpanelsuite) |

---

## 실행 모드

### 시뮬레이션 모드 (하드웨어 없음)
하드웨어나 드라이버 설치 없이 바로 실행할 수 있습니다. 앱 시작 시 **"시뮬레이션 (하드웨어 없음)"** 을 선택하면 모든 모션·DAQ 동작이 로그만 남기고 정상 실행됩니다.

### WMX3 모션 컨트롤러
`lib/WMX3/`의 DLL이 자동으로 참조됩니다. 실제 하드웨어 통신을 위해서는 SoftServo WMX3 드라이버가 설치되어 있어야 합니다.

### Ajin AMP 모션 컨트롤러
`lib/Ajin/AXL.dll`이 자동으로 참조됩니다. 실제 하드웨어 통신을 위해서는 EzSoftware UC 드라이버가 설치되어 있어야 합니다.

### NI-DAQ
`lib/NationalInstruments/`의 DLL이 자동으로 참조됩니다. 실제 데이터 수집을 위해서는 NI-DAQmx 드라이버가 필요합니다. 디바이스 이름은 `MainForm.cs`에서 수정하세요.

```csharp
private const string DaqModule  = "cDAQ2Mod1"; // NI MAX 디바이스 이름
private const string DaqChannel = "ai0:2";     // X/Y/Z 채널
```

---

## 데이터 저장 경로

| 데이터 종류 | 기본 경로 |
|---|---|
| 가속도 CSV | `E:\Data\PHM_Logs\Signals\<날짜>_Axis<n>\Accel\` |
| 토크 CSV | `E:\Data\PHM_Logs\Signals\<날짜>_Axis<n>\Torque\` |
| 센서 설정 | `E:\Data\PHM_Logs\Tests\sensitivity.csv` |
| 레이아웃 | `layout.xml` (실행 파일과 동일 경로) |
| 축 설정 | `axis_config.json` (실행 파일과 동일 경로) |

`E:\Data\PHM_Logs` 폴더가 없으면 자동으로 `C:\PHM_Logs`로 대체됩니다.

---

## HTTP 실시간 스트리밍

Python 추론 서버와 연동합니다. `MainForm.cs`에서 서버 주소를 수정하세요.

```csharp
private const string HttpServerUrl = "http://10.100.17.221:8000/api/ingest";
```

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| SampleRate | 1280 Hz | Python `daq_sender.py`와 동일하게 맞출 것 |
| FrameSamples | 64 | 한 번에 전송하는 샘플 수 |
| NumWorkers | 6 | 병렬 전송 워커 수 |

---

## 기여 방법

1. 이 레포를 Fork합니다
2. 기능 브랜치를 생성합니다 (`git checkout -b feature/my-feature`)
3. 변경 사항을 커밋합니다 (`git commit -m 'feat: 기능 설명'`)
4. 브랜치에 Push합니다 (`git push origin feature/my-feature`)
5. Pull Request를 생성합니다

---

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참고하세요.
