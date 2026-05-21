# PHM 서버 인프라

서버 PC에서 `git pull` 후 이 폴더에서 `docker compose up -d` 로 전체 스택을 실행합니다.

## 디렉터리 구조

```
server/
├── docker-compose.yml          # 전체 서비스 정의
├── influx_config.json          # InfluxDB 연결 설정 (C# 앱용)
├── airflow/
│   └── dags/
│       └── phm_retrain_dag.py  # Airflow 재학습 DAG
├── phm_scripts/
│   ├── train_dl_model.py       # 1D-CNN / AE-CNN 학습 스크립트
│   └── inference_server.py     # FastAPI + ONNX 추론 서버
├── postgres/
│   └── init.sql                # PostgreSQL 초기화 SQL
└── grafana/
    └── provisioning/           # Grafana 대시보드 프로비저닝 (선택)
```

런타임에 자동 생성되는 폴더 (git 미추적):
- `phm_data/`   — C# 앱이 기록한 CSV 파일
- `phm_models/` — 학습된 ONNX 모델
- `airflow/logs/`
- `mlflow/`

## 최초 실행

```bash
cd server

# 1) 필요 폴더 생성
mkdir -p phm_data phm_models airflow/logs mlflow

# 2) 전체 스택 기동
docker compose up -d

# 3) Airflow 초기화 완료 대기 (약 30초)
docker compose logs -f airflow-init
```

## 서비스 포트

| 서비스           | 포트  | 기본 계정           |
|----------------|-------|-------------------|
| InfluxDB        | 8086  | admin / adminpassword |
| Grafana         | 3000  | admin / admin      |
| MLflow          | 5000  | -                 |
| Airflow (UI)    | 8080  | admin / admin      |
| PHM 추론 서버    | 8000  | -                 |

## C# 앱 설정

PHM 모션 스튜디오의 **서버 설정** 메뉴에서:

- **Airflow URL**: `http://<서버IP>:8080`
- **추론 서버 URL**: `http://<서버IP>:8000`
- **연속 수집 경로**: `\\<서버IP>\phm_data` (Windows 공유 폴더)

Windows 공유 폴더 설정 (서버 PC 관리자 권한):
```bat
net share phm_data=D:\...\server\phm_data /grant:everyone,full
```

## 업데이트

```bash
git pull
docker compose pull          # 이미지 업데이트 (선택)
docker compose up -d         # 변경된 서비스만 재시작
```
