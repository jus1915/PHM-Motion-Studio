@echo off
echo [PHM] WSL2 포트 포워딩 복구 중...
docker restart phm_inference phm_airflow_webserver
echo [PHM] 완료. 30초 후 서비스 사용 가능.
