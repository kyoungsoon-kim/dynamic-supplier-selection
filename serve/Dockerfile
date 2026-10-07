# 경량 파이썬 베이스 (slim = 불필요한 OS 패키지 뺀 버전)
FROM python:3.12-slim

WORKDIR /app

# torch는 GPU판이 기본이라 용량 큼 → CPU 전용 인덱스에서 설치해 이미지 축소
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# 나머지 의존성
RUN pip install --no-cache-dir fastapi==0.115.6 "uvicorn[standard]==0.34.0" pydantic==2.10.4

# 앱 코드 + 학습된 모델 가중치 (재학습판)
# (build context = repo 루트. 빌드 명령: docker build -f serve/Dockerfile -t dss-api .)
COPY serve/model.py serve/app.py ./
COPY serve/retrained/best_model_v3.pt ./best_model_v3.pt

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
