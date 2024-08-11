FROM python:3.11-slim

WORKDIR /app

RUN apt-get update
RUN pip install --upgrade pip setuptools

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

# Put models in a predictable location
ENV HF_HOME=/app/models

RUN python3 download_vit.py

WORKDIR /app

ENV APP_PORT=8080

ENV APP_TIMEOUT=5

ENV OMP_NUM_THREADS=1

ENTRYPOINT ["/bin/bash", "-c"]
CMD ["uvicorn app:app --host 0.0.0.0 --port 8080 --port $APP_PORT --timeout-keep-alive $APP_TIMEOUT"]