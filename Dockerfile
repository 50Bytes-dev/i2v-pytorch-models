FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    && pip install --no-cache-dir --upgrade pip setuptools \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py vectorizer.py image2vec_vit.py ./

ENV HF_HOME=/app/models \
    APP_PORT=8080 \
    APP_TIMEOUT=5 \
    OMP_NUM_THREADS=1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "$APP_PORT", "--timeout-keep-alive", "$APP_TIMEOUT"]