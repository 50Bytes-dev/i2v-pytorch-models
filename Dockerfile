FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    && pip install --no-cache-dir --upgrade pip setuptools \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ARG HF_TOKEN

ENV HF_HOME=/app/models \
    APP_PORT=8080 \
    APP_TIMEOUT=5 \
    OMP_NUM_THREADS=1 \
    HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

COPY download_vit.py ./

RUN python download_vit.py && unset HF_TOKEN && unset HUGGING_FACE_HUB_TOKEN

COPY app.py vectorizer.py image2vec_vit.py ./

ENTRYPOINT [ "bash", "-c" ]
CMD [ "uvicorn app:app --host 0.0.0.0 --port $APP_PORT --timeout-keep-alive $APP_TIMEOUT" ]