FROM python:3.11-slim

WORKDIR /app

RUN apt-get update
RUN pip install --upgrade pip setuptools

COPY requirements.txt .
RUN pip3 install -r requirements.txt

ARG MODEL_NAME
COPY download_model.py .
RUN ./download_model.py

COPY . .

# Put models in a predictable location
ENV HF_HOME=/app/models

RUN python3 download_vit.py

WORKDIR /app

ENV APP_PORT=8080

ENV APP_TIMEOUT=5

ENTRYPOINT ["/bin/sh", "-c"]
CMD ["uvicorn app:app --host 0.0.0.0 --port 8080 --port $APP_PORT --timeout-keep-alive $APP_TIMEOUT"]
