FROM python:3.14-slim-bookworm

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY predict.py .

ENTRYPOINT ["python", "-u", "/app/predict.py"]
