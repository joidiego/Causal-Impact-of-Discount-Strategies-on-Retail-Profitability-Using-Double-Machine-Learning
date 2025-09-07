# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/
COPY notebooks/ notebooks/
COPY run_pipeline.py .

CMD ["python", "run_pipeline.py"]