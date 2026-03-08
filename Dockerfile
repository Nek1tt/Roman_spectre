FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./

COPY outputs/ ./outputs/

EXPOSE 8080

ENV PYTHONUNBUFFERED=1

CMD ["python", "app_gradio.py"]