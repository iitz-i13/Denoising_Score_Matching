FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
	python3-pip \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

RUN mkdir -p output

CMD ["python", "scripts/main.py"]
