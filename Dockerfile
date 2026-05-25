FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# TensorFlow / Pillow often need these libs on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

# Switch between CLI and web using APP_MODE
ENV APP_MODE=web

CMD ["sh", "-c", "if [ \"$APP_MODE\" = \"cli\" ] && [ -f cli_app.py ]; then python cli_app.py; else gunicorn --bind 0.0.0.0:5000 app:app; fi"]