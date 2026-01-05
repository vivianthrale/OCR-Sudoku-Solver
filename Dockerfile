FROM python:3.10-slim

# System deps (build tools; add more libs later only if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps from your existing requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Fly.io default internal port
ENV PORT=8080

# Run the Flask app via gunicorn (app.py exposes app = Flask(...))
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]
