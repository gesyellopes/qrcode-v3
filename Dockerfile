FROM python:3.12-slim

# pyzbar depende da lib nativa libzbar0
# opencv-headless depende de libgl1 e libglib2
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libzbar0 \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]