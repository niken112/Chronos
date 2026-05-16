# Gunakan image Python yang ringan
FROM python:3.10-slim

# Set folder kerja di dalam container
WORKDIR /app

# Copy semua file kita ke dalam container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Jalankan FastAPI menggunakan Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
