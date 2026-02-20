# Build: docker build -t delivery-date-api .
# Run:   docker run -p 8000:8000 delivery-date-api
# (Ensure experiments/ contains a .pt checkpoint, or mount it: -v $(pwd)/experiments:/app/experiments)

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app and checkpoints. If you have no .pt yet, build then run with: -v $(pwd)/experiments:/app/experiments
COPY src/ ./src/
COPY experiments/ ./experiments/

EXPOSE 8000

# Serve the FastAPI app
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
