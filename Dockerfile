FROM python:3.10-slim

WORKDIR /app

# Install CPU-only PyTorch first (much smaller, no CUDA), then other deps
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir Pillow>=10.0.0 flask python-dotenv requests

COPY . .

RUN mkdir -p app/static/uploads

EXPOSE 5000

CMD ["python", "run.py"]
