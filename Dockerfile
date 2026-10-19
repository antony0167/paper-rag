FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# models are downloaded on first use; mount a volume on /root/.cache to keep them
ENV RAG_DATA_DIR=/app/data
EXPOSE 8000
CMD ["paper-rag", "serve", "--host", "0.0.0.0", "--port", "8000"]
