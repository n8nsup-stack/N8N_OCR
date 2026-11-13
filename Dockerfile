# 1. Use modern, supported Debian 12 Bookworm base image (solves 404 apt errors)
FROM python:3.10-slim-bookworm

# 2. Set working directory
WORKDIR /app

# 3. Install modern system dependencies
# libgomp1 is critical for Paddle's multi-threading; libgl1 and libglib2.0-0 for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Create a non-root user for Hugging Face security standards
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# 5. [CRITICAL] Set Compatibility Flags
# These bypass the 'set_optimization_level' and 'PIR' attribute errors in Paddle 2.6
ENV FLAGS_enable_pir_api=0 \
    FLAGS_enable_new_executor=0 \
    FLAGS_use_onednn=0 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# 6. Copy and install Python dependencies
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 7. Copy the application code
COPY --chown=user . .

# 8. Hugging Face Spaces port
EXPOSE 7860

# 9. Run the app on port 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
