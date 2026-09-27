# Serving image for the HDFC FAQ assistant: CPU-only, runs on a Hugging Face Space or any Docker host.
# Model files are not baked in: at startup inference.py downloads the adapter and FAQ index from the
# private HF repo (HF_REPO) using the HF_TOKEN secret, and checks the adapter's SHA-256.
FROM python:3.14-slim

# Hugging Face Spaces run the container as user 1000; give it a home it can write the model cache to
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    MODEL_VERSION=llama_v2 \
    HF_REPO=shyam003/hdfc-faq-assistant
WORKDIR /home/user/app

# Dependencies first, so code changes don't reinstall them
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY --chown=user . .

# 7860 is the port Hugging Face Spaces expects
EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
