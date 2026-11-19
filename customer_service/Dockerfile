FROM python:3.11-slim

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST} \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY __init__.py run_service.py /app/customer_service/
COPY core /app/customer_service/core
COPY retrieval /app/customer_service/retrieval
COPY policy /app/customer_service/policy
COPY generation /app/customer_service/generation
COPY agent /app/customer_service/agent
COPY app /app/customer_service/app
COPY data /app/customer_service/data

RUN useradd -r -u 10001 appuser \
    && mkdir -p /data \
    && chown -R appuser:appuser /app /data

USER appuser
WORKDIR /app/customer_service

EXPOSE 8000

CMD ["python", "run_service.py", "--host", "0.0.0.0", "--port", "8000"]
