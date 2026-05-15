FROM ultralytics/ultralytics:8.3.96-cpu

WORKDIR /app

# The upstream/base build environment may carry host-local proxy values.
# Clear them so runtime image fetches do not try container-local 127.0.0.1 proxies.
ENV HTTP_PROXY="" \
    HTTPS_PROXY="" \
    http_proxy="" \
    https_proxy="" \
    NO_PROXY="" \
    no_proxy="" \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# torch + ultralytics already included in base image, only install app deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

COPY . .

EXPOSE 9000
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-9000} --workers 1 --threads 2 --timeout 120 app:app"]
