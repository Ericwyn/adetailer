FROM ultralytics/ultralytics:8.3.96-cpu

WORKDIR /app

# The upstream/base build environment may carry host-local proxy values.
# Clear them so runtime image fetches do not try container-local 127.0.0.1 proxies.
ENV HTTP_PROXY="" \
    HTTPS_PROXY="" \
    http_proxy="" \
    https_proxy="" \
    NO_PROXY="" \
    no_proxy=""

# torch + ultralytics already included in base image, only install app deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

COPY . .

EXPOSE 9000
CMD ["python", "app.py"]
