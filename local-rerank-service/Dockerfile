# bge-reranker-v2-m3 推理服务镜像：Cohere 兼容 /rerank，供 WeKnora 以 Generic(remote) provider 接入。
#
# 设计取舍：
# - 镜像只装运行时与依赖（约 1.5GB）；2.27GB 模型权重在首次启动时下载，缓存到 /hf 卷，
#   权重不进镜像，升级模型或换副本都不必重建镜像。
# - 显式安装 CPU 版 torch。默认源会拉 CUDA 轮子，镜像会从 ~1.5GB 涨到 7GB+，
#   而本服务是纯 CPU 推理（WeKnora 调 /rerank，无 GPU 需求）。
# - 不使用 `# syntax=docker/dockerfile:1` 前端镜像：本文件没有用到 heredoc 等新语法，
#   去掉后可避免在受限网络下额外拉取 docker.io 的前端镜像。
# - 基础镜像可通过 build-arg 换成镜像源（国内直连 Docker Hub 常超时）：
#   docker build --build-arg PYTHON_IMAGE=docker.m.daocloud.io/library/python:3.12-slim .
ARG PYTHON_IMAGE=python:3.12-slim-bookworm
FROM ${PYTHON_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/hf \
    HF_ENDPOINT=https://hf-mirror.com \
    PORT=8012 \
    OMP_NUM_THREADS=4 \
    TOKENIZERS_PARALLELISM=false

# CPU 版 torch 单独装：用 pytorch 的 cpu index，避免拉入 nvidia-* 依赖
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

# 1.4.x 在 CPU 设备上会自动把 use_fp16 置 False（inference/reranker/encoder_only/base.py），
# 因此 rerank_server.py 里写死的 use_fp16=True 在 CPU 上不会触发半精度算子问题；
# 锁 1.4.x 大版本，避免这个内部行为随版本漂移。
RUN pip install "FlagEmbedding>=1.4,<2" fastapi "uvicorn[standard]"

WORKDIR /app
COPY rerank_server.py ./

# 非 root 运行；/hf 预建并授权，named volume 首次创建时会继承该属主
RUN useradd --create-home --uid 10001 rerank \
    && mkdir -p /hf \
    && chown -R rerank:rerank /hf /app
USER rerank

EXPOSE 8012

# /health 在模型未加载完时返回 200 且 ready=false，这里按 ready 判定，
# 让编排层等待权重下载完成。首次下载 2.27GB，start-period 给足 10 分钟。
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD python -c "import json,os,urllib.request as u;d=json.load(u.urlopen('http://127.0.0.1:%s/health'%os.environ.get('PORT','8012'),timeout=4));raise SystemExit(0 if d.get('ready') else 1)" || exit 1

CMD ["python", "rerank_server.py"]
