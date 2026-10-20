#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地常驻 rerank 服务：Cohere 兼容 /rerank 接口 + bge-reranker-v2-m3。

用途：供 WeKnora 以「Custom (OpenAI-compatible)」provider 接入本地重排。
WeKnora 的 Generic provider 走 Cohere wire 协议（internal/models/api/cohererank/client.go），
因此本服务只需实现：
  POST /rerank   请求 {model, query, documents: [str], top_n?, return_documents?}
                 响应 {results: [{index, relevance_score, document?}]}
  GET  /health   存活探针

启动（模型加载一次，常驻内存约 1-2.5GB）：
  pip install fastapi "uvicorn[standard]" FlagEmbedding psutil
  python3 rerank_server.py                # 默认 0.0.0.0:8012
  PORT=8012 python3 rerank_server.py

WeKnora 接入要点：
  - WeKnora 若跑在 Docker 里，BaseURL 填 http://host.docker.internal:8012
    （而非 localhost，容器内 localhost 指向容器自身）；
  - bge 分数经 normalize 后是 0~1 概率，provider 的 score_scale 保持默认
    probability 即可，勿选 logit，否则租户阈值（默认 0.2）会失效。
"""

import logging
import os
import threading
import time

# 镜像开关必须放在 import transformers/FlagEmbedding 之前才生效
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("rerank_server")

MODEL_ID = "BAAI/bge-reranker-v2-m3"
PORT = int(os.environ.get("PORT", "8012"))

# FlagReranker 非线程安全，用锁串行化推理；本地单机场景吞吐足够
_infer_lock = threading.Lock()
_reranker = None


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """启动时加载一次模型，后续请求复用，避免每次冷启动"""
    global _reranker
    from FlagEmbedding import FlagReranker

    t0 = time.perf_counter()
    log.info("正在加载 %s（首次运行含权重下载，约 2.27GB）...", MODEL_ID)
    _reranker = FlagReranker(MODEL_ID, use_fp16=True)
    log.info("模型加载完成，耗时 %.1fs", time.perf_counter() - t0)
    yield


app = FastAPI(title="local-bge-reranker", version="1.1", lifespan=_lifespan)


class RerankRequest(BaseModel):
    """Cohere 兼容请求体；WeKnora cohererank client 的字段为其子集"""

    model: str = ""  # WeKnora 会回传配置的模型名，本地仅有一个模型，不做校验
    query: str
    documents: list[str] = Field(default_factory=list)
    top_n: int | None = None  # 只返回前 N 条；为空返回全部
    return_documents: bool | None = False  # true 时在结果里回显文档文本


class RerankResultItem(BaseModel):
    index: int
    relevance_score: float
    document: dict | None = None  # Cohere 格式回显为 {"text": ...}


class RerankResponse(BaseModel):
    results: list[RerankResultItem]
    model: str = MODEL_ID


class HealthResponse(BaseModel):
    status: str
    model: str
    ready: bool


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok", model=MODEL_ID, ready=_reranker is not None
    )


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    """对 query-documents 打相关性分并按分数降序返回

    输入示例：
      {"query": "什么是重排序", "documents": ["rerank 是精排模型", "今天天气好"], "top_n": 1}
    输出示例：
      {"results": [{"index": 0, "relevance_score": 0.88}], "model": "..."}
    """
    if _reranker is None:
        # 模型未就绪（理论上 startup 完成后不会出现），返回 503 让调用方走降级
        raise HTTPException(status_code=503, detail="model not ready")
    if not req.query or not req.documents:
        raise HTTPException(status_code=400, detail="query and documents are required")

    pairs = [[req.query, doc] for doc in req.documents]
    t0 = time.perf_counter()
    with _infer_lock:
        scores = _reranker.compute_score(pairs, normalize=True)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # compute_score 单条时可能返回 float / np.float32 / 0 维数组，
    # 批量时返回 1 维数组；atleast_1d 统一成 1 维再转 Python float 列表
    scores = np.atleast_1d(np.asarray(scores)).tolist()

    # 组装结果并按分数降序（Cohere 协议约定 results 有序）
    items = [
        RerankResultItem(
            index=i,
            relevance_score=float(s),
            document={"text": req.documents[i]} if req.return_documents else None,
        )
        for i, s in enumerate(scores)
    ]
    items.sort(key=lambda x: x.relevance_score, reverse=True)

    top_n = req.top_n if req.top_n and req.top_n > 0 else len(items)
    items = items[:top_n]

    log.info(
        "rerank query=%r docs=%d top_n=%s 耗时=%.0fms top_score=%.4f",
        req.query[:32], len(req.documents), req.top_n, elapsed_ms, items[0].relevance_score,
    )
    return RerankResponse(results=items, model=req.model or MODEL_ID)


if __name__ == "__main__":
    import uvicorn

    # 绑 0.0.0.0 是为了让 Docker 容器经 host.docker.internal 访问（绑 127.0.0.1 容器进不来）；
    # 服务本身无鉴权，仅限可信内网使用，公网环境需另加 token 或防火墙规则。
    # 本地单用户场景单 worker 即可；多 worker 会重复加载多份模型
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
