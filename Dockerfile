# syntax=docker/dockerfile:1.7

ARG GO_VERSION=1.22
ARG NODE_VERSION=20

# Stage 1: 构建前端 WebUI 产物
FROM --platform=$BUILDPLATFORM node:${NODE_VERSION}-bookworm-slim AS frontend-builder
WORKDIR /src/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: 构建 Go 语言后端二进制
FROM --platform=$BUILDPLATFORM golang:${GO_VERSION}-bookworm AS backend-builder
WORKDIR /src
COPY backend/go.mod backend/go.sum ./backend/
RUN cd backend && go mod download
COPY backend/ ./backend/
ARG TARGETOS
ARG TARGETARCH
RUN cd backend && CGO_ENABLED=0 GOOS=${TARGETOS:-linux} GOARCH=${TARGETARCH:-amd64} \
    go build -trimpath -ldflags="-s -w" -o /out/qwen2api .

# Stage 3: 最终运行镜像 (集成 Python3 + DrissionPage + Playwright 环境)
FROM debian:bookworm-slim
WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    ENGINE_MODE=browser \
    BROWSER_POOL_SIZE=1 \
    LOG_LEVEL=INFO \
    DATA_DIR=/app/data \
    LOGS_DIR=/app/logs \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# 1. 安装系统底层动态依赖库、Python3 与中文字体
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    wget \
    python3 \
    python3-pip \
    python3-venv \
    unzip \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libcups2 \
    libdbus-1-3 \
    libdrm2 \
    libgbm1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libpangocairo-1.0-0 \
    libpulse0 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxkbcommon0 \
    libxrandr2 \
    libxshmfence1 \
    fonts-liberation \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# 2. 安装 DrissionPage 及依赖环境
RUN pip3 install --no-cache-dir --break-system-packages DrissionPage

# 3. 复制编译好的二进制文件与代码
COPY --from=backend-builder /out/qwen2api /usr/local/bin/qwen2api
COPY --from=frontend-builder /src/frontend/dist ./frontend/dist
COPY backend/ ./backend/

# 4. 离线预装 Playwright 浏览器二进制，创建工作目录
RUN mkdir -p /app/data /app/logs /ms-playwright \
    && /usr/local/bin/qwen2api --install-browsers

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-7860}/healthz" || exit 1

CMD ["/usr/local/bin/qwen2api"]
