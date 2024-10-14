FROM ghcr.io/osai-ai/dokai:24.06-gpu.pytorch
COPY --from=ghcr.io/astral-sh/uv:0.4.20 /uv /bin/uv

# Setting working directory to the root user default home directory
WORKDIR /home/deepsport

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-install-workspace > /tmp/uv_sync.log 2>&1
