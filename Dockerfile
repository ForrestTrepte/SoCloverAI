FROM python:3.12-slim

# Workdir where Dev Containers will mount this git repo in the container
# Note that, after the container is built, WORKDIR will be overwritten with the repository content
WORKDIR /workspaces/SoCloverAI

RUN apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*

# Install Claude Code
RUN curl -fsSL https://claude.ai/install.sh | bash
ENV PATH="/root/.local/bin:${PATH}"
# Pre-seed onboarding state so the "let's get started" flow is skipped
RUN echo '{"hasCompletedOnboarding":true,"lastOnboardingVersion":"1.0.0","projects":{"/workspaces/WordSpace":{"hasTrustDialogAccepted":true}}}' > /root/.claude.json

# Install uv and package dependencies
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_PROJECT_ENVIRONMENT=/opt/venvs/SoCloverAI
COPY pyproject.toml uv.lock ./
RUN uv sync

CMD ["bash"]
