# syntax=docker/dockerfile:1.7
# Nometria control plane — gateway + API.
#
# All four permissive-licence classifier/embedding models (Granite Guardian, PIGuard,
# the protectai ensemble backstop, and the embedding-similarity detector — all
# Apache-2.0/MIT) are baked in and pre-fetched at build time below, plus the
# licence-gated Llama Guard 3-8B tier (Appendix A.4: non-OSI licence, acceptable-use
# policy, >700M-MAU clause — legal review required before commercial deployment,
# which is why it stays behind NOMETRIA_ACCEPT_RESTRICTED_MODEL_LICENSES=1 at runtime
# and behind an explicit build secret here, never a plain ENV, so the token can't leak
# into an image layer or `docker history`).
#
# This makes the image materially bigger (~1-2GB of model weights) and the build
# requires network egress + a Hugging Face token that has accepted Meta's Llama
# licence at huggingface.co — see deploy/README (or ask) for exactly how to build it.
# The running container itself still makes no outbound calls for any of this: weights
# are read from the image, never fetched at request time (NFR-4/NFR-9).
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/var/nometria/hf-cache

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libpq5 \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
# alembic.ini and migrations/ are force-included into the wheel (see
# pyproject.toml), so the build fails outright without them — "Forced include
# not found: /app/alembic.ini". They also have to be here on their own merit:
# this image's own CMD comment sends schema upgrades through `agentfox db
# upgrade`, and until the wheel carried the migration scripts that command
# could not run inside the container at all.
#
# Before `src`, deliberately: migrations change far less often, so the install
# layer stays cached across ordinary source edits.
COPY alembic.ini ./
COPY migrations ./migrations
COPY src ./src

# `postgres`/`otel` for the compose stack, `classifiers` (transformers + torch) for
# every model-based detector below.
RUN pip install --upgrade pip && pip install ".[postgres,otel,classifiers]"

# CI builds and stops here (`docker build --target deps`, no secret, no GPU-scale
# download) — enough to catch what actually broke production once already (finding
# 1.1, production-readiness-review.md §1.1: COPY paths pointing at directories that
# don't exist in the repo). A real smoke test, not just Dockerfile syntax: this stage
# fails the same way an operator's build would if `src ./src` or the extras install
# ever stopped matching what the app needs to import.
FROM base AS deps
RUN python -c "from nometria.gateway.app import create_app; create_app(); print('image smoke: OK')"

FROM deps AS weights
RUN mkdir -p /var/nometria/hf-cache

# Permissive tier — Apache-2.0/MIT, no licence gate, no secret needed.
#
# granite-guardian-3.0-2b declares architectures: ["GraniteForCausalLM"], so
# AutoModelForSequenceClassification raises "Unrecognized configuration class
# GraniteConfig for this kind of AutoModel" and takes the whole layer with it. The
# other three genuinely are sequence-classification or embedding models.
#
# This had never built. ci.yml's docker-smoke targets `deps`, one stage earlier, so
# nothing in CI ever reached this line — it surfaced the first time a workflow tried
# to publish the finished image.
RUN python -c "\
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer, AutoModel; \
AutoTokenizer.from_pretrained('ibm-granite/granite-guardian-3.0-2b'); \
AutoModelForCausalLM.from_pretrained('ibm-granite/granite-guardian-3.0-2b'); \
AutoTokenizer.from_pretrained('leolee99/PIGuard', trust_remote_code=True); \
AutoModelForSequenceClassification.from_pretrained('leolee99/PIGuard', trust_remote_code=True); \
AutoTokenizer.from_pretrained('protectai/deberta-v3-base-prompt-injection-v2'); \
AutoModelForSequenceClassification.from_pretrained('protectai/deberta-v3-base-prompt-injection-v2'); \
AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'); \
AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'); \
print('permissive-tier weights cached')"

# Restricted tier — meta-llama/Llama-Guard-3-8B is a *gated* HF repo: this only
# succeeds once the account behind the token has clicked "accept" on Meta's licence
# for this model at https://huggingface.co/meta-llama/Llama-Guard-3-8B. The secret is
# mounted only for this one step (least privilege) and is never written to a layer —
# BuildKit secret mounts don't persist in the final image or `docker history`.
# Build with: DOCKER_BUILDKIT=1 docker build --secret id=hf_token,env=HF_TOKEN ...
# (docker compose: see deploy/docker-compose.yml's `secrets:` block).
# Skipped, not failed, when no token is mounted. Two reasons. A build without a
# token is the normal case — anyone building this image who has not accepted Meta's
# licence, including the public image workflow, which must never bake gated weights
# into something we publish. And an empty secret does not fail cleanly: huggingface_hub
# sends `Authorization: Bearer ` and httpx rejects it as an illegal header value,
# so the build died on a confusing protocol error rather than a missing credential.
# `safety.restricted` then loads its weights at runtime, or stays unavailable.
RUN --mount=type=secret,id=hf_token \
    sh -eu -c 'HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null || true)"; \
    if [ -z "$HF_TOKEN" ]; then \
      echo "no hf_token mounted - skipping the licence-gated restricted tier"; \
      exit 0; \
    fi; \
    export HF_TOKEN; \
    python -c "import os; from huggingface_hub import login; login(token=os.environ[\"HF_TOKEN\"]); from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained(\"meta-llama/Llama-Guard-3-8B\"); AutoModelForCausalLM.from_pretrained(\"meta-llama/Llama-Guard-3-8B\"); print(\"restricted-tier weights cached\")"'

RUN useradd --create-home --uid 10001 nometria \
 && mkdir -p /var/nometria/evidence \
 && chown -R nometria:nometria /var/nometria /app
USER nometria

EXPOSE 8080

# Seed on first boot so the stack is demonstrable immediately, then serve. Seeding is
# idempotent, so a restart against an existing database is a no-op. Schema *upgrades*
# on an existing database go through `agentfox db upgrade` (Alembic), not this line —
# `create_all` cannot evolve an existing schema, and this command doesn't try to.
CMD ["sh", "-c", "agentfox seed || true; exec uvicorn nometria.gateway.app:app --host 0.0.0.0 --port 8080"]
