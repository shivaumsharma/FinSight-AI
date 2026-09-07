#!/usr/bin/env bash
# digitalocean_droplet_setup.sh
#
# Deploys FinSight's FastAPI backend to a plain DigitalOcean droplet
# using the existing, platform-agnostic Dockerfile (already proven on
# Railway/Cloud Run -- see README.md's Deployment section) -- nothing
# DigitalOcean-specific needed in the image itself, just Docker on a
# VM instead of a managed container platform.
#
# Run this ON the droplet itself (SSH in first), as root or a sudo user,
# from an Ubuntu 22.04/24.04 droplet -- NOT from your local machine.
# DigitalOcean account creation, payment method, and droplet provisioning
# are account-level steps only the account owner can do (see README.md's
# "DigitalOcean droplet" deployment section for that walkthrough) -- this
# script starts from "I have a fresh droplet and SSH access to it."
#
# What this does:
#   1. Installs Docker if not already present.
#   2. Clones (or updates) the FinSight-AI repo.
#   3. Sets up .env from .env.example if one doesn't already exist --
#      you still need to fill in real values before the container will
#      actually work (see the printed reminder at the end).
#   4. Creates a persistent host directory for jobs.db/reports/llm_logs
#      and bind-mounts it into the container -- without this, all of
#      that is lost on every `docker run`/redeploy, the exact ephemeral-
#      filesystem problem EVALUATION.md documents for other free-tier
#      targets. A droplet's own disk persists across container
#      restarts (unlike Cloud Run/HF Spaces' ephemeral filesystem), so
#      this bind mount is enough -- no separate volume service needed.
#   5. Builds the image and runs the container with --restart unless-stopped
#      (survives a droplet reboot) on port 8000.
#
# Not scripted here, documented instead (needs a real domain + DNS
# propagation to test, can't be verified from this script alone): HTTPS
# via nginx + certbot once your domain points at this droplet's IP. See
# the printed next-steps at the end.

set -euo pipefail

REPO_URL="https://github.com/shivaumsharma/FinSight-AI.git"
APP_DIR="/opt/finsight-ai"
DATA_DIR="/opt/finsight-data"
CONTAINER_NAME="finsight-api"
IMAGE_NAME="finsight-api:latest"
PORT=8000

echo "=== 1/5: Docker ==="
if ! command -v docker &> /dev/null; then
    echo "Docker not found -- installing via the official convenience script."
    curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
    sh /tmp/get-docker.sh
    rm /tmp/get-docker.sh
else
    echo "Docker already installed: $(docker --version)"
fi

echo
echo "=== 2/5: Repo ==="
if [ -d "$APP_DIR/.git" ]; then
    echo "Repo already exists at $APP_DIR -- pulling latest main."
    git -C "$APP_DIR" pull origin main
else
    git clone "$REPO_URL" "$APP_DIR"
fi

echo
echo "=== 3/5: Environment file ==="
if [ ! -f "$APP_DIR/.env" ]; then
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    echo "Created $APP_DIR/.env from .env.example -- STILL NEEDS REAL VALUES."
    echo "(see the reminder printed at the end of this script)"
else
    echo "$APP_DIR/.env already exists -- left untouched, not overwritten."
fi

echo
echo "=== 4/5: Persistent data directory ==="
mkdir -p "$DATA_DIR"
echo "$DATA_DIR created/confirmed -- jobs.db/reports/llm_logs will live here,"
echo "bind-mounted into the container, so they survive redeploys and reboots."

echo
echo "=== 5/5: Build and run ==="
docker build -t "$IMAGE_NAME" "$APP_DIR"

# Stop/remove any previous run of this container so re-running this
# script is a safe, repeatable redeploy, not a failure on a name clash.
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

docker run -d \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    -p "${PORT}:${PORT}" \
    -e PORT="${PORT}" \
    -e DATA_DIR=/data \
    --env-file "$APP_DIR/.env" \
    -v "$DATA_DIR:/data" \
    "$IMAGE_NAME"

echo
echo "=================================================================="
echo "Container started. Check it's actually healthy:"
echo "  docker logs -f $CONTAINER_NAME"
echo "  curl http://localhost:${PORT}/health   (or whatever your health route is)"
echo
echo "STILL NEEDED before this is a real, working deployment:"
echo "  1. Edit $APP_DIR/.env with real values (FINNHUB_API_KEY, LLM_API_KEY"
echo "     if LLM_PROVIDER=hosted, API_KEY for the app's own auth, etc. --"
echo "     see .env.example's own comments for what each one does), then:"
echo "       docker restart $CONTAINER_NAME"
echo "  2. Open port ${PORT} in the droplet's firewall (DigitalOcean Cloud"
echo "     Firewall or ufw), or put nginx in front of it on 80/443."
echo "  3. To point finsightresearch.me (or wherever) at this droplet:"
echo "     add an A record for the domain -> this droplet's IP, then set up"
echo "     nginx + certbot for HTTPS -- not scripted here since it needs a"
echo "     real, propagated domain to actually test against."
echo "=================================================================="
