#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVIDENCE_DIR="$DIR/evidence"
mkdir -p "$EVIDENCE_DIR/screenshots" "$EVIDENCE_DIR/benchmarks"

echo "=================================================="
echo "    HackwithBangalore 4.0 - Track A (PipeGen)     "
echo "           System & Environment Evidence          "
echo "=================================================="

ENV_FILE="$EVIDENCE_DIR/environment.txt"
rm -f "$ENV_FILE"

log() {
    echo -e "$1" | tee -a "$ENV_FILE"
}

log "=== 1. TIMESTAMP ==="
log "Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
log "Local:     $(date)"

log "\n=== 2. OPERATING SYSTEM & KERNEL ==="
log "$(uname -a)"
if [ -f /etc/os-release ]; then
    log "$(cat /etc/os-release | grep -E "PRETTY_NAME|VERSION_ID")"
fi

log "\n=== 3. HARDWARE SPECIFICATIONS ==="
log "CPU: $(lscpu | grep "Model name:" | sed 's/Model name:[ \t]*//')"
log "Logical Cores: $(nproc)"
log "System Memory:"
log "$(free -h)"

log "\n=== 4. NVIDIA GPU SPECIFICATIONS ==="
if command -v nvidia-smi &> /dev/null; then
    log "$(nvidia-smi)"
    nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit,compute_cap --format=csv > "$EVIDENCE_DIR/gpu_specs.csv"
else
    log "nvidia-smi not found in PATH"
fi

log "\n=== 5. DOCKER & CONTAINER ENVIRONMENT ==="
if command -v docker &> /dev/null; then
    log "$(docker --version)"
    log "Docker Daemon Status: $(service docker status 2>&1 | head -n 3)"
else
    log "docker not found"
fi

log "\n=== 6. NVIDIA CONTAINER TOOLKIT ==="
if command -v nvidia-ctk &> /dev/null; then
    log "$(nvidia-ctk --version)"
else
    log "nvidia-ctk not found"
fi

log "\n=== 7. PIPEGEN INSTALLATION ==="
if command -v pipegen &> /dev/null; then
    log "$(pipegen --version)"
    log "$(dpkg -s pipegen | grep -E "Package|Version|Status|Architecture")"
else
    log "pipegen not found"
fi

log "\n=== 8. DISPLAY & X11 STATUS ==="
log "DISPLAY=$DISPLAY"
log "X11 Sockets: $(ls -la /tmp/.X11-unix 2>&1)"

log "\n=== 9. DISK UTILIZATION ==="
log "$(df -h / /var/lib/docker)"

echo "--------------------------------------------------"
echo "Evidence recorded successfully at: $ENV_FILE"
echo "--------------------------------------------------"
