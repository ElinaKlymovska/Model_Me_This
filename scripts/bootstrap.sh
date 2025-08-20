#!/usr/bin/env bash
set -euo pipefail
if [ -f .env ]; then set -a; source .env; set +a; fi

A1111_DIR="/workspace/stable-diffusion-webui"
PRESET_DIR="/workspace/portrait-enhancer"
PORT="${PORT:-7860}"

echo "[*] Installing base tools"
apt-get update -y && apt-get install -y git curl tmux unzip python3 python3-venv python3-pip wget
ln -sf $(command -v python3) /usr/bin/python || true

echo "[*] Cloning A1111"
if [ ! -d "$A1111_DIR" ]; then
  git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui "$A1111_DIR"
fi

echo "[*] Installing extensions"
mkdir -p "$A1111_DIR/extensions"
[ -d "$A1111_DIR/extensions/sd-webui-controlnet" ] || git clone https://github.com/Mikubill/sd-webui-controlnet "$A1111_DIR/extensions/sd-webui-controlnet"
[ -d "$A1111_DIR/extensions/adetailer" ] || git clone https://github.com/Bing-su/adetailer "$A1111_DIR/extensions/adetailer"

echo "[*] Creating model directories"
mkdir -p "$A1111_DIR/models/Stable-diffusion" "$A1111_DIR/models/ControlNet"

echo "[*] Auto-downloading models"
python /workspace/models_auto.py || true

# Auto-detect model names
if [ -z "${MODEL_CKPT_NAME:-}" ]; then
  MODEL_CKPT_NAME=$(ls -t "$A1111_DIR/models/Stable-diffusion"/*.safetensors 2>/dev/null | head -n1 | xargs -r -n1 basename || true)
fi
if [ -z "${CONTROLNET_FILE:-}" ]; then
  CONTROLNET_FILE=$(ls "$A1111_DIR/models/ControlNet"/*softedge*sd*xl*.safetensors 2>/dev/null | head -n1 || ls -t "$A1111_DIR/models/ControlNet"/*.safetensors 2>/dev/null | head -n1 || echo "")
  CONTROLNET_FILE=$(basename "$CONTROLNET_FILE" 2>/dev/null || echo "")
fi
if [ -z "${CONTROLNET2_FILE:-}" ]; then
  CONTROLNET2_FILE=$(ls "$A1111_DIR/models/ControlNet"/*canny*sd*xl*.safetensors 2>/dev/null | head -n1 || ls "$A1111_DIR/models/ControlNet"/*lineart*sd*xl*.safetensors 2>/dev/null | head -n1 || echo "")
  CONTROLNET2_FILE=$(basename "$CONTROLNET2_FILE" 2>/dev/null || echo "")
fi

echo "[*] Starting A1111"
if tmux has-session -t webui 2>/dev/null; then
  echo " - already running"
else
  tmux new -d -s webui "cd '$A1111_DIR' && python launch.py --api --listen --port $PORT --xformers --nowebui"
fi
until curl -s "http://127.0.0.1:$PORT/sdapi/v1/sd-models" >/dev/null; do sleep 2; done
echo " - API ready"

echo "[*] Setting up portrait enhancer"
cd "$PRESET_DIR"
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

echo "[*] Setting up ADetailer 2CN Plus"
cd "$PRESET_DIR/.."
if [ -d "adetailer_2cn_plus" ]; then
    cd adetailer_2cn_plus
    source ../portrait-enhancer/.venv/bin/activate
    pip install -r requirements.txt
    echo "✓ ADetailer 2CN Plus dependencies installed"
else
    echo "⚠ ADetailer 2CN Plus directory not found"
fi
cd "$PRESET_DIR"

echo "[*] Updating configuration"
sed -i 's/backend:.*/backend: "a1111"/' config.yaml
sed -i 's|a1111_endpoint:.*|a1111_endpoint: "http://127.0.0.1:'"$PORT"'"|' config.yaml
if [ -n "${MODEL_CKPT_NAME:-}" ]; then
  sed -i 's|model_checkpoint:.*|model_checkpoint: "'"$MODEL_CKPT_NAME"'"|' config.yaml
fi
if [ -n "${CONTROLNET_FILE:-}" ]; then
  sed -i 's|controlnet_model:.*|controlnet_model: "'"$CONTROLNET_FILE"'"|' config.yaml
  sed -i 's/use_controlnet: .*$/use_controlnet: true/' config.yaml
fi
if [ -n "${CONTROLNET2_FILE:-}" ]; then
  sed -i 's|controlnet2_model:.*|controlnet2_model: "'"$CONTROLNET2_FILE"'"|' config.yaml
  sed -i 's/use_controlnet2: .*$/use_controlnet2: true/' config.yaml
fi
sed -i 's/use_adetailer: .*$/use_adetailer: true/' config.yaml

echo "[*] Running portrait enhancement"
python batch.py

echo "[*] Complete -> portrait-enhancer/output"
echo "[*] Vast.ai instance ready!"
