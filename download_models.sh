#!/bin/bash
# Download model weights at runtime if they don't exist.
# This keeps the Docker image small and caches models on a RunPod network volume.

set -e

download_if_missing() {
    local url="$1"
    local dest="$2"
    if [ ! -f "$dest" ]; then
        echo "Downloading $(basename "$dest")..."
        mkdir -p "$(dirname "$dest")"
        wget -q "$url" -O "$dest"
        echo "Done: $(basename "$dest")"
    else
        echo "Already exists: $(basename "$dest")"
    fi
}

echo "=== Checking model weights ==="

# Diffusion models
download_if_missing \
    "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors" \
    "/ComfyUI/models/diffusion_models/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors"

download_if_missing \
    "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors" \
    "/ComfyUI/models/diffusion_models/Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors"

download_if_missing \
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors" \
    "/ComfyUI/models/diffusion_models/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors"

download_if_missing \
    "https://huggingface.co/Kijai/MelBandRoFormer_comfy/resolve/main/MelBandRoformer_fp16.safetensors" \
    "/ComfyUI/models/diffusion_models/MelBandRoformer_fp16.safetensors"

# LoRA
download_if_missing \
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors" \
    "/ComfyUI/models/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"

# VAE
download_if_missing \
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors" \
    "/ComfyUI/models/vae/Wan2_1_VAE_bf16.safetensors"

# Text encoder
download_if_missing \
    "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-fp8_e4m3fn.safetensors" \
    "/ComfyUI/models/text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors"

# CLIP vision
download_if_missing \
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors" \
    "/ComfyUI/models/clip_vision/clip_vision_h.safetensors"

# Wav2Vec2 safetensors model (used by Wav2VecModelLoader node)
download_if_missing \
    "https://huggingface.co/Kijai/wav2vec2_safetensors/resolve/main/wav2vec2-chinese-base_fp16.safetensors" \
    "/ComfyUI/models/diffusion_models/wav2vec2-chinese-base_fp16.safetensors"

# Wav2Vec2 HuggingFace model for InfiniteTalk audio feature extraction
WAV2VEC_DIR="/ComfyUI/models/transformers/TencentGameMate/chinese-wav2vec2-base"
if [ ! -f "$WAV2VEC_DIR/pytorch_model.bin" ]; then
    echo "Downloading TencentGameMate/chinese-wav2vec2-base..."
    mkdir -p "$WAV2VEC_DIR"
    python -c "from huggingface_hub import snapshot_download; snapshot_download('TencentGameMate/chinese-wav2vec2-base', local_dir='$WAV2VEC_DIR')"
    echo "Done: chinese-wav2vec2-base"
else
    echo "Already exists: chinese-wav2vec2-base"
fi

echo "=== All models ready ==="
