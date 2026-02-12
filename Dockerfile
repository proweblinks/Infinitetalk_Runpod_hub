# Use specific version of nvidia cuda image
FROM wlsdml1114/engui_genai-base_blackwell:1.1 as runtime

# wget 설치 (URL 다운로드를 위해)
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN pip install -U "huggingface_hub[hf_transfer]"
RUN pip install runpod websocket-client librosa
RUN pip install insightface onnxruntime-gpu

WORKDIR /

RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd /ComfyUI && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Comfy-Org/ComfyUI-Manager.git && \
    cd ComfyUI-Manager && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/city96/ComfyUI-GGUF && \
    cd ComfyUI-GGUF && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-KJNodes && \
    cd ComfyUI-KJNodes && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite && \
    cd ComfyUI-VideoHelperSuite && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/orssorbit/ComfyUI-wanBlockswap

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-MelBandRoFormer && \
    cd ComfyUI-MelBandRoFormer && \
    pip install -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper && \
    cd ComfyUI-WanVideoWrapper && \
    pip install -r requirements.txt

# Create model directories
RUN mkdir -p /ComfyUI/models/diffusion_models \
             /ComfyUI/models/loras \
             /ComfyUI/models/vae \
             /ComfyUI/models/text_encoders \
             /ComfyUI/models/clip_vision

# Download ALL model weights at build time for instant cold starts.
# Each download is a separate layer for build resilience.
# Total: ~25GB of model weights.

# Diffusion model: InfiniteTalk Single (~5GB fp8)
RUN wget -q "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors" \
    -O /ComfyUI/models/diffusion_models/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors

# Diffusion model: InfiniteTalk Multi (~5GB fp8)
RUN wget -q "https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors" \
    -O /ComfyUI/models/diffusion_models/Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors

# Diffusion model: Wan2.1 I2V 14B (~8GB fp8)
RUN wget -q "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors" \
    -O /ComfyUI/models/diffusion_models/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors

# Diffusion model: MelBandRoFormer (~100MB fp16)
RUN wget -q "https://huggingface.co/Kijai/MelBandRoFormer_comfy/resolve/main/MelBandRoformer_fp16.safetensors" \
    -O /ComfyUI/models/diffusion_models/MelBandRoformer_fp16.safetensors

# LoRA: LightX2V step distill (~250MB)
RUN wget -q "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors" \
    -O /ComfyUI/models/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors

# VAE: Wan2.1 VAE (~300MB bf16)
RUN wget -q "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors" \
    -O /ComfyUI/models/vae/Wan2_1_VAE_bf16.safetensors

# Text encoder: UMT5-XXL (~5GB fp8)
RUN wget -q "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-fp8_e4m3fn.safetensors" \
    -O /ComfyUI/models/text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors

# CLIP vision: clip_vision_h (~1.7GB)
RUN wget -q "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors" \
    -O /ComfyUI/models/clip_vision/clip_vision_h.safetensors

# Download InsightFace buffalo_sc model (~30MB) for face detection in multi-mode
RUN python -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider']); app.prepare(ctx_id=0)"

COPY . .

# Copy custom face detection node into ComfyUI custom_nodes
RUN cp -r /comfyui_face_mask /ComfyUI/custom_nodes/comfyui_face_mask

RUN chmod +x /entrypoint.sh /download_models.sh

CMD ["/entrypoint.sh"]
