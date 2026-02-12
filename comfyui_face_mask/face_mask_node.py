import torch
import numpy as np


class FaceDetectMask:
    """
    ComfyUI node that detects faces in an image using InsightFace and generates
    per-face binary masks sorted left-to-right. Output shape: [N+1, H, W] where
    N = number of speakers and the last channel is the background mask.

    This is designed to feed into MultiTalkWav2VecEmbeds.ref_target_masks so that
    multi-person InfiniteTalk mode knows which face region gets which audio track.
    """

    _face_app = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "expand_ratio": (
                    "FLOAT",
                    {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1},
                ),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "detect_faces"
    CATEGORY = "face"

    @classmethod
    def _get_face_app(cls):
        """Lazy-load InsightFace model once and cache it."""
        if cls._face_app is None:
            from insightface.app import FaceAnalysis

            cls._face_app = FaceAnalysis(
                name="buffalo_sc",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            cls._face_app.prepare(ctx_id=0, det_size=(640, 640))
        return cls._face_app

    def detect_faces(self, image, expand_ratio=1.5):
        # image: [B, H, W, C] tensor, float32 in [0, 1], RGB
        img = image[0]  # Use first frame: [H, W, C]
        H, W, _C = img.shape

        # Convert to numpy BGR for InsightFace
        img_np = (img.cpu().numpy() * 255).astype(np.uint8)
        img_bgr = img_np[:, :, ::-1].copy()

        # Detect faces
        try:
            app = self._get_face_app()
            faces = app.get(img_bgr)
        except Exception as e:
            print(f"[FaceDetectMask] Detection failed: {e}, using fallback masks")
            faces = []

        num_detected = len(faces)

        if num_detected >= 2:
            # Sort left-to-right by bbox x-center
            faces_sorted = sorted(
                faces, key=lambda f: (f.bbox[0] + f.bbox[2]) / 2
            )

            masks = []
            combined = np.zeros((H, W), dtype=np.float32)

            for face in faces_sorted:
                x1, y1, x2, y2 = face.bbox
                # Expand bbox by expand_ratio
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                bw = (x2 - x1) * expand_ratio
                bh = (y2 - y1) * expand_ratio

                ex1 = max(0, int(cx - bw / 2))
                ey1 = max(0, int(cy - bh / 2))
                ex2 = min(W, int(cx + bw / 2))
                ey2 = min(H, int(cy + bh / 2))

                mask = np.zeros((H, W), dtype=np.float32)
                mask[ey1:ey2, ex1:ex2] = 1.0
                masks.append(mask)
                combined = np.clip(combined + mask, 0, 1)

            # Background = inverse of all face regions combined
            bg_mask = 1.0 - combined
            masks.append(bg_mask)

        elif num_detected == 1:
            # Single face in multi-mode: split image at midpoint as fallback.
            # The detected face anchors one side, the other half becomes the
            # second speaker region.
            face = faces[0]
            cx = (face.bbox[0] + face.bbox[2]) / 2

            mask1 = np.zeros((H, W), dtype=np.float32)
            mask2 = np.zeros((H, W), dtype=np.float32)

            mid = W // 2
            mask1[:, :mid] = 1.0
            mask2[:, mid:] = 1.0

            # Ensure mask1 is the side with the detected face (left-to-right order)
            if cx > mid:
                mask1, mask2 = mask2, mask1

            bg_mask = np.zeros((H, W), dtype=np.float32)
            masks = [mask1, mask2, bg_mask]

        else:
            # No faces detected: naive left/right split as last resort
            mask1 = np.zeros((H, W), dtype=np.float32)
            mask2 = np.zeros((H, W), dtype=np.float32)

            mid = W // 2
            mask1[:, :mid] = 1.0
            mask2[:, mid:] = 1.0

            bg_mask = np.zeros((H, W), dtype=np.float32)
            masks = [mask1, mask2, bg_mask]

        # Stack to [N+1, H, W] and convert to torch tensor
        mask_tensor = torch.from_numpy(np.stack(masks, axis=0))

        print(
            f"[FaceDetectMask] Detected {num_detected} face(s), "
            f"output mask shape: {mask_tensor.shape}"
        )

        return (mask_tensor,)
