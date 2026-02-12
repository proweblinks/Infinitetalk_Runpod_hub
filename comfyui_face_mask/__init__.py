from .face_mask_node import FaceDetectMask

NODE_CLASS_MAPPINGS = {
    "FaceDetectMask": FaceDetectMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceDetectMask": "Face Detect Mask",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
