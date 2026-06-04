import sys

import cv2
import numpy as np
import torch
import torchvision
import yaml
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def main() -> int:
    print(f"python: {sys.executable}")
    print(f"cv2: {cv2.__version__}")
    print(f"numpy: {np.__version__}")
    print(f"PyYAML: {yaml.__version__}")
    print(f"torch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"torch cuda runtime: {torch.version.cuda}")
    print(f"torch cuda available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("ERROR: PyTorch cannot see the NVIDIA GPU from this interpreter.")
        return 1

    device_name = torch.cuda.get_device_name(0)
    x = torch.ones((256, 256), device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print(f"gpu: {device_name}")
    print(f"cuda tensor check: {float(y[0, 0])}")

    cv_cuda_devices = 0
    if hasattr(cv2, "cuda"):
        cv_cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
    print(f"opencv cuda devices: {cv_cuda_devices}")
    if cv_cuda_devices == 0:
        print("note: opencv-python exposes cv2, but its pip wheels are not built with CUDA.")
        print("note: this project trains on CUDA through PyTorch.")

    MeanAveragePrecision(iou_thresholds=[0.5])
    print("torchmetrics mAP import: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
