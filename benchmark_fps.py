"""
FPS benchmark — §5 of the ablation protocol (abiliation/study_specifications.md).

Batch-1 streaming inference: warmup, then timed forward passes with
torch.cuda.synchronize() around the timed region, in FP32 and FP16.
Inference and NMS-postprocess are reported separately; preprocess is excluded
(VnPollenBee ships still images, so resize cost depends on source resolution,
not the model). This is per-frame detection throughput, not end-to-end video.

Usage:
    python benchmark_fps.py --attn cbam_botnet --ckpt runs/<run>/best.pt
"""
import argparse
import statistics
import time

import torch

from yolov8_model import MyYolo
from phase1_pipeline_v2 import non_max_suppression


def timed(fn, iters, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.mean(times), statistics.stdev(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', default='m', help='YOLOv8 scale (n|s|m|l|x)')
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--attn', default='none', choices=['none', 'cbam', 'botnet', 'cbam_botnet'])
    ap.add_argument('--ckpt', default='', help='optional best.pt to load')
    ap.add_argument('--iters', type=int, default=100)
    args = ap.parse_args()

    assert torch.cuda.is_available(), 'FPS benchmark requires CUDA'
    device = 'cuda'
    print(f'GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | cuda {torch.version.cuda}')
    print(f'yolov8{args.version} attn={args.attn} imgsz={args.imgsz} batch=1 iters={args.iters}')

    model = MyYolo(version=args.version, num_classes=2, attn=args.attn, img_size=args.imgsz)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, map_location=device)['model'])
        print(f'loaded {args.ckpt}')
    model.eval().to(device)

    for precision in ('fp32', 'fp16'):
        half = precision == 'fp16'
        m = model.half() if half else model.float()
        x = torch.rand(1, 3, args.imgsz, args.imgsz, device=device,
                       dtype=torch.float16 if half else torch.float32)
        with torch.no_grad():
            inf_ms, inf_sd = timed(lambda: m(x), args.iters)
            raw = m(x)
            nms_ms, nms_sd = timed(
                lambda: non_max_suppression([raw[0].float()], 0.25, 0.70), args.iters)
        total = inf_ms + nms_ms
        print(f'{precision}: inference {inf_ms:.2f}±{inf_sd:.2f} ms | '
              f'NMS {nms_ms:.2f}±{nms_sd:.2f} ms | total {total:.2f} ms | {1000 / total:.1f} FPS')


if __name__ == '__main__':
    main()
