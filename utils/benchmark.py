"""
Latency benchmarking per operating point (M_1 ... M_N), and per-level
parameter counts. FLOPs counting is left to `fvcore`/`thop` if installed
(optional dependency) since exact FLOPs require tracing every branch
including the dynamic gating paths.
"""

import time
import torch


@torch.no_grad()
def benchmark_latency(model, input_size=(1, 3, 1024, 2048), device="cuda", warmup=10, iters=50):
    model.eval().to(device)
    x = torch.randn(*input_size, device=device)
    results = {}

    for level in range(1, model.num_levels + 1):
        for _ in range(warmup):
            model(x, max_level=level)
        if device == "cuda":
            torch.cuda.synchronize()

        start = time.time()
        for _ in range(iters):
            model(x, max_level=level)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start

        avg_ms = (elapsed / iters) * 1000
        results[f"level_{level}"] = {
            "avg_latency_ms": avg_ms,
            "fps": 1000.0 / avg_ms,
            "resolution": model.resolutions[level - 1],
        }
    return results


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable, "total_params_M": total / 1e6}
