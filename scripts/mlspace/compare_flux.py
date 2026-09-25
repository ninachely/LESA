"""Paired full-FLUX / LESA smoke comparison with warmed, synchronized timings."""
import argparse
import json
from pathlib import Path
import time


def image_metrics(reference, candidate):
    import numpy as np
    from skimage.metrics import structural_similarity
    a, b = np.asarray(reference, dtype=np.float32), np.asarray(candidate, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError(f"Image shape mismatch: {a.shape} versus {b.shape}")
    mse = float(np.mean((a - b) ** 2))
    return {
        "psnr_db": float(10 * np.log10(255.0**2 / mse)) if mse else "inf",
        "ssim_rgb": float(structural_similarity(a, b, data_range=255, channel_axis=-1)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir

    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from flux.sampling import get_noise, get_schedule, prepare, unpack, denoise_cache
    from flux.util import load_t5, load_clip, load_flow_model, load_ae
    from predictor import Predictor, Config, cache_init, cal_type

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    config = json.loads((run / "config.json").read_text())
    predictor = Predictor(Config(), num_steps=50, enable_training=False)
    predictor.load_model(str(run / "weights/predictor_flux_N7_E3.pt"))
    if not all(torch.isfinite(p).all() for p in predictor.model.parameters()):
        raise RuntimeError("Predictor contains non-finite parameters")
    t5, clip = load_t5(device, max_length=512), load_clip(device)
    model = load_flow_model("flux-dev", device).eval().requires_grad_(False)
    ae = load_ae("flux-dev", device).eval().requires_grad_(False)

    def kwargs(interval):
        return dict(num_steps=50, height=1024, width=1024, test_FLOPs=False,
                    monitor_gpu_usage=False, data_prepare=False, data_dir=None,
                    phase=None, idx=0, interval=interval, first_enhance=3)

    def full_steps(interval):
        cache, state = cache_init(**kwargs(interval))
        steps = []
        for i in range(50):
            state["step"] = i
            cal_type(cache, state)
            if state["type"] == "full":
                steps.append(i)
        return steps

    @torch.no_grad()
    def generate(inp, timesteps, interval, decode=True):
        cache, state = cache_init(**kwargs(interval))
        # Isolate modes even if denoising later acquires in-place updates.
        inputs = {k: v.clone() for k, v in inp.items()}
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        latent = denoise_cache(model, **inputs, timesteps=timesteps, guidance=3.5,
                               predictor=predictor, cache_dic=cache, current=state)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() / 1024**3
        if not torch.isfinite(latent).all():
            raise RuntimeError("Non-finite denoising output")
        if not decode:
            return None, elapsed, peak
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pixels = ae.decode(unpack(latent.float(), 1024, 1024))
        if not torch.isfinite(pixels).all():
            raise RuntimeError("Non-finite decoded image")
        pixels = (127.5 * (pixels.clamp(-1, 1)[0].permute(1, 2, 0).float() + 1)).byte().cpu().numpy()
        return Image.fromarray(pixels), elapsed, peak

    rows = []
    for index, prompt in enumerate(config["test_prompts"]):
        seed = config["test_seed"] + index
        noise = get_noise(1, 1024, 1024, device, torch.bfloat16, seed)
        with torch.no_grad():
            inp = prepare(t5, clip, noise, prompt)
        timesteps = get_schedule(50, inp["img"].shape[1], shift=True)
        if index == 0:
            print("Warming up full FLUX and LESA (excluded from timings)", flush=True)
            generate(inp, timesteps, 1, decode=False)
            generate(inp, timesteps, 7, decode=False)
        outputs = {}
        modes = [("full", 1), ("lesa", 7)]
        for name, interval in modes if index % 2 == 0 else reversed(modes):
            print(f"COMPARE prompt={index} mode={name}", flush=True)
            outputs[name] = generate(inp, timesteps, interval)
            (run / name).mkdir(exist_ok=True)
            outputs[name][0].save(run / name / f"img_{index}.png")
        full, full_time, full_peak = outputs["full"]
        lesa, lesa_time, lesa_peak = outputs["lesa"]
        panel = Image.new("RGB", (1024, 550), "white")
        panel.paste(full.resize((512, 512)), (0, 38))
        panel.paste(lesa.resize((512, 512)), (512, 38))
        draw = ImageDraw.Draw(panel)
        draw.text((12, 12), "Full FLUX / 50 steps", fill="black")
        draw.text((524, 12), "LESA / N7 E3 / 50 steps", fill="black")
        panel.save(run / f"comparison_{index}.png")
        row = dict(prompt=prompt, seed=seed, full_denoise_seconds=full_time,
                   lesa_denoise_seconds=lesa_time, denoise_speedup=full_time / lesa_time,
                   full_peak_allocated_gib=full_peak, lesa_peak_allocated_gib=lesa_peak,
                   **image_metrics(full, lesa))
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary = dict(
        purpose="Smoke test on tiny training/test sets; not paper reproduction",
        timing="Denoising only; one warmup per mode; CUDA synchronized; excludes text encoding, decoding, saving and model loading",
        images="Lossless RGB PNG; no watermark; RGB SSIM and float32 PSNR",
        full_step_indices=full_steps(1), lesa_full_step_indices=full_steps(7),
        denoise_speedup=sum(r["full_denoise_seconds"] for r in rows) / sum(r["lesa_denoise_seconds"] for r in rows),
        mean_ssim_rgb=float(np.mean([r["ssim_rgb"] for r in rows])), samples=rows,
    )
    (run / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
