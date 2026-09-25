"""Single-GPU end-to-end smoke test; not a reproduction of paper quality."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
FLUX = REPO / "LESA_FLUX.1-dev_FLUX.1-schnell"


def select_prompts(train_text, test_text, count=8):
    train = [p.strip() for p in train_text.splitlines() if p.strip()][:count]
    test = [p.strip() for p in test_text.splitlines() if p.strip() and p.strip() not in set(train)][:2]
    if len(train) != count or len(test) != 2:
        raise ValueError("Not enough distinct train/test prompts")
    return train, test


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-prompts", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=1, help="Passes per training phase")
    args = parser.parse_args()
    if args.train_prompts < 1 or args.iterations < 1:
        parser.error("counts must be positive")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Submit this script as a GPU job; the CPU Jupyter server is only for preparation.")
    torch.cuda.set_device(0)
    if not torch.cuda.is_bf16_supported() or torch.cuda.get_device_properties(0).total_memory < 70 * 1024**3:
        raise RuntimeError("This configuration expects one A100 80GB (native BF16, >=70 GiB VRAM).")
    storage = Path(os.environ["LESA_STORAGE"])
    paths = json.loads((storage / "model_paths.json").read_text())
    for key, relative_path in paths.items():
        path = Path(os.environ["HF_HOME"]) / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        os.environ[key] = str(path)

    run = REPO / "runs" / ("flux-smoke-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"-{os.getpid()}")
    run.mkdir(parents=True, exist_ok=False)
    train, test = select_prompts((FLUX / "prompts/train_prompts_image.txt").read_text(),
                                 (FLUX / "prompts/DrawBench200.txt").read_text(), args.train_prompts)
    (run / "train.txt").write_text("\n".join(train) + "\n")
    (run / "test.txt").write_text("\n".join(test) + "\n")
    commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    config = dict(commit=commit, train_prompts=train, test_prompts=test,
                  iterations=args.iterations, width=1024, height=1024, num_steps=50,
                  interval=7, first_enhance=3, guidance=3.5, train_seed=0, test_seed=10000,
                  gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
    (run / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (run / "requirements.freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    print(f"RUN_DIR={run}", flush=True)
    # A fresh cwd isolates the original trainer's relative log/ directory.
    common = ["--prompt_file", str(run / "train.txt"), "--model_name", "flux-dev",
              "--num_steps", "50", "--width", "1024", "--height", "1024",
              "--guidance", "3.5", "--seed", "0", "--first_enhance", "3",
              "--data_dir", str(run / "data")]
    launcher = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1", str(FLUX / "src/train.py")]
    stages = [
        ("prepare", ["--data_prepare", "True", "--interval", "1", "--iterations", "1",
                     "--weights_dir", str(run / "unused-prepare-weights")]),
        ("gt-guided", ["--phase", "GT-Guided Training", "--interval", "7", "--iterations", str(args.iterations),
                       "--weights_dir", str(run / "weights")]),
        ("cl-ar", ["--phase", "CL-AR Training", "--interval", "7", "--iterations", str(args.iterations),
                   "--weights_dir", str(run / "weights")]),
    ]
    for name, extra in stages:
        print(f"STAGE={name}", flush=True)
        subprocess.run(launcher + common + extra + ["--output_dir", str(run / name)], cwd=run, check=True)
        if name == "gt-guided":
            import shutil
            shutil.copy2(run / "weights/predictor_flux_N7_E3.pt", run / "gt-guided.pt")
    subprocess.run([sys.executable, str(REPO / "scripts/mlspace/compare_flux.py"), "--run-dir", str(run)], cwd=run, check=True)
    print(f"LESA_SMOKE=PASS\nRESULTS={run / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
