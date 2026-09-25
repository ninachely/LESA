"""Download on the CPU server, using HF authentication stored by `hf auth login`."""
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


def main():
    storage = Path(os.environ["LESA_STORAGE"]).resolve()
    hf_home = Path(os.environ["HF_HOME"]).resolve()
    storage.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, filename in (("FLUX_MODEL", "flux1-dev.safetensors"), ("FLUX_AE", "ae.safetensors")):
        path = hf_hub_download("black-forest-labs/FLUX.1-dev", filename)
        # Relative paths survive different NFS mount points in notebook and job.
        paths[key] = str(Path(path).resolve().relative_to(hf_home))
    snapshot_download("google/t5-v1_1-xxl", allow_patterns=[
        "*.json", "spiece.model", "pytorch_model.bin",
    ])
    snapshot_download("openai/clip-vit-large-patch14", allow_patterns=[
        "*.json", "merges.txt", "model.safetensors",
    ])
    (storage / "model_paths.json").write_text(json.dumps(paths, indent=2) + "\n")
    print("Models downloaded. Ready to submit the GPU job.")


if __name__ == "__main__":
    main()
