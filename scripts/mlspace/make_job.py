"""Render a local YAML using the working RF-Solver ML Space job format."""
import argparse
import json
from pathlib import Path
import shlex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-repo", required=True, help="Absolute LESA path visible inside GPU jobs")
    parser.add_argument("--output", default=".runtime/flux_smoke_job.yaml")
    args = parser.parse_args()
    root = Path(args.worker_repo)
    if not root.is_absolute():
        parser.error("--worker-repo must be absolute")
    command = "bash " + shlex.quote(str(root / "scripts/mlspace/job.sh"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        'job:\n'
        '  description: "LESA FLUX smoke: 8 train / 2 test, one A100 80GB"\n'
        '  environment:\n'
        '    image: cr.ai.cloud.ru/aicloud-base-images/cuda12.1-torch2-py311:0.0.36\n'
        '    variables:\n'
        '      SSL_CERT_FILE: /etc/ssl/certs/ca-certificates.crt\n'
        '      REQUESTS_CA_BUNDLE: /etc/ssl/certs/ca-certificates.crt\n'
        '      CURL_CA_BUNDLE: /etc/ssl/certs/ca-certificates.crt\n'
        '  resource:\n'
        '    instance_type: a100plus.1gpu.80vG.12C.96G\n'
        f'  script: {json.dumps(command)}\n'
        '  type: binary\n'
    )
    print(output.resolve())
    print(output.read_text())


if __name__ == "__main__":
    main()
