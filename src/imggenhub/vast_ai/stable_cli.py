"""CLI for running Stable Diffusion pipelines on Vast.ai."""
from __future__ import annotations

import argparse
import sys
from typing import Optional

from imggenhub.vast_ai.pipelines import PipelineOrchestrator


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stable Diffusion deployment on Vast.ai")
    parser.add_argument("--instance-id", type=int, required=True, help="Active Vast.ai instance ID")
    parser.add_argument("--ssh-key", dest="ssh_key", type=str, required=True, help="Path to SSH private key")
    parser.add_argument("--api-key", dest="api_key", type=str, default=None, help="Vast.ai API key")
    parser.add_argument("--repo-path", dest="repo_path", type=str, default=".", help="Local repository root to upload")
    parser.add_argument(
        "--config-path",
        dest="config_path",
        type=str,
        default="src/imggenhub/kaggle/config",
        help="Config directory to upload",
    )
    parser.add_argument("--prompts-file", dest="prompts_file", type=str, default="config/prompts.json", help="Remote prompts file path")
    parser.add_argument("--prompt", dest="prompt", type=str, default=None, help="Single prompt override")
    parser.add_argument("--negative-prompt", dest="negative_prompt", type=str, default=None, help="Negative prompt")
    parser.add_argument("--model-name", dest="model_name", type=str, default="stabilityai/stable-diffusion-3.5-large", help="Model identifier")
    parser.add_argument("--guidance", dest="guidance", type=float, default=8.5, help="Guidance scale")
    parser.add_argument("--steps", dest="steps", type=int, default=40, help="Inference steps")
    parser.add_argument("--precision", dest="precision", type=str, default="fp16", choices=["fp32", "fp16", "int8", "int4"], help="Numerical precision")
    parser.add_argument("--expected-images", dest="expected_images", type=int, default=4, help="Images expected per run for logging")
    parser.add_argument("--hf-token", dest="hf_token", type=str, default=None, help="Hugging Face token passed to the pipeline")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    orchestrator = PipelineOrchestrator(
        api_key=args.api_key,
        repo_path=args.repo_path,
        config_path=args.config_path,
    )

    try:
        execution = orchestrator.run_stable_diffusion(
            instance_id=args.instance_id,
            ssh_key=args.ssh_key,
            prompts_file=args.prompts_file,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            model_name=args.model_name,
            guidance=args.guidance,
            steps=args.steps,
            precision=args.precision,
            expected_images=args.expected_images,
            hf_token=args.hf_token,
        )
    except Exception as exc:
        print(f"Stable Diffusion deployment failed: {exc}", file=sys.stderr)
        return 1

    result = execution.result
    print(f"Stable Diffusion run finished with exit code {result.exit_code}")
    if result.log_file:
        print(f"Log file: {result.log_file}")
    print(f"Artifacts downloaded to: {result.output_dir}")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
