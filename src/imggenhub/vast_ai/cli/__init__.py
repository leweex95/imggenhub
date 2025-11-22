"""CLI interface for Vast.ai remote execution and automation."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable, Optional

from imggenhub.vast_ai.api import VastAiClient
from imggenhub.vast_ai.auto_deploy import AutoDeploymentOrchestrator, SearchCriteria
from imggenhub.vast_ai.pipelines import PipelineOrchestrator
from imggenhub.vast_ai.types import RemoteRunResult

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")
    else:
        root.setLevel(level)


def _add_api_key_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--api-key", dest="api_key", type=str, default=None, help="Vast.ai API key (defaults to .env)")


def _add_search_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gpu-name", type=str, default=None, help="Exact GPU model filter (e.g., 'RTX 4090')")
    parser.add_argument("--min-vram", type=int, default=24, help="Minimum VRAM required in GB (default: 24)")
    parser.add_argument("--max-price", type=float, default=1.0, help="Maximum hourly price in USD (default: $1.00)")
    parser.add_argument("--min-reliability", type=float, default=95.0, help="Minimum reliability percentage (default: 95)")
    parser.add_argument("--no-spot", action="store_true", help="Exclude spot instances (prefers on-demand only)")


def _add_generation_arguments(parser: argparse.ArgumentParser, *, include_instance: bool) -> None:
    if include_instance:
        parser.add_argument("--instance-id", type=int, required=True, help="Vast.ai instance ID")

    parser.add_argument("--model-name", type=str, default="stabilityai/stable-diffusion-3.5-large", help="Model identifier to run")
    parser.add_argument("--guidance", type=float, default=8.5, help="Guidance scale (default: 8.5)")
    parser.add_argument("--steps", type=int, default=40, help="Inference steps (default: 40)")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "int8", "int4"], help="Computation precision")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt override")
    parser.add_argument("--prompts-file", type=str, default="config/prompts.json", help="Remote prompts JSON (default: config/prompts.json)")
    parser.add_argument("--negative-prompt", type=str, default=None, help="Negative prompt")
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace token for gated models")
    parser.add_argument("--refiner-model-name", type=str, default=None, help="Optional refiner model")
    parser.add_argument("--refiner-guidance", type=float, default=None, help="Refiner guidance scale")
    parser.add_argument("--refiner-steps", type=int, default=None, help="Refiner inference steps")
    parser.add_argument("--refiner-precision", type=str, default=None, choices=["fp32", "fp16", "int8", "int4"], help="Refiner precision")
    parser.add_argument("--expected-images", type=int, default=4, help="Number of expected images for logging")
    parser.add_argument("--ssh-key", type=str, default="~/.ssh/id_ed25519", help="Path to SSH private key")
    parser.add_argument("--repo-path", type=str, default=".", help="Local repository root to sync")
    parser.add_argument("--config-path", type=str, default="src/imggenhub/kaggle/config", help="Config directory to sync")


def _build_search_criteria(args: argparse.Namespace) -> SearchCriteria:
    return SearchCriteria(
        min_vram=args.min_vram,
        max_price=args.max_price,
        gpu_name=args.gpu_name,
        prefer_spot=not args.no_spot,
        min_reliability=args.min_reliability,
    )


def _print_offers(offers: list[dict], limit: int) -> None:
    if not offers:
        print("No offers matched the given criteria.")
        return

    header = f"{'Offer':>8}  {'GPU':<20}  {'VRAM':>5}  {'$/hr':>8}  {'Reliab%':>8}  {'Location':<10}"
    print(header)
    print("-" * len(header))
    for offer in offers[:limit]:
        print(
            f"{offer.get('id'):>8}  {offer.get('gpu_name','-'):<20}  "
            f"{offer.get('gpu_total_ram','-'):>5}  {offer.get('dph_total','-'):>8}  "
            f"{offer.get('reliability2','-'):>8}  {offer.get('location','-'):<10}"
        )


def _expand_ssh_key(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return str(Path(path).expanduser())


def handle_list(args: argparse.Namespace) -> int:
    orchestrator = AutoDeploymentOrchestrator(api_key=args.api_key)
    offers = orchestrator.search_and_rank(_build_search_criteria(args))
    _print_offers(offers, args.limit)
    return 0 if offers else 1


def handle_reserve(args: argparse.Namespace) -> int:
    client = VastAiClient(args.api_key)
    instance = client.create_instance(
        offer_id=args.offer_id,
        image=args.image,
        disk_size=args.disk_size,
        label=args.label,
    )

    print("Instance created:")
    print(f"  Instance ID : {instance.id}")
    print(f"  GPU         : {instance.gpu_name}")
    print(f"  Hourly cost : ${instance.price_per_hour:.4f}")
    print(f"  SSH command : ssh -p {instance.ssh_port} {instance.ssh_user}@{instance.ssh_host}")
    return 0


def handle_run(args: argparse.Namespace) -> int:
    pipeline = PipelineOrchestrator(
        api_key=args.api_key,
        repo_path=args.repo_path,
        config_path=args.config_path,
    )

    execution = pipeline.run_model(
        instance_id=args.instance_id,
        ssh_key=_expand_ssh_key(args.ssh_key),
        model_name=args.model_name,
        guidance=args.guidance,
        steps=args.steps,
        precision=args.precision,
        prompt=args.prompt,
        prompts_file=args.prompts_file,
        negative_prompt=args.negative_prompt,
        expected_images=args.expected_images,
        hf_token=args.hf_token,
    )

    result = execution.result
    print(f"Run finished with exit code {result.exit_code}")
    if result.log_file:
        print(f"Log file      : {result.log_file}")
    print(f"Output folder : {result.output_dir}")
    return result.exit_code


def handle_auto(args: argparse.Namespace) -> int:
    orchestrator = AutoDeploymentOrchestrator(api_key=args.api_key, ssh_key_path=_expand_ssh_key(args.ssh_key))
    criteria = _build_search_criteria(args)

    try:
        instance = orchestrator.rent_cheapest(
            criteria,
            image=args.image,
            disk_size=args.disk_size,
            label=args.label,
        )
    except Exception as exc:  # pragma: no cover - network dependent
        logger.error("Failed to rent instance: %s", exc)
        return 1

    if not orchestrator.wait_for_ssh_ready(timeout=args.ssh_timeout):
        if not args.preserve_on_failure:
            orchestrator.destroy_instance()
        return 1

    pipeline = PipelineOrchestrator(
        api_key=args.api_key,
        repo_path=args.repo_path,
        config_path=args.config_path,
    )

    execution = pipeline.run_model(
        instance_id=instance.instance_id,
        ssh_key=_expand_ssh_key(args.ssh_key),
        model_name=args.model_name,
        guidance=args.guidance,
        steps=args.steps,
        precision=args.precision,
        prompt=args.prompt,
        prompts_file=args.prompts_file,
        negative_prompt=args.negative_prompt,
        expected_images=args.expected_images,
        hf_token=args.hf_token,
    )

    cost = instance.price_per_hour * (execution.duration_seconds / 3600)
    print(f"Generation completed in {execution.duration_seconds:.1f}s (~${cost:.4f})")

    exit_code = execution.result.exit_code
    if exit_code != 0 and args.preserve_on_failure:
        logger.warning("Instance preserved for debugging: %s", instance.instance_id)
        return exit_code

    if not args.keep_instance:
        orchestrator.destroy_instance()

    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage Vast.ai instances and deployments")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    list_parser = subparsers.add_parser("list", help="List available GPU offers")
    _add_api_key_argument(list_parser)
    _add_search_arguments(list_parser)
    list_parser.add_argument("--limit", type=int, default=10, help="Number of offers to display (default: 10)")
    list_parser.set_defaults(handler=handle_list)

    # reserve
    reserve_parser = subparsers.add_parser("reserve", help="Reserve a specific offer by ID")
    _add_api_key_argument(reserve_parser)
    reserve_parser.add_argument("--offer-id", type=int, required=True, help="Offer/ask ID returned by the list command")
    reserve_parser.add_argument("--image", type=str, default="pytorch/pytorch:latest", help="Docker image to boot (default: pytorch/pytorch:latest)")
    reserve_parser.add_argument("--disk-size", type=int, default=40, help="Disk size in GB (default: 40)")
    reserve_parser.add_argument("--label", type=str, default=None, help="Optional label for the instance")
    reserve_parser.set_defaults(handler=handle_reserve)

    # run
    run_parser = subparsers.add_parser("run", help="Deploy and run a model on an existing instance")
    _add_api_key_argument(run_parser)
    _add_generation_arguments(run_parser, include_instance=True)
    run_parser.set_defaults(handler=handle_run)

    # auto
    auto_parser = subparsers.add_parser("auto", help="Search, rent, run, and optionally destroy automatically")
    _add_api_key_argument(auto_parser)
    _add_search_arguments(auto_parser)
    _add_generation_arguments(auto_parser, include_instance=False)
    auto_parser.add_argument("--ssh-timeout", type=int, default=300, help="Seconds to wait for SSH (default: 300)")
    auto_parser.add_argument("--image", type=str, default="pytorch/pytorch:latest", help="Docker image to boot")
    auto_parser.add_argument("--disk-size", type=int, default=40, help="Disk size in GB (default: 40)")
    auto_parser.add_argument("--label", type=str, default=None, help="Optional label for the rented instance")
    auto_parser.add_argument("--keep-instance", action="store_true", help="Keep instance running after completion")
    auto_parser.add_argument("--preserve-on-failure", action="store_true", help="Do not destroy instance if generation fails")
    auto_parser.set_defaults(handler=handle_auto)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)

    handler: Callable[[argparse.Namespace], int] = args.handler
    try:
        return handler(args)
    except Exception as exc:  # pragma: no cover - CLI safety net
        logger.error("Command failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
