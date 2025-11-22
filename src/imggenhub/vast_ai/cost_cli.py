"""
Vast.ai cost estimation and budget planning tool.
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

from imggenhub.vast_ai.cost_tracking import CostEstimator, RunLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def estimate_command(args):
    """Handle cost estimation."""
    logger.info("=== Vast.ai Cost Estimation ===")

    if args.subcommand == "generation":
        if not args.num_images or args.num_images < 1:
            raise ValueError("--num-images must be >= 1")
        if not args.avg_time_sec or args.avg_time_sec < 1:
            raise ValueError("--avg-time-sec must be >= 1")
        if not args.gpu_dph or args.gpu_dph <= 0:
            raise ValueError("--gpu-dph must be > 0")

        result = CostEstimator.estimate_generation_cost(
            num_images=args.num_images,
            avg_time_per_image=args.avg_time_sec,
            instance_dph=args.gpu_dph,
        )

        logger.info(f"Generating {args.num_images} images")
        logger.info(f"  Model avg time: {args.avg_time_sec} sec/image")
        logger.info(f"  GPU rate: ${args.gpu_dph}/hr")
        logger.info("")
        logger.info("Estimated Costs:")
        logger.info(f"  Total duration: {result['total_duration_min']:.1f} minutes")
        logger.info(f"  Total cost: ${result['total_cost']:.4f}")
        logger.info(f"  Cost per image: ${result['cost_per_image']:.4f}")

    elif args.subcommand == "budget":
        if not args.budget or args.budget <= 0:
            raise ValueError("--budget must be > 0")
        if not args.gpu_dph or args.gpu_dph <= 0:
            raise ValueError("--gpu-dph must be > 0")

        result = CostEstimator.recommend_model_for_budget(
            budget_usd=args.budget,
            instance_dph=args.gpu_dph,
        )

        logger.info(f"Budget: ${args.budget}")
        logger.info(f"GPU rate: ${args.gpu_dph}/hr")
        logger.info("")
        logger.info("Model Recommendations:")

        for i, rec in enumerate(result['recommendations'], 1):
            logger.info(f"\n{i}. {rec['model']}")
            logger.info(f"   Speed: {rec['speed']}")
            logger.info(f"   Quality: {rec['quality']}")
            logger.info(f"   Estimated images: {rec['estimated_images']}")
            logger.info(f"   Avg time: {rec['avg_time_sec']}s per image")
            logger.info(f"   Recommended settings:")
            logger.info(f"     Guidance: {rec['guidance']}")
            logger.info(f"     Steps: {rec['steps']}")

    elif args.subcommand == "model":
        models = {
            "flux-schnell": {
                "model_name": "black-forest-labs/FLUX.1-schnell",
                "avg_time_sec": 30,
                "quality": "good",
                "vram_gb": 15,
                "precision": "fp16",
                "guidance": "3.5-5.0",
                "steps": "10-20",
            },
            "sd35-large": {
                "model_name": "stabilityai/stable-diffusion-3.5-large",
                "avg_time_sec": 60,
                "quality": "very-good",
                "vram_gb": 19,
                "precision": "fp16",
                "guidance": "7.5-9.0",
                "steps": "30-40",
            },
            "flux-pro": {
                "model_name": "black-forest-labs/FLUX.1-pro",
                "avg_time_sec": 90,
                "quality": "best",
                "vram_gb": 22,
                "precision": "fp16",
                "guidance": "7.5-12.0",
                "steps": "30-50",
            },
        }

        if args.model not in models:
            logger.error(f"Unknown model. Available: {list(models.keys())}")
            return

        model_info = models[args.model]
        logger.info(f"Model: {model_info['model_name']}")
        logger.info(f"  Quality: {model_info['quality']}")
        logger.info(f"  Avg time: {model_info['avg_time_sec']}s per image")
        logger.info(f"  VRAM: {model_info['vram_gb']}GB (P40 has 24GB)")
        logger.info(f"  Precision: {model_info['precision']}")
        logger.info(f"  Recommended settings:")
        logger.info(f"    Guidance: {model_info['guidance']}")
        logger.info(f"    Steps: {model_info['steps']}")

        if args.gpu_dph:
            cost_10 = (args.gpu_dph * model_info['avg_time_sec'] * 10) / 3600
            cost_100 = (args.gpu_dph * model_info['avg_time_sec'] * 100) / 3600
            logger.info(f"\n  Estimated costs at ${args.gpu_dph}/hr:")
            logger.info(f"    10 images: ${cost_10:.2f}")
            logger.info(f"    100 images: ${cost_100:.2f}")


def log_command(args):
    """Handle run logging."""
    logger.info("=== Vast.ai Run Logging ===")

    run_logger = RunLogger(output_dir=Path(args.output_dir))

    if args.subcommand == "view":
        log_file = Path(args.log_file)
        if not log_file.exists():
            logger.error(f"Log file not found: {log_file}")
            return

        with open(log_file, 'r') as f:
            log_data = json.load(f)

        summary = log_data.get('summary', {})
        logger.info("Run Summary:")
        logger.info(f"  Total runs: {summary.get('total_runs', 0)}")
        logger.info(f"  Total images: {summary.get('total_images', 0)}")
        logger.info(f"  Total duration: {summary.get('total_duration_hours', 0):.2f} hours")
        logger.info(f"  Total cost: ${summary.get('total_cost', 0):.4f}")
        logger.info(f"  Avg cost per image: ${summary.get('avg_cost_per_image', 0):.4f}")

        logger.info("\nRecent runs:")
        runs = log_data.get('runs', [])
        for run in runs[-5:]:
            logger.info(f"  {run['timestamp']}: {run['num_images']} images, ${run['total_cost']:.4f} ({run['model_name']})")

    elif args.subcommand == "add":
        if not args.instance_id or not args.model or not args.num_images or not args.duration_sec or not args.gpu_dph:
            raise ValueError("--instance-id, --model, --num-images, --duration-sec, --gpu-dph are required")

        run_logger.load_log(Path(args.log_file)) if Path(args.log_file).exists() else None

        run_logger.log_run(
            instance_id=args.instance_id,
            instance_type=args.instance_type or "unknown",
            instance_dph=args.gpu_dph,
            model_name=args.model,
            num_images=args.num_images,
            duration_seconds=args.duration_sec,
            status=args.status or "completed",
            notes=args.notes,
        )

        log_file = run_logger.save_log(Path(args.log_file) if args.log_file else None)
        logger.info(f"Run logged and saved to: {log_file}")


def main():
    parser = argparse.ArgumentParser(description="Vast.ai cost estimation and run logging")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Estimate command
    estimate_parser = subparsers.add_parser("estimate", help="Estimate costs")
    estimate_subparsers = estimate_parser.add_subparsers(dest="subcommand", required=True)

    # Generation estimation
    gen_parser = estimate_subparsers.add_parser("generation", help="Estimate generation costs")
    gen_parser.add_argument("--num-images", type=int, required=True, help="Number of images")
    gen_parser.add_argument("--avg-time-sec", type=float, required=True, help="Average seconds per image")
    gen_parser.add_argument("--gpu-dph", type=float, required=True, help="GPU price per hour (USD)")
    gen_parser.set_defaults(func=estimate_command)

    # Budget recommendation
    budget_parser = estimate_subparsers.add_parser("budget", help="Recommend models for budget")
    budget_parser.add_argument("--budget", type=float, required=True, help="Budget in USD")
    budget_parser.add_argument("--gpu-dph", type=float, required=True, help="GPU price per hour (USD)")
    budget_parser.set_defaults(func=estimate_command)

    # Model info
    model_parser = estimate_subparsers.add_parser("model", help="Get model info and costs")
    model_parser.add_argument("--model", type=str, required=True, choices=["flux-schnell", "sd35-large", "flux-pro"], help="Model name")
    model_parser.add_argument("--gpu-dph", type=float, default=None, help="GPU price per hour (optional, for cost estimate)")
    model_parser.set_defaults(func=estimate_command)

    # Log command
    log_parser = subparsers.add_parser("log", help="Manage run logs")
    log_subparsers = log_parser.add_subparsers(dest="subcommand", required=True)

    # View logs
    view_parser = log_subparsers.add_parser("view", help="View run logs")
    view_parser.add_argument("--log-file", type=str, default="vast_ai_runs.json", help="Log file path")
    view_parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    view_parser.set_defaults(func=log_command)

    # Add run
    add_parser = log_subparsers.add_parser("add", help="Add a new run to logs")
    add_parser.add_argument("--instance-id", type=str, required=True, help="Vast.ai instance ID")
    add_parser.add_argument("--instance-type", type=str, default=None, help="Instance type (e.g., tesla-p40)")
    add_parser.add_argument("--model", type=str, required=True, help="Model name used")
    add_parser.add_argument("--num-images", type=int, required=True, help="Number of images generated")
    add_parser.add_argument("--duration-sec", type=float, required=True, help="Duration in seconds")
    add_parser.add_argument("--gpu-dph", type=float, required=True, help="GPU price per hour (USD)")
    add_parser.add_argument("--status", type=str, default="completed", choices=["completed", "failed", "cancelled"])
    add_parser.add_argument("--notes", type=str, default=None, help="Additional notes")
    add_parser.add_argument("--log-file", type=str, default="vast_ai_runs.json", help="Log file path")
    add_parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    add_parser.set_defaults(func=log_command)

    args = parser.parse_args()

    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except ValueError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
