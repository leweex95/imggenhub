#!/usr/bin/env python3
"""
Download kernel output and check for success.
This script runs after the kernel completes.
"""
import subprocess
import os
from pathlib import Path

output_dir = Path("c:\\Users\\csibi\\Desktop\\imggenhub\\kernel_output_final")
output_dir.mkdir(exist_ok=True)

print("Downloading kernel output...")
result = subprocess.run(
    ["poetry", "run", "kaggle", "kernels", "output", 
     "leventecsibi/stable-diffusion-batch-generator", "-p", str(output_dir)],
    capture_output=True,
    text=True,
    cwd="c:\\Users\\csibi\\Desktop\\imggenhub"
)

if result.returncode != 0:
    print(f"Error downloading: {result.stderr[:500]}")
else:
    print("Download successful")
    
    # Check for log file
    log_file = output_dir / "stable-diffusion-batch-generator.log"
    if log_file.exists():
        size = log_file.stat().st_size
        print(f"\nLog file size: {size} bytes")
        if size > 0:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                if len(content) > 2000:
                    print("Last 2000 chars of log:")
                    print(content[-2000:])
                else:
                    print(content)
        else:
            print("Log is empty")
    
    # Check for images
    img_dir = output_dir / "output_images_flux_official"
    if img_dir.exists():
        images = list(img_dir.glob("*.png"))
        print(f"\n✓ Generated {len(images)} images!")
        for img in images[:3]:
            print(f"  - {img.name}")
    else:
        print("\n✗ No image output directory found")
