import subprocess
result = subprocess.run(
    ["poetry", "run", "kaggle", "kernels", "status", "leventecsibi/stable-diffusion-batch-generator"],
    capture_output=True,
    text=True,
    timeout=30,
    cwd="c:\\Users\\csibi\\Desktop\\imggenhub"
)
print(result.stdout.strip() if result.returncode == 0 else f"ERROR: {result.stderr}")
