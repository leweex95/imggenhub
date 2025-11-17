from pathlib import Path
status_file = Path("c:\\Users\\csibi\\Desktop\\imggenhub\\kernel_status.txt")
if status_file.exists():
    print(status_file.read_text())
else:
    print("Status file not found - kernel still running")
