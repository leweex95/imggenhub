import json

with open("C:/tmp/kagout2/imggenhub-generator.log") as f:
    lines = f.read().strip().split("\n")

entries = []
for line in lines:
    try:
        entries.append(json.loads(line))
    except Exception:
        pass

stdout = [e for e in entries if e.get("stream_name") == "stdout"]
print(f"Total log entries: {len(entries)}, stdout entries: {len(stdout)}")
print("=" * 60)
for e in stdout[:50]:
    print(f"[{e['time']:.1f}s] {e['data'][:120]!r}")
