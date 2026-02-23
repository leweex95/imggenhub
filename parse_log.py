import json

with open("C:/temp/koutput/imggenhub-generator.log", "r") as f:
    content = f.read()

print(f"Length: {len(content)}")

try:
    data = json.loads(content)
    print(f"Parsed as JSON array: {len(data)} entries")
    for entry in data:
        print(f"t={entry.get('time', 0):.2f} [{entry.get('stream_name', '')}]: {entry.get('data', '')}", end="")
        print()
except Exception as e:
    print(f"Parse error: {e}")
    for i, line in enumerate(content.splitlines()[:30]):
        print(f"Line {i}: {repr(line[:300])}")
