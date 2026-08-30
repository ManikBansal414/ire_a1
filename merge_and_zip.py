import zipfile, time
from pathlib import Path

BASE = Path("outputs/submissions/ebnerd/lgbm")
parts = [BASE/"part1.txt", BASE/"part2.txt", BASE/"part3.txt"]
out = BASE/"predictions.txt"
zip_out = BASE/"submission.zip"

print("Waiting for all parts to finish...")
while True:
    done = all(p.exists() and p.stat().st_size > 1_000_000 for p in parts)
    sizes = [f"{p.stat().st_size//1_000_000}MB" if p.exists() else "missing" for p in parts]
    print(f"Parts: {sizes}")
    if done:
        # Check they stopped growing (workers finished)
        import time as t
        t.sleep(5)
        sizes2 = [p.stat().st_size for p in parts]
        t.sleep(5)
        sizes3 = [p.stat().st_size for p in parts]
        if sizes2 == sizes3:
            break
    t.sleep(30)

print("All parts done! Merging...")
with open(out, "w", encoding="utf-8") as fout:
    # Write the first 7.8M rows already in predictions.txt
    with open(out, "r") as base:
        pass  # Already there, we append parts
    for part_path in parts:
        with open(part_path, "r", encoding="utf-8") as fp:
            for line in fp:
                fout.write(line)

print("Zipping...")
with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(out, arcname="predictions.txt")
print(f"Done! Submission at {zip_out}")
