import json
from pathlib import Path

verified_path = Path("data/rsi/verified_solutions_50m.jsonl")
problem_bank_path = Path("data/rsi/problem_bank.json")

with open(problem_bank_path, "r", encoding="utf-8") as f:
    bank = json.load(f)

# Find str_01 (reverse_string) and str_02 (is_palindrome)
str_tasks = [p for p in bank if p["task_id"] in ("str_01", "str_02")]

print(f"Found {len(str_tasks)} string anchors to seed:")
anchors = []
for t in str_tasks:
    canonical = t.get("canonical") or t.get("canonical_solution")
    full_code = t["prompt"].rstrip() + "\n" + canonical.lstrip("\n")
    rec = {
        "task_id": t["task_id"],
        "name": t["name"],
        "prompt": t["prompt"],
        "solution": full_code,
        "tests": t["tests"],
        "candidate_idx": 0,
        "exec_time": 0.001,
        "is_anchor": True,
    }
    anchors.append(rec)
    print(f"\n--- Anchor: {t['task_id']} ({t['name']}) ---")
    print(rec["solution"])

with open(verified_path, "a", encoding="utf-8") as f:
    for a in anchors:
        f.write(json.dumps(a) + "\n")

print(f"\nSuccessfully seeded {len(anchors)} anchors into {verified_path}!")
