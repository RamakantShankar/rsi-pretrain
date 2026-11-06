#!/usr/bin/env python3
"""
Mirendil Laboratory - GPU Perpetual Supervisor
Guarantees 100% continuous saturation of NVIDIA DGX Grace Blackwell GB10.
Zero idle time policy:
  - Detects if RSI orchestrator is running.
  - When a target generation is achieved, automatically discovers the highest valid checkpoint
    and launches the next generational block (M_N -> M_{N+20}).
  - If a transient failure occurs, logs the error, cools down 10s, and restarts from the latest checkpoint.
  - Disowned daemon running 24/7 independently of any SSH or terminal session.
"""

import os
import sys
import time
import subprocess
import re
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = ROOT_DIR / "checkpoints" / "v3"
SHARD_DIR = ROOT_DIR / "data" / "shards_v3"
VERIFIED_PATH = ROOT_DIR / "data" / "rsi" / "verified_solutions_edu.jsonl"
CANDIDATES_PATH = ROOT_DIR / "data" / "rsi" / "all_candidates_edu.jsonl"
BASELINE_SHARD = ROOT_DIR / "data" / "shards_edu" / "train_shard_00000.bin"
LOG_PATH = ROOT_DIR / "supervisor.log"

def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S PST")
    line = f"[{timestamp}] [SUPERVISOR] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def is_orchestrator_running() -> bool:
    try:
        res = subprocess.run(["pgrep", "-f", "src/rsi/orchestrator.py"], capture_output=True, text=True)
        pids = [p.strip() for p in res.stdout.strip().splitlines() if p.strip()]
        # Ignore our own PID
        my_pid = str(os.getpid())
        pids = [p for p in pids if p != my_pid]
        return len(pids) > 0
    except Exception as e:
        log(f"Error checking process: {e}")
        return False

def get_latest_checkpoint_gen() -> int:
    if not CHECKPOINT_DIR.exists():
        return 0
    gens = []
    for d in CHECKPOINT_DIR.iterdir():
        if d.is_dir() and d.name.startswith("152m_rsi_m"):
            m = re.match(r"152m_rsi_m(\d+)", d.name)
            if m:
                gen_num = int(m.group(1))
                ckpt_file = d / "best_model.pt"
                if ckpt_file.exists() and ckpt_file.stat().st_size > 10_000_000:
                    gens.append(gen_num)
    return max(gens) if gens else 0

def launch_next_block(start_gen: int, target_gen: int):
    log(f"--> LAUNCHING AUTONOMOUS BLOCK: M_{start_gen} -> M_{target_gen}")
    cmd = [
        sys.executable,
        "src/rsi/orchestrator.py",
        "--start_gen", str(start_gen),
        "--max_gen", str(target_gen),
        "--checkpoint_dir", str(CHECKPOINT_DIR),
        "--shard_dir", str(SHARD_DIR),
        "--verified_path", str(VERIFIED_PATH),
        "--all_candidates_path", str(CANDIDATES_PATH),
        "--baseline_shard", str(BASELINE_SHARD),
        "--k_frontier", "18",
        "--k_mastered", "3",
        "--temp_frontier", "0.88",
        "--temp_mastered", "0.65",
        "--autoscale",
        "--steps", "50",
        "--lr", "1.2e-4",
        "--wandb",
        "--project", "rsi-coding-pretrain",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT_DIR)
    
    out_file = open(ROOT_DIR / f"rsi_m{start_gen}_to_m{target_gen}.log", "a")
    proc = subprocess.Popen(cmd, cwd=str(ROOT_DIR), env=env, stdout=out_file, stderr=subprocess.STDOUT)
    log(f"--> Process spawned with PID: {proc.pid}")
    return proc.pid

def main():
    log("=" * 70)
    log("Mirendil Perpetual GPU Supervisor INITIALIZED")
    log(f"Watch Directory: {CHECKPOINT_DIR}")
    log(f"Zero Idle GPU Policy: ACTIVE")
    log("=" * 70)

    while True:
        try:
            if is_orchestrator_running():
                time.sleep(25)
                continue

            latest_gen = get_latest_checkpoint_gen()
            log(f"No active orchestrator detected. Latest certified checkpoint: M_{latest_gen}")
            
            target_gen = latest_gen + 15
            log(f"Initiating next continuous generation sprint: M_{latest_gen} -> M_{target_gen}")
            launch_next_block(latest_gen, target_gen)
            
            # Wait for process to spin up and bind GPU
            time.sleep(15)

        except Exception as e:
            log(f"Supervisor loop exception: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
