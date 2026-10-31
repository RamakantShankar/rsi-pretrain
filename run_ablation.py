"""
Autonomous Ablation Runner & Experiment Ledger (RSI Loop).
Automates comparative pretraining runs, evaluates validation loss curves,
and logs structured ablation results to rank architectural/hyperparameter choices.
"""

import os
import copy
import json
import time
from typing import List, Dict, Any
import yaml

from train import load_config, main as run_train
import train


class AblationEngine:
    def __init__(self, base_config_path: str, results_file: str = "ablation_results.json"):
        self.base_config = load_config(base_config_path)
        self.results_file = results_file
        self.results = self._load_results()

    def _load_results(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.results_file):
            with open(self.results_file, "r") as f:
                return json.load(f)
        return []

    def _save_results(self):
        with open(self.results_file, "w") as f:
            json.dump(self.results, f, indent=2)

    def run_experiment(self, exp_name: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Runs a single isolated ablation run with modified hyperparameters."""
        print(f"\n{'='*30}\n[Ablation Runner] Launching Experiment: {exp_name}\n{'='*30}")

        # Deep copy base config and apply overrides
        cfg = copy.deepcopy(self.base_config)
        cfg["experiment_name"] = exp_name

        for section, params in overrides.items():
            if section in cfg and isinstance(params, dict):
                cfg[section].update(params)
            else:
                cfg[section] = params

        temp_cfg_path = f"config/temp_{exp_name}.yaml"
        with open(temp_cfg_path, "w") as f:
            yaml.dump(cfg, f)

        # Mock argv to train with temp_cfg_path
        start_time = time.time()
        import sys
        sys.argv = ["train.py", "--config", temp_cfg_path]

        try:
            run_train()
            duration = time.time() - start_time
            status = "SUCCESS"
        except Exception as e:
            duration = time.time() - start_time
            status = f"FAILED: {str(e)}"
            print(f"[Ablation Runner] Error during {exp_name}: {e}")

        # Clean up temp config
        if os.path.exists(temp_cfg_path):
            os.remove(temp_cfg_path)

        # Read metrics from checkpoint log
        metrics_file = os.path.join("checkpoints", exp_name, "metrics.jsonl")
        best_val_loss = None
        final_train_loss = None

        if os.path.exists(metrics_file):
            with open(metrics_file, "r") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if "val_loss" in entry:
                        val = entry["val_loss"]
                        if best_val_loss is None or val < best_val_loss:
                            best_val_loss = val
                    if "train_loss" in entry:
                        final_train_loss = entry["train_loss"]

        result = {
            "experiment_name": exp_name,
            "status": status,
            "overrides": overrides,
            "best_val_loss": best_val_loss,
            "final_train_loss": final_train_loss,
            "duration_sec": round(duration, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        self.results.append(result)
        self._save_results()
        return result

    def print_summary(self):
        """Displays formatted comparison table of all ablation runs."""
        print(f"\n{'='*35} ABLATION LEADERBOARD {'='*35}")
        print(f"{'Experiment':<25} | {'Status':<8} | {'Best Val Loss':<14} | {'Train Loss':<11} | {'Time (s)':<8}")
        print("-" * 75)

        sorted_results = sorted(
            self.results,
            key=lambda r: (r.get("best_val_loss") is None, r.get("best_val_loss", float("inf")))
        )

        for r in sorted_results:
            val_str = f"{r['best_val_loss']:.4f}" if r['best_val_loss'] is not None else "N/A"
            train_str = f"{r['final_train_loss']:.4f}" if r['final_train_loss'] is not None else "N/A"
            print(f"{r['experiment_name']:<25} | {r['status']:<8} | {val_str:<14} | {train_str:<11} | {r['duration_sec']:<8.1f}")
        print("=" * 75)


if __name__ == "__main__":
    engine = AblationEngine(base_config_path="config/micro_local.yaml")

    # Ablation 1: Grouped-Query Attention (GQA 2 KV heads) vs Standard Multi-Head Attention (MHA 6 KV heads)
    print("\n[Ablation Study] Testing GQA vs MHA efficiency and convergence...")
    
    # Run 1: GQA baseline (n_heads=6, n_kv_heads=2)
    engine.run_experiment(
        exp_name="ablation_gqa_kv2",
        overrides={"model": {"n_heads": 6, "n_kv_heads": 2}, "training": {"total_steps": 50, "eval_interval": 25}}
    )

    # Run 2: Full MHA (n_heads=6, n_kv_heads=6)
    engine.run_experiment(
        exp_name="ablation_mha_kv6",
        overrides={"model": {"n_heads": 6, "n_kv_heads": 6}, "training": {"total_steps": 50, "eval_interval": 25}}
    )

    # Output leaderboard
    engine.print_summary()
