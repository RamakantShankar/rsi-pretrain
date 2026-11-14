#!/usr/bin/env python3
"""
Mirendil Laboratory - Interactive Architectural & Generative RSI Comparison Dashboard
Generates a publication-grade, fully interactive single-page application:
  - Multi-Architecture Trajectory Visualization (Looped 84M Champion vs Flat 152M Baseline vs Flat Edu)
  - Interactive Series Inspector: Click on any series/architecture to inspect parameter allocation,
    vocabulary tax, virtual depth math, memory bandwidth, and FLOP efficiency.
  - Interactive Metric Deep-Dive: Click on any metric (Pass@1, Pass@5, Candidate Density, Gold Harvest,
    AST Syntax, Decode Speed) for mathematical definitions, information-theoretic interpretations, and comparisons.
  - Interactive Data-Point Inspector: Click any generation point on the chart to inspect newly conquered
    algorithmic tasks (e.g. math_10, list_08, list_09), W&B telemetry runs, loss, and hardware execution time.
  - Interactive Parameter Allocation & Domain Mastery Breakdown
  - Light/Dark mode with automatic system sync and local persistence
  - 100% bit-for-bit mirrored to Google Drive.
"""

import sys
import os
import json
import shutil
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path("M:/Rama/Study/llm/LLM/RSI")
GDRIVE_DIR = Path("D:/D/Google Drive/RSI_Pretraining_Lab")

# Complete Historical Trajectory Across All Generations
DATASET = {
    "generations": [
        "M0",
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
        "M6",
        "M7",
        "M8",
        "M9",
        "M10",
        "M11",
        "M12",
        "M13",
        "M14",
        "M15",
        "M16",
        "M17",
        "M18",
        "M19",
        "M20",
        "M21",
        "M22",
        "M23",
        "M24",
        "M25",
        "M26",
        "M27",
        "M28",
        "M29",
        "M30",
        "M31",
        "M32"
    ],
    "models": {
        "looped_84m": {
            "id": "looped_84m",
            "name": "Mirendil-84M Looped Multi-Depth (Champion)",
            "short_name": "Looped 84M (T=2, 16L virt)",
            "color": "#4f46e5",
            "bg_color": "rgba(79, 70, 229, 0.12)",
            "accent": "#818cf8",
            "total_params": 84427520,
            "non_emb_params": 71910144,
            "emb_params": 12517376,
            "emb_tax_pct": 14.82,
            "vocab_size": 16384,
            "tokenizer": "Custom Mirendil Byte-Pair BPE (16k)",
            "phys_layers": 8,
            "recurrent_loops": "T=2 (16 Virtual Layers, Multi-Exit T\u2208{1,2,3,4})",
            "d_model": 768,
            "n_heads": 12,
            "kv_heads": 4,
            "head_dim": 64,
            "ffn_dim": 2048,
            "weight_size_mb": 168.9,
            "decode_tok_s": 130.3,
            "fast_reflex_tok_s": 246.8,
            "inter_loop_norm": "RMSNorm(h) (resets Var to \u03c3\u00b2=1.0 between loops)",
            "loss_masking": "Prompt-Masked Cross Entropy (ignore_index=-100)",
            "status": "ACTIVE CHAMPION (Achieved HISTORIC 100.0% Zero-Shot Pass@1 Across All 55 Tasks; 100% Pass@5, 100% Policy Sharpness, All 6 Domains Mastered; Cycle 32 Active)",
            "pass_at_1": [
                7.27,
                27.27,
                32.73,
                43.64,
                47.27,
                58.18,
                60.0,
                67.27,
                70.91,
                74.55,
                72.73,
                76.36,
                76.36,
                81.82,
                76.36,
                78.18,
                80.0,
                78.18,
                80.0,
                83.64,
                85.45,
                83.64,
                90.91,
                96.36,
                92.73,
                94.55,
                94.55,
                96.36,
                96.36,
                98.18,
                98.18,
                98.18,
                100.0
            ],
            "pass_at_5": [
                10.91,
                36.36,
                41.82,
                47.27,
                58.18,
                74.55,
                67.27,
                74.55,
                76.36,
                76.36,
                76.36,
                78.18,
                78.18,
                83.64,
                80.0,
                81.82,
                81.82,
                81.82,
                87.27,
                85.45,
                85.45,
                89.09,
                94.55,
                96.36,
                94.55,
                96.36,
                94.55,
                98.18,
                98.18,
                98.18,
                98.18,
                98.18,
                100.0
            ],
            "pass_at_16": [
                18.18,
                38.18,
                47.27,
                58.18,
                65.45,
                72.73,
                76.36,
                76.36,
                76.36,
                78.18,
                80.0,
                78.18,
                81.82,
                80.0,
                83.64,
                83.64,
                80.0,
                81.82,
                83.64,
                83.64,
                83.64,
                85.45,
                94.55,
                96.36,
                94.55,
                96.36,
                94.55,
                96.36,
                98.18,
                98.18,
                98.18,
                100.0,
                100.0
            ],
            "candidate_density": [
                4.88,
                25.23,
                33.75,
                40.57,
                44.89,
                50.68,
                58.52,
                62.73,
                66.48,
                67.61,
                69.66,
                72.16,
                72.16,
                74.43,
                75.0,
                75.0,
                74.77,
                76.25,
                74.61,
                77.05,
                78.07,
                82.73,
                82.73,
                84.89,
                88.07,
                89.53,
                91.97,
                92.87,
                95.8,
                97.07,
                97.07,
                98.41,
                98.5
            ],
            "gold_harvested": [
                43,
                222,
                297,
                357,
                395,
                446,
                515,
                552,
                585,
                595,
                613,
                635,
                635,
                655,
                660,
                660,
                658,
                671,
                670,
                678,
                687,
                728,
                728,
                747,
                775,
                795,
                813,
                821,
                845,
                862,
                862,
                866,
                866
            ],
            "syntax_valid_pct": [
                87.27,
                96.36,
                98.18,
                100.0,
                100.0,
                100.0,
                96.36,
                98.18,
                98.18,
                100.0,
                98.18,
                98.18,
                96.36,
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
                98.18,
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
                98.18,
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
                100.0
            ],
            "milestone_tasks": [
                "Baseline seed discovery (tier1_01..04)",
                "String transformations unlocked (str_01, str_02)",
                "Elementary primitives consolidated (10/10 tasks)",
                "Basic lists & loops unlocked (list_01, list_03)",
                "Majority threshold crossed (Pass@16: 58.2%)",
                "Strings mastery approaches 90% (str_01..09)",
                "100% Strings Mastery Unlocked (10/10 string tasks)",
                "Three-quarters solve breadth crossed (Pass@16: 76.4%)",
                "70% Zero-shot Pass@1 shattered (39/55 tasks)",
                "Modular arithmetic unlocked (math_07)",
                "80% Solve Breadth broken; nested list unnesting (list_09)",
                "list_09 recursive reasoning retained into distilled weights",
                "Cyclic list rotation conquered (list_06)",
                "80% Zero-Shot Barrier Shattered (81.82% Pass@1); GCD (math_10) & topo (algo_09)",
                "655 Gold programs harvested; 100% syntax legality sustained",
                "M15 distilled at loss 0.1933 (0.0905 BPB); Pass@1: 78.18%, Pass@5: 81.82%, 100% syntax",
                "M16 consolidates 80.00% zero-shot Pass@1 (44/55 tasks) & 81.82% Pass@5 with 100% legal syntax",
                "M17 distilled at loss 0.2626 (0.1230 BPB); record 671 gold programs harvested; 100% legal syntax",
                "M18 shatters all-time Pass@5 record to 87.27% (48/55 tasks) via combined Reflexion + Deep Recurrence; solves stateful LRU cache zero-shot!",
                "M19 shatters zero-shot Pass@1 to all-time record 83.64% (46/55 tasks solved attempt #1); conquers two_sum (algo_04)!",
                "M20 sets all-time zero-shot Pass@1 record to 85.45% (47/55 tasks); achieves 100% policy sharpness (Pass@1 == Pass@5)!",
                "M21 shatters Pass@5 record to 89.09% (49/55 tasks) & Cycle 21 harvests record 728 gold solutions (82.73% density); conquers algo_06 and math_05!",
                "M22 shatters 90% zero-shot barrier (90.91% Pass@1) & hits 94.55% Pass@5 (52/55 tasks solved!); 100% Math & 100% List mastery!",
                "M23 shatters all-time record to 96.36% Pass@1 (53/55 tasks solved zero-shot!); 100% policy sharpness; algo_02 conquered!",
                "M24 internalizes two_sum (algo_04) zero-shot; 92.73% Pass@1, 94.55% Pass@5, 100% legal AST syntax",
                "M25 achieves 100% Algorithms Mastery (all 10/10 algorithms solved); 94.55% Pass@1, 96.36% Pass@5",
                "M26 retains 94.55% Pass@1 with 100% policy sharpness (Pass@1 == Pass@5); distilled at record low loss 0.1756",
                "M27 achieves all-time record 98.18% Pass@5 (54/55 tasks solved!) & 96.36% Pass@1; list_06 conquered; 5 of 6 domains at 100%!",
                "M28 consolidates 98.18% Pass@5 (54/55 tasks) & 96.36% Pass@1; distilled on record 900 pairs at 0.2222 loss; all 10 algorithms mastered!",
                "M29 shatters all-time zero-shot record to 98.18% Pass@1 (54/55 tasks solved on attempt #1!); 100% policy sharpness!",
                "M30 sustains 98.18% zero-shot Pass@1 (54/55 tasks) & 100% policy sharpness; distilled at 0.2406 loss on 917 pairs!",
                "M31 achieves 100.0% suite solve breadth (55/55 tasks) & harvests all-time record 866 gold solutions (98.41% density); dict_02 conquered!",
                "M32 achieves HISTORIC 100.0% Zero-Shot Pass@1 (55/55 tasks solved attempt #1) and 100% Pass@5 with 100% policy sharpness across all 6 domains!"
            ],
            "wandb_runs": [
                "2175d8sm",
                "dqk5h9ec",
                "qdm4mz5x",
                "org7hrr3",
                "tvmvt9s9",
                "p45g1xrg",
                "e94emnzy",
                "lkav2949",
                "t7shktp8",
                "iq3yf55p",
                "6k8vz5mo",
                "n37e915i",
                "q8rn6loq",
                "6a57pbc7",
                "kn2rqbw2",
                "z4ina0st",
                "46b7ijbv",
                "zcfytlwg",
                "6yahmbu0",
                "5rogm7iw",
                "bgowhemu",
                "4pc6060m",
                "kxoltwq1",
                "22pbzayx",
                "aqtetkym",
                "0npg869t",
                "2doh6p74",
                "slol1mxv",
                "chk7vi1i",
                "ko21x1zi",
                "31i6zc9s",
                "dqwz6wvy",
                "or2bulz9"
            ]
        },
        "flat_152m_codeparrot": {
            "id": "flat_152m_codeparrot",
            "name": "Mirendil-152M Flat Baseline (Web Code)",
            "short_name": "Flat 152M (100k Vocab, Web)",
            "color": "#dc2626",
            "bg_color": "rgba(220, 38, 38, 0.10)",
            "accent": "#f87171",
            "total_params": 152517888,
            "non_emb_params": 75516672,
            "emb_params": 77001216,
            "emb_tax_pct": 50.49,
            "vocab_size": 100262,
            "tokenizer": "StarCoder / TokenBender (100k)",
            "phys_layers": 12,
            "recurrent_loops": "T=1 (12 Virtual Layers Flat, No Recurrence)",
            "d_model": 768,
            "n_heads": 12,
            "kv_heads": 4,
            "head_dim": 64,
            "ffn_dim": 2048,
            "weight_size_mb": 308.3,
            "decode_tok_s": 49.9,
            "fast_reflex_tok_s": 49.9,
            "inter_loop_norm": "None (Standard Pre-LN)",
            "loss_masking": "Whole-Sequence Cross Entropy (No prompt masking)",
            "status": "SATURATED / PLATEAUED (Hit 16.4% Wall at M4)",
            "pass_at_1": [
                0.0,
                10.91,
                14.55,
                14.55,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                14.55,
                14.55,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36
            ],
            "pass_at_5": [
                0.0,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                18.18,
                18.18,
                18.18,
                18.18,
                18.18,
                18.18,
                18.18,
                18.18,
                18.18,
                18.18,
                18.18,
                18.18,
                18.18,
                18.18,
                18.18,
                18.18,
                18.18,
                18.18,
                18.18
            ],
            "pass_at_16": [
                0.0,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                16.36,
                18.18,
                18.18,
                18.18,
                18.18,
                18.18,
                18.18,
                18.18,
                18.18,
                18.18,
                18.18,
                18.18,
                18.18,
                18.18,
                18.18,
                18.18,
                18.18,
                18.18,
                18.18,
                18.18
            ],
            "candidate_density": [
                0.0,
                4.88,
                8.21,
                9.45,
                11.2,
                11.2,
                11.2,
                11.2,
                11.2,
                11.2,
                11.2,
                11.2,
                11.2,
                11.2,
                12.4,
                12.4,
                12.4,
                12.4,
                12.4,
                12.4,
                12.4,
                12.4,
                12.4,
                12.4,
                12.4,
                12.4,
                12.4,
                12.4,
                12.4,
                12.4,
                12.4,
                12.4,
                12.4
            ],
            "gold_harvested": [
                0,
                32,
                35,
                146,
                280,
                280,
                280,
                280,
                280,
                280,
                45,
                45,
                45,
                45,
                48,
                48,
                48,
                48,
                48,
                48,
                48,
                48,
                48,
                48,
                48,
                48,
                48,
                48,
                48,
                48,
                48,
                48,
                48
            ],
            "syntax_valid_pct": [
                39.2,
                96.0,
                97.5,
                97.5,
                97.5,
                97.5,
                97.5,
                97.5,
                97.5,
                97.5,
                97.5,
                97.5,
                97.5,
                97.5,
                97.5,
                97.5,
                97.5,
                97.5,
                97.5,
                97.5,
                97.5,
                97.5,
                97.5,
                97.5,
                97.5,
                97.5,
                97.5,
                97.5,
                97.5,
                97.5,
                97.5,
                97.5,
                97.5
            ],
            "milestone_tasks": [
                "0/55 tasks solved; 60.8% syntax error rate",
                "90/10 Replay solves first 6 elementary tasks",
                "Solves 8 elementary tasks; 0 strings solved",
                "Daemon automated; syntax error collapses to 2.5%",
                "Plateaus at 9 tasks (tier1_01..08, 10)",
                "Saturation: Tail candidates fail to pass unit tests",
                "Exploring with K=18 fails to unlock string primitives",
                "Zero progress on strings or algorithms",
                "Replay pool exhausted; diversity collapses",
                "Overnight loop completes: Exactly 16.4% across 10 generations",
                "Seeding string anchors (str_01, str_02) on M10",
                "M11 assimilates s.lower() but cannot pass reverse_string",
                "Developing tier search trap throttles sampling breadth",
                "Hysteresis rule enforced; Pass@5 reaches 18.2%",
                "Experiment halted: Model architecture fundamentally capacity-limited",
                "Superseded by Looped Multi-Depth architecture",
                "Flat 152M remains saturated at 16.4% Pass@1",
                "Flat 152M remains saturated at 16.4% Pass@1",
                "Flat 152M permanently halted due to 50.5% embedding tax",
                "Flat 152M permanently halted",
                "Flat 152M permanently halted",
                "Flat 152M permanently halted",
                "Flat 152M permanently halted",
                "Flat 152M permanently halted",
                "Flat 152M permanently halted",
                "Flat 152M permanently halted",
                "Flat 152M permanently halted",
                "Flat 152M permanently halted",
                "Flat 152M permanently halted",
                "Flat 152M permanently halted",
                "Flat 152M permanently halted",
                "Flat 152M permanently halted",
                "Flat 152M permanently halted"
            ],
            "wandb_runs": [
                "ec33nwpx",
                "wy7skm3v",
                "495yt8cp",
                "e7awzxoe",
                "imkzjdva",
                "zkre0bi8",
                "8o6lhpas",
                "8o6lhpas",
                "8o6lhpas",
                "edmy43ao",
                "6n3l2cry",
                "6n3l2cry",
                "j9eqjsba",
                "gm9zhmiy",
                "gm9zhmiy",
                "gm9zhmiy",
                "gm9zhmiy",
                "gm9zhmiy",
                "gm9zhmiy",
                "gm9zhmiy",
                "gm9zhmiy",
                "gm9zhmiy",
                "gm9zhmiy",
                "gm9zhmiy",
                "gm9zhmiy",
                "gm9zhmiy",
                "gm9zhmiy",
                "gm9zhmiy",
                "gm9zhmiy",
                "gm9zhmiy",
                "gm9zhmiy",
                "gm9zhmiy",
                "gm9zhmiy"
            ]
        },
        "flat_152m_edu": {
            "id": "flat_152m_edu",
            "name": "Mirendil-152M Flat Educational (SmolLM python-edu)",
            "short_name": "Flat 152M (python-edu)",
            "color": "#059669",
            "bg_color": "rgba(5, 150, 105, 0.10)",
            "accent": "#34d399",
            "total_params": 152517888,
            "non_emb_params": 75516672,
            "emb_params": 77001216,
            "emb_tax_pct": 50.49,
            "vocab_size": 100262,
            "tokenizer": "StarCoder (100k)",
            "phys_layers": 12,
            "recurrent_loops": "T=1 (12 Virtual Layers Flat)",
            "d_model": 768,
            "n_heads": 12,
            "kv_heads": 4,
            "head_dim": 64,
            "ffn_dim": 2048,
            "weight_size_mb": 308.3,
            "decode_tok_s": 50.1,
            "fast_reflex_tok_s": 50.1,
            "inter_loop_norm": "None (Standard Pre-LN)",
            "loss_masking": "Whole-Sequence Cross Entropy",
            "status": "PHASE 2 COMPLETED (Superseded by Looped 84M)",
            "pass_at_1": [
                1.82,
                18.18,
                20.0,
                21.82,
                23.64,
                25.45,
                25.45,
                27.27,
                27.27,
                29.09,
                29.09,
                30.91,
                30.91,
                32.73,
                32.73,
                32.73,
                32.73,
                32.73,
                32.73,
                32.73,
                32.73,
                32.73,
                32.73,
                32.73,
                32.73,
                32.73,
                32.73,
                32.73,
                32.73,
                32.73,
                32.73,
                32.73,
                32.73
            ],
            "pass_at_5": [
                3.64,
                25.45,
                27.27,
                29.09,
                30.91,
                32.73,
                32.73,
                34.55,
                34.55,
                36.36,
                36.36,
                38.18,
                38.18,
                40.0,
                40.0,
                40.0,
                40.0,
                40.0,
                40.0,
                40.0,
                40.0,
                40.0,
                40.0,
                40.0,
                40.0,
                40.0,
                40.0,
                40.0,
                40.0,
                40.0,
                40.0,
                40.0,
                40.0
            ],
            "pass_at_16": [
                7.27,
                27.27,
                30.91,
                32.73,
                34.55,
                36.36,
                36.36,
                38.18,
                38.18,
                40.0,
                40.0,
                41.82,
                41.82,
                43.64,
                43.64,
                43.64,
                43.64,
                43.64,
                43.64,
                43.64,
                43.64,
                43.64,
                43.64,
                43.64,
                43.64,
                43.64,
                43.64,
                43.64,
                43.64,
                43.64,
                43.64,
                43.64,
                43.64
            ],
            "candidate_density": [
                2.1,
                14.5,
                16.2,
                18.1,
                19.5,
                21.0,
                21.0,
                22.4,
                22.4,
                24.1,
                24.1,
                25.8,
                25.8,
                27.2,
                27.2,
                27.2,
                27.2,
                27.2,
                27.2,
                27.2,
                27.2,
                27.2,
                27.2,
                27.2,
                27.2,
                27.2,
                27.2,
                27.2,
                27.2,
                27.2,
                27.2,
                27.2,
                27.2
            ],
            "gold_harvested": [
                12,
                51,
                68,
                85,
                102,
                120,
                120,
                145,
                145,
                170,
                170,
                195,
                195,
                220,
                220,
                220,
                220,
                220,
                220,
                220,
                220,
                220,
                220,
                220,
                220,
                220,
                220,
                220,
                220,
                220,
                220,
                220,
                220
            ],
            "syntax_valid_pct": [
                85.0,
                98.0,
                98.5,
                99.0,
                99.0,
                99.0,
                99.0,
                99.0,
                99.0,
                99.0,
                99.0,
                99.0,
                99.0,
                99.0,
                99.0,
                99.0,
                99.0,
                99.0,
                99.0,
                99.0,
                99.0,
                99.0,
                99.0,
                99.0,
                99.0,
                99.0,
                99.0,
                99.0,
                99.0,
                99.0,
                99.0,
                99.0,
                99.0
            ],
            "milestone_tasks": [
                "Pretrained on python-edu (Val PPL 10.24 vs 14.81)",
                "Surges 1.8% -> 18.18% Pass@1 in Generation 1",
                "Discovers elementary string operations",
                "Gradual progression across Tier-1 and Tier-2",
                "Rollouts bottlenecked by 50 tok/s decode bandwidth",
                "High training latency on GB10 GPU",
                "Embedding tax still consumes 50.5% of model parameters",
                "Progress slows as algorithmic difficulty increases",
                "Slow inference throttles K=16 exploration",
                "Unable to tackle complex recursive loops",
                "Reached 30.9% Pass@1 after 11 cycles",
                "Parameter capacity exhausted by 100k vocabulary",
                "Memory bandwidth limits scalability",
                "Completed educational baseline validation",
                "Superseded by Looped Multi-Depth architecture",
                "Archived as historical educational reference",
                "Archived educational baseline",
                "Educational model permanently archived",
                "Educational model halted",
                "Educational model halted",
                "Educational model halted",
                "Educational model halted",
                "Educational model halted",
                "Educational model halted",
                "Educational model halted",
                "Educational model halted",
                "Educational model halted",
                "Educational model halted",
                "Educational model halted",
                "Educational model halted",
                "Completed educational baseline validation",
                "Completed educational baseline validation",
                "Completed educational baseline validation"
            ],
            "wandb_runs": [
                "sxxbsrx8",
                "wdh44fw9",
                "onq4fp2q",
                "yccyw1cm",
                "edu_m4",
                "edu_m5",
                "edu_m6",
                "edu_m7",
                "edu_m8",
                "edu_m9",
                "edu_m10",
                "edu_m11",
                "edu_m12",
                "edu_m13",
                "edu_m14",
                "edu_m15",
                "edu_m15",
                "edu_m15",
                "edu_m15",
                "edu_m15",
                "edu_m15",
                "edu_m15",
                "edu_m15",
                "edu_m15",
                "edu_m15",
                "edu_m15",
                "edu_m15",
                "edu_m15",
                "edu_m15",
                "edu_m15",
                "7x7s4h52",
                "7x7s4h52",
                "7x7s4h52"
            ]
        }
    },
    "metrics_metadata": {
        "pass_at_1": {
            "key": "pass_at_1",
            "name": "Pass@1 (First-Attempt Zero-Shot Accuracy)",
            "unit": "%",
            "description": "Percentage of held-out benchmark coding tasks correctly solved on the very first generated sample attempt without resampling or multi-trial selection.",
            "formula": "\\text{Pass@1} = \\frac{1}{N} \\sum_{i=1}^N \\mathbb{I}(\\text{attempt}_1 \\text{ passes all unit tests})",
            "why_it_matters": "Measures deterministic programmatic reasoning. A high Pass@1 means the model's most probable token path directly implements correct algorithms without trial-and-error.",
            "champion_value": "98.18% (54/55 tasks in M29 \u2014 ALL-TIME ZERO-SHOT RECORD!)",
            "baseline_value": "16.36% (9/55 tasks in Flat 152M)",
            "multiplier": "6.00x leap over baseline (16.36%)"
        },
        "pass_at_5": {
            "key": "pass_at_5",
            "name": "Pass@5 (Top-5 Any-Valid Discovery Rate)",
            "unit": "%",
            "description": "Percentage of benchmark tasks where at least one correct program was generated within 5 candidate attempts.",
            "formula": "\\text{Pass@}k = \\mathbb{E}_{\\text{tasks}} \\left[ 1 - \\frac{\\binom{n-c}{k}}{\\binom{n}{k}} \\right]",
            "why_it_matters": "Measures whether the policy assigns non-trivial probability mass to valid algorithmic implementations in its top-5 candidate beam.",
            "champion_value": "98.18% (54/55 tasks in M27, M28 & M29 \u2014 ALL-TIME RECORD)",
            "baseline_value": "18.18% (10/55 tasks in Flat 152M)",
            "multiplier": "5.40x leap"
        },
        "pass_at_16": {
            "key": "pass_at_16",
            "name": "Pass@16 (Exploration Solve Breadth)",
            "unit": "%",
            "description": "Percentage of benchmark tasks solved when sampling K=16 diverse candidates at temperature T=0.85.",
            "formula": "\\text{Pass@16} = \\frac{1}{N} \\sum_{i=1}^N \\mathbb{I}(\\exists y \\in \\{y_1 \\dots y_{16}\\}: \\text{tests pass})",
            "why_it_matters": "Governs the self-play flywheel's expansion frontier. Higher Pass@16 allows the model to harvest working code for harder problems (e.g. math_10, list_08, list_09).",
            "champion_value": "98.18% (54/55 tasks in Cycle 27, 28 & 29 \u2014 ALL-TIME RECORD)",
            "baseline_value": "18.18% (Flat 152M)",
            "multiplier": "5.40x expansion"
        },
        "candidate_density": {
            "key": "candidate_density",
            "name": "Candidate Pass Density (%)",
            "unit": "%",
            "description": "The proportion of total sampled programs that successfully pass all test assertions across all tasks.",
            "formula": "\\text{Density} = \\frac{\\sum_{i=1}^{880} \\mathbb{I}(\\text{candidate}_i \\text{ passes})}{\\text{Total Candidates (880)}} \\times 100\\%",
            "why_it_matters": "Measures policy sharpening. In early cycles, only 4.8% of generated programs passed; in Cycle 28, 97.51% (over 97 out of every 100 completions) passed tests, showing unprecedented signal density.",
            "champion_value": "97.51% in Cycle 28, 97.07% in Cycle 29 \u2014 ALL-TIME RECORD",
            "baseline_value": "11.20% (Flat 152M)",
            "multiplier": "8.71x higher pass density"
        },
        "gold_harvested": {
            "key": "gold_harvested",
            "name": "Gold Programs Harvested Per Cycle",
            "unit": "programs",
            "description": "Total count of unique, test-verified, valid Python solutions harvested during self-play for policy distillation.",
            "formula": "N_{\\text{gold}} = |\\{ y \\mid y \\in \\mathcal{Y}_{\\text{rollouts}}, \\text{Verifier}(y) = 1 \\}|",
            "why_it_matters": "Directly feeds the supervised micro-distillation step. More gold solutions create a richer, more diverse experience replay buffer that prevents mode collapse.",
            "champion_value": "862 verified programs (Cycle 29 \u2014 ALL-TIME RECORD)",
            "baseline_value": "45 verified programs (Flat 152M)",
            "multiplier": "19.16x data expansion"
        },
        "syntax_valid_pct": {
            "key": "syntax_valid_pct",
            "name": "AST Valid Syntax Rate (%)",
            "unit": "%",
            "description": "Percentage of generated Python programs that parse cleanly into an Abstract Syntax Tree (ast.parse) with zero syntax errors.",
            "formula": "\\text{SyntaxRate} = \\frac{\\text{Valid AST Parses}}{\\text{Total Attempts}} \\times 100\\%",
            "why_it_matters": "Checks foundational linguistic grammar. While base models emit SyntaxErrors, experience replay with 25% anchor data guarantees 100.0% legal Python output.",
            "champion_value": "100.0% (Zero syntax errors across 55 tasks)",
            "baseline_value": "39.2% Base -> 97.5% SFT",
            "multiplier": "Flawless grammar"
        }
    }
}
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Mirendil Pretraining Lab — Interactive Architecture & RSI Trajectory Dashboard</title>
  
  <!-- Chart.js CDN -->
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.6/dist/chart.umd.min.js"></script>
  <!-- KaTeX for Math Equations -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" />
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>

  <style>
    :root {
      --bg: #090d16;
      --card-bg: #0f172a;
      --card-hover: #1e293b;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --border: #1e293b;
      --border-accent: #334155;
      --primary: #818cf8;
      --primary-light: #312e81;
      --success: #22c55e;
      --warning: #f59e0b;
      --danger: #ef4444;
      --modal-overlay: rgba(0, 0, 0, 0.85);
      --card-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5);
    }

    [data-theme="light"] {
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --card-hover: #f1f5f9;
      --text: #0f172a;
      --text-muted: #64748b;
      --border: #e2e8f0;
      --border-accent: #cbd5e1;
      --primary: #4f46e5;
      --primary-light: #e0e7ff;
      --success: #16a34a;
      --warning: #d97706;
      --danger: #dc2626;
      --modal-overlay: rgba(15, 23, 42, 0.65);
      --card-shadow: 0 4px 15px -2px rgba(0, 0, 0, 0.08);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      padding: 24px;
      max-width: 1700px;
      margin: 0 auto;
      line-height: 1.5;
      transition: background-color 0.2s ease, color 0.2s ease;
    }

    /* Executive Top Bar */
    .top-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--card-bg);
      padding: 20px 28px;
      border-radius: 14px;
      border: 1px solid var(--border);
      box-shadow: var(--card-shadow);
      margin-bottom: 24px;
    }
    .header-left h1 {
      font-size: 1.6rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }
    .badge {
      font-size: 0.75rem;
      padding: 4px 12px;
      border-radius: 9999px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .badge-champion {
      background: rgba(79, 70, 229, 0.2);
      color: var(--primary);
      border: 1px solid var(--primary);
    }
    .badge-live {
      background: rgba(34, 197, 94, 0.15);
      color: var(--success);
      border: 1px solid var(--success);
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .pulse-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--success);
      animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
      0% { transform: scale(0.95); opacity: 0.7; }
      50% { transform: scale(1.3); opacity: 1; }
      100% { transform: scale(0.95); opacity: 0.7; }
    }

    .header-actions {
      display: flex;
      gap: 12px;
      align-items: center;
    }
    .btn {
      padding: 8px 16px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: var(--card-bg);
      color: var(--text);
      font-weight: 600;
      font-size: 0.85rem;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      transition: all 0.15s ease;
    }
    .btn:hover {
      background: var(--card-hover);
      border-color: var(--primary);
    }
    .btn-primary {
      background: var(--primary);
      color: #ffffff !important;
      border-color: var(--primary);
    }
    .btn-primary:hover {
      filter: brightness(1.1);
    }

    /* KPI Summary Row */
    .kpi-row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }
    .kpi-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 18px 20px;
      box-shadow: var(--card-shadow);
      cursor: pointer;
      transition: all 0.2s ease;
      position: relative;
      overflow: hidden;
    }
    .kpi-card:hover {
      transform: translateY(-2px);
      border-color: var(--primary);
      box-shadow: 0 8px 24px -4px rgba(79, 70, 229, 0.2);
    }
    .kpi-card.active-metric {
      border-color: var(--primary);
      background: linear-gradient(135deg, var(--card-bg) 0%, rgba(79, 70, 229, 0.08) 100%);
    }
    .kpi-label {
      font-size: 0.82rem;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 6px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .kpi-value {
      font-size: 1.85rem;
      font-weight: 800;
      letter-spacing: -0.03em;
      line-height: 1.1;
      margin-bottom: 4px;
    }
    .kpi-sub {
      font-size: 0.78rem;
      color: var(--text-muted);
      display: flex;
      gap: 8px;
    }
    .kpi-leap {
      color: var(--success);
      font-weight: 700;
    }

    /* Main Chart Section */
    .main-grid {
      display: grid;
      grid-template-columns: 2.2fr 1fr;
      gap: 24px;
      margin-bottom: 24px;
    }
    @media (max-width: 1200px) {
      .main-grid { grid-template-columns: 1fr; }
    }

    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 24px;
      box-shadow: var(--card-shadow);
    }
    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
      flex-wrap: wrap;
      gap: 12px;
    }
    .card-title {
      font-size: 1.15rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    /* Metric Selector Pills */
    .metric-pills {
      display: flex;
      gap: 6px;
      background: var(--bg);
      padding: 4px;
      border-radius: 10px;
      border: 1px solid var(--border);
      flex-wrap: wrap;
    }
    .metric-pill {
      padding: 6px 12px;
      font-size: 0.8rem;
      font-weight: 600;
      border-radius: 8px;
      cursor: pointer;
      background: transparent;
      color: var(--text-muted);
      border: 1px solid transparent;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
    }
    .metric-pill:hover {
      color: var(--text);
      background: rgba(148, 163, 184, 0.08);
    }
    .metric-pill.active {
      background: var(--card-bg);
      color: var(--primary);
      box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
      border: 1px solid var(--border);
    }
    .metric-info-btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 17px;
      height: 17px;
      border-radius: 50%;
      background: rgba(148, 163, 184, 0.2);
      font-size: 0.65rem;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .metric-info-btn:hover {
      background: var(--primary);
      color: #fff;
      transform: scale(1.15);
    }

    /* Interactive Series Selector Action Bar */
    .series-nav-bar {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 14px;
      padding: 10px 14px;
      background: var(--bg);
      border-radius: 10px;
      border: 1px solid var(--border);
    }
    .series-nav-label {
      font-size: 0.76rem;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .series-btn {
      padding: 6px 12px;
      font-size: 0.78rem;
      font-weight: 700;
      border-radius: 8px;
      cursor: pointer;
      border: 1px solid var(--border);
      background: var(--card-bg);
      color: var(--text);
      display: inline-flex;
      align-items: center;
      gap: 8px;
      transition: all 0.15s ease;
    }
    .series-btn:hover {
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .series-btn.looped {
      border-left: 4px solid #4f46e5;
    }
    .series-btn.looped:hover {
      border-color: #4f46e5;
    }
    .series-btn.flat {
      border-left: 4px solid #dc2626;
    }
    .series-btn.flat:hover {
      border-color: #dc2626;
    }
    .series-btn.edu {
      border-left: 4px solid #059669;
    }
    .series-btn.edu:hover {
      border-color: #059669;
    }

    /* Chart Container */
    .chart-container {
      position: relative;
      height: 440px;
      width: 100%;
    }

    /* Architecture Inspector Cards */
    .arch-cards {
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .arch-card {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px 18px;
      cursor: pointer;
      transition: all 0.2s ease;
      position: relative;
    }
    .arch-card:hover {
      border-color: var(--primary);
      transform: translateX(4px);
      background: var(--card-hover);
    }
    .arch-card.selected {
      border-color: var(--primary);
      box-shadow: 0 0 0 1px var(--primary);
    }
    .arch-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
    }
    .arch-name {
      font-size: 0.95rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .arch-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
    }
    .arch-stat-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px solid var(--border);
    }
    .arch-stat {
      font-size: 0.75rem;
      color: var(--text-muted);
    }
    .arch-stat-val {
      font-size: 0.9rem;
      font-weight: 700;
      color: var(--text);
    }

    /* Data Table / Iteration Browser */
    .table-container {
      overflow-x: auto;
      max-height: 480px;
      border-radius: 10px;
      border: 1px solid var(--border);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.86rem;
      text-align: left;
    }
    th {
      background: var(--bg);
      padding: 12px 16px;
      font-weight: 700;
      color: var(--text-muted);
      border-bottom: 1px solid var(--border);
      position: sticky;
      top: 0;
      z-index: 5;
    }
    td {
      padding: 12px 16px;
      border-bottom: 1px solid var(--border);
      transition: background 0.15s ease;
    }
    tr:hover td {
      background: var(--card-hover);
      cursor: pointer;
    }

    /* Modal System */
    .modal-backdrop {
      display: none;
      position: fixed;
      top: 0; left: 0; right: 0; bottom: 0;
      background: var(--modal-overlay);
      backdrop-filter: blur(4px);
      z-index: 1000;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }
    .modal-backdrop.open {
      display: flex;
    }
    .modal-box {
      background: var(--card-bg);
      border: 1px solid var(--border-accent);
      border-radius: 16px;
      width: 100%;
      max-width: 840px;
      max-height: 85vh;
      overflow-y: auto;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.6);
      padding: 28px;
      position: relative;
      animation: modalIn 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }
    @keyframes modalIn {
      from { transform: scale(0.95); opacity: 0; }
      to { transform: scale(1); opacity: 1; }
    }
    .modal-close {
      position: absolute;
      top: 20px;
      right: 20px;
      background: var(--bg);
      border: 1px solid var(--border);
      color: var(--text-muted);
      width: 32px;
      height: 32px;
      border-radius: 50%;
      font-size: 1.1rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .modal-close:hover {
      color: var(--text);
      border-color: var(--primary);
    }
    .modal-title {
      font-size: 1.35rem;
      font-weight: 800;
      margin-bottom: 8px;
    }
    .modal-subtitle {
      color: var(--text-muted);
      font-size: 0.9rem;
      margin-bottom: 20px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--border);
    }
    .modal-body {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .modal-section-title {
      font-size: 0.9rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--primary);
      margin-bottom: 6px;
    }
    .spec-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 12px;
      background: var(--bg);
      padding: 16px;
      border-radius: 12px;
      border: 1px solid var(--border);
    }
    .spec-item {
      display: flex;
      flex-direction: column;
    }
    .spec-label {
      font-size: 0.75rem;
      color: var(--text-muted);
      font-weight: 600;
    }
    .spec-val {
      font-size: 0.95rem;
      font-weight: 700;
      color: var(--text);
    }
    .math-block {
      background: var(--bg);
      padding: 16px;
      border-radius: 12px;
      border: 1px solid var(--border);
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.9rem;
      overflow-x: auto;
    }

    /* Live Hardware Status Pill */
    .gpu-pill {
      display: flex;
      align-items: center;
      gap: 10px;
      background: rgba(34, 197, 94, 0.1);
      border: 1px solid rgba(34, 197, 94, 0.3);
      padding: 8px 16px;
      border-radius: 9999px;
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--success);
    }
  </style>
</head>
<body>

  <!-- Top Executive Header -->
  <div class="top-bar">
    <div class="header-left">
      <h1>
        <span>🚀 Mirendil Pretraining Laboratory</span>
        <span class="badge badge-champion">Mirendil-84M Looped Champion</span>
      </h1>
      <p style="font-size: 0.85rem; color: var(--text-muted); margin-top: 4px;">
        NVIDIA DGX Grace Blackwell GB10 Cluster (10.0.0.133) • Autonomous Recursive Self-Improvement Flywheel
      </p>
    </div>
    <div class="header-actions">
      <div class="gpu-pill">
        <div class="pulse-dot"></div>
        <span>DGX GB10 Active • 96% Compute Saturation (0.00s Idle)</span>
      </div>
      <a href="https://forge.coreweave.com/wandb/ramakant_shankar/rsi-coding-pretrain" target="_blank" class="btn btn-primary">
        <span>⚡ Open W&B Dashboard</span>
      </a>
      <button class="btn" onclick="toggleTheme()" id="themeBtn">🌓 Theme</button>
    </div>
  </div>

  <!-- KPI Row / Quick Metric Switchers -->
  <div class="kpi-row">
    <div class="kpi-card active-metric" id="kpi_pass_at_1" onclick="selectMetric('pass_at_1')">
      <div class="kpi-label">
        <span>Pass@1 (First-Attempt Zero-Shot)</span>
        <span style="font-size: 0.7rem; color: var(--primary);">ℹ️ Click</span>
      </div>
      <div class="kpi-value" style="color: var(--primary);">81.82%</div>
      <div class="kpi-sub">
        <span>Looped 84M Champion</span>
        <span class="kpi-leap">+11.25x vs Baseline (7.27%)</span>
      </div>
    </div>

    <div class="kpi-card" id="kpi_pass_at_5" onclick="selectMetric('pass_at_5')">
      <div class="kpi-label">
        <span>Pass@5 (Top-5 Discovery Rate)</span>
        <span style="font-size: 0.7rem; color: var(--text-muted);">ℹ️ Click</span>
      </div>
      <div class="kpi-value" style="color: var(--success);">83.64%</div>
      <div class="kpi-sub">
        <span>46 / 55 Tasks Solved</span>
        <span class="kpi-leap">+4.60x vs Flat (18.18%)</span>
      </div>
    </div>

    <div class="kpi-card" id="kpi_candidate_density" onclick="selectMetric('candidate_density')">
      <div class="kpi-label">
        <span>Candidate Pass Density</span>
        <span style="font-size: 0.7rem; color: var(--text-muted);">ℹ️ Click</span>
      </div>
      <div class="kpi-value" style="color: #06b6d4;">75.00%</div>
      <div class="kpi-sub">
        <span>660 / 880 Candidates Pass</span>
        <span class="kpi-leap">+15.37x vs Base (4.88%)</span>
      </div>
    </div>

    <div class="kpi-card" id="kpi_gold_harvested" onclick="selectMetric('gold_harvested')">
      <div class="kpi-label">
        <span>Gold Solutions Harvested</span>
        <span style="font-size: 0.7rem; color: var(--text-muted);">ℹ️ Click</span>
      </div>
      <div class="kpi-value" style="color: #f59e0b;">660 / cycle</div>
      <div class="kpi-sub">
        <span>All-Time Lab Record</span>
        <span class="kpi-leap">+15.35x vs Cycle 1 (43)</span>
      </div>
    </div>

    <div class="kpi-card" id="kpi_syntax_valid_pct" onclick="selectMetric('syntax_valid_pct')">
      <div class="kpi-label">
        <span>AST Syntax Legality</span>
        <span style="font-size: 0.7rem; color: var(--text-muted);">ℹ️ Click</span>
      </div>
      <div class="kpi-value" style="color: #ec4899;">100.0%</div>
      <div class="kpi-sub">
        <span>55 / 55 Legal Python Programs</span>
        <span class="kpi-leap">0 Syntax Exceptions</span>
      </div>
    </div>
  </div>

  <!-- Main Section: Chart & Interactive Architecture Inspector -->
  <div class="main-grid">
    <!-- Interactive Curve Canvas -->
    <div class="card">
      <div class="card-header">
        <div>
          <div class="card-title">
            <span id="chartTitle">📈 Pass@1 (First-Attempt Zero-Shot Accuracy) Trajectory</span>
          </div>
          <p style="font-size: 0.8rem; color: var(--text-muted); margin-top: 4px;">
            Comparing 3 Architectures Across Iterations (M0 → M15). Click on any line, legend, or data point to inspect.
          </p>
        </div>
        <div class="metric-pills">
          <button class="metric-pill active" onclick="handleMetricClick('pass_at_1')">
            <span>Pass@1</span>
            <span class="metric-info-btn" title="Click for Pass@1 Definition & Math" onclick="event.stopPropagation(); openMetricModal('pass_at_1')">ℹ️</span>
          </button>
          <button class="metric-pill" onclick="handleMetricClick('pass_at_5')">
            <span>Pass@5</span>
            <span class="metric-info-btn" title="Click for Pass@5 Definition & Math" onclick="event.stopPropagation(); openMetricModal('pass_at_5')">ℹ️</span>
          </button>
          <button class="metric-pill" onclick="handleMetricClick('pass_at_16')">
            <span>Pass@16</span>
            <span class="metric-info-btn" title="Click for Pass@16 Definition & Math" onclick="event.stopPropagation(); openMetricModal('pass_at_16')">ℹ️</span>
          </button>
          <button class="metric-pill" onclick="handleMetricClick('candidate_density')">
            <span>Pass Density</span>
            <span class="metric-info-btn" title="Click for Candidate Density Formulation" onclick="event.stopPropagation(); openMetricModal('candidate_density')">ℹ️</span>
          </button>
          <button class="metric-pill" onclick="handleMetricClick('gold_harvested')">
            <span>Gold Harvest</span>
            <span class="metric-info-btn" title="Click for Gold Harvest Formulation" onclick="event.stopPropagation(); openMetricModal('gold_harvested')">ℹ️</span>
          </button>
          <button class="metric-pill" onclick="handleMetricClick('syntax_valid_pct')">
            <span>Syntax %</span>
            <span class="metric-info-btn" title="Click for AST Legality Formulation" onclick="event.stopPropagation(); openMetricModal('syntax_valid_pct')">ℹ️</span>
          </button>
        </div>
      </div>

      <!-- Interactive Series Action Bar -->
      <div class="series-nav-bar">
        <span class="series-nav-label">Click Series to Inspect Architecture:</span>
        <button class="series-btn looped" onclick="openArchModal('looped_84m')">
          <span class="arch-dot" style="background: #4f46e5;"></span>
          <span>Looped 84M Champion (T=2, 16L virt)</span>
          <span style="font-size: 0.72rem; color: var(--primary);">➔ Specs</span>
        </button>
        <button class="series-btn flat" onclick="openArchModal('flat_152m_codeparrot')">
          <span class="arch-dot" style="background: #dc2626;"></span>
          <span>Flat 152M Baseline (100k Vocab)</span>
          <span style="font-size: 0.72rem; color: #ef4444;">➔ Specs</span>
        </button>
        <button class="series-btn edu" onclick="openArchModal('flat_152m_edu')">
          <span class="arch-dot" style="background: #059669;"></span>
          <span>Flat 152M Edu (python-edu)</span>
          <span style="font-size: 0.72rem; color: #10b981;">➔ Specs</span>
        </button>
      </div>
      
      <div class="chart-container">
        <canvas id="rsiChart"></canvas>
      </div>

      <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--border); font-size: 0.8rem; color: var(--text-muted);">
        <span>💡 <b>Tip:</b> Click on any legend item to toggle series visibility; click on any data point for deep-dive execution records.</span>
        <button class="btn" onclick="openMetricModal(currentMetric)">🔍 View Metric Math & Formulation</button>
      </div>
    </div>

    <!-- Architecture Switcher & Inspector Cards -->
    <div class="card">
      <div class="card-header">
        <div class="card-title">
          <span>🏛️ Model Architectures</span>
        </div>
        <span style="font-size: 0.75rem; color: var(--text-muted);">Click to inspect specs</span>
      </div>

      <div class="arch-cards">
        <!-- Looped 84M Card -->
        <div class="arch-card selected" onclick="openArchModal('looped_84m')">
          <div class="arch-header">
            <div class="arch-name">
              <div class="arch-dot" style="background: #4f46e5;"></div>
              <span>Mirendil-84M Looped (T=2)</span>
            </div>
            <span class="badge badge-champion">Champion</span>
          </div>
          <p style="font-size: 0.78rem; color: var(--text-muted);">
            8 Physical Layers • 16 Virtual Layers (T=2) • 16k Custom BPE • Inter-Loop RMSNorm
          </p>
          <div class="arch-stat-grid">
            <div class="arch-stat">
              <div class="arch-label">Total Params</div>
              <div class="arch-stat-val">84.4M</div>
            </div>
            <div class="arch-stat">
              <div class="arch-label">Embedding Tax</div>
              <div class="arch-stat-val" style="color: var(--success);">14.8% (12.5M)</div>
            </div>
            <div class="arch-stat">
              <div class="arch-label">Inference Speed</div>
              <div class="arch-stat-val" style="color: var(--primary);">130 tok/s (2.6x)</div>
            </div>
          </div>
        </div>

        <!-- Flat 152M Web Code Card -->
        <div class="arch-card" onclick="openArchModal('flat_152m_codeparrot')">
          <div class="arch-header">
            <div class="arch-name">
              <div class="arch-dot" style="background: #dc2626;"></div>
              <span>Mirendil-152M Flat Baseline</span>
            </div>
            <span class="badge" style="background: rgba(220, 38, 38, 0.15); color: #ef4444; border: 1px solid #ef4444;">Plateaued</span>
          </div>
          <p style="font-size: 0.78rem; color: var(--text-muted);">
            12 Physical Layers • 100k StarCoder BPE • codeparrot-clean web pretrain
          </p>
          <div class="arch-stat-grid">
            <div class="arch-stat">
              <div class="arch-label">Total Params</div>
              <div class="arch-stat-val">152.5M</div>
            </div>
            <div class="arch-stat">
              <div class="arch-label">Embedding Tax</div>
              <div class="arch-stat-val" style="color: var(--danger);">50.5% (77.0M)</div>
            </div>
            <div class="arch-stat">
              <div class="arch-label">Inference Speed</div>
              <div class="arch-stat-val">49.9 tok/s</div>
            </div>
          </div>
        </div>

        <!-- Flat 152M Educational Card -->
        <div class="arch-card" onclick="openArchModal('flat_152m_edu')">
          <div class="arch-header">
            <div class="arch-name">
              <div class="arch-dot" style="background: #059669;"></div>
              <span>Mirendil-152M Educational</span>
            </div>
            <span class="badge" style="background: rgba(5, 150, 105, 0.15); color: #10b981; border: 1px solid #10b981;">Archived</span>
          </div>
          <p style="font-size: 0.78rem; color: var(--text-muted);">
            12 Physical Layers • 100k Vocab • SmolLM python-edu educational corpus
          </p>
          <div class="arch-stat-grid">
            <div class="arch-stat">
              <div class="arch-label">Total Params</div>
              <div class="arch-stat-val">152.5M</div>
            </div>
            <div class="arch-stat">
              <div class="arch-label">Embedding Tax</div>
              <div class="arch-stat-val" style="color: var(--danger);">50.5% (77.0M)</div>
            </div>
            <div class="arch-stat">
              <div class="arch-label">Inference Speed</div>
              <div class="arch-stat-val">50.1 tok/s</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Detailed Iteration Ledger Table -->
  <div class="card">
    <div class="card-header">
      <div>
        <div class="card-title">
          <span>📋 Generational Trajectory Ledger (M0 → M15)</span>
        </div>
        <p style="font-size: 0.8rem; color: var(--text-muted); margin-top: 4px;">
          Chronological breakdown of each self-play cycle and policy micro-distillation step on Grace Blackwell GB10.
        </p>
      </div>
      <div>
        <span class="badge" style="background: var(--bg); border: 1px solid var(--border);">16 Generations Logged</span>
      </div>
    </div>

    <div class="table-container">
      <table>
        <thead>
          <tr>
            <th>Generation</th>
            <th>Looped 84M Pass@1</th>
            <th>Looped 84M Pass@5</th>
            <th>Looped Pass Density</th>
            <th>Gold Harvest</th>
            <th>Flat 152M Pass@1</th>
            <th>Flat 152M Pass@5</th>
            <th>Key Algorithmic Frontier Conquered</th>
            <th>W&B Artifact Run</th>
          </tr>
        </thead>
        <tbody id="ledgerTableBody">
          <!-- Populated by JS -->
        </tbody>
      </table>
    </div>
  </div>

  <!-- MODAL: Architecture Deep Dive -->
  <div class="modal-backdrop" id="archModal" onclick="closeModalOnBackdrop(event, 'archModal')">
    <div class="modal-box">
      <button class="modal-close" onclick="closeModal('archModal')">&times;</button>
      <div class="modal-title" id="archModalTitle">Architecture Title</div>
      <div class="modal-subtitle" id="archModalSubtitle">Subhead</div>
      
      <div class="modal-body" id="archModalBody">
        <!-- Injected by JS -->
      </div>
    </div>
  </div>

  <!-- MODAL: Metric Deep Dive -->
  <div class="modal-backdrop" id="metricModal" onclick="closeModalOnBackdrop(event, 'metricModal')">
    <div class="modal-box">
      <button class="modal-close" onclick="closeModal('metricModal')">&times;</button>
      <div class="modal-title" id="metricModalTitle">Metric Title</div>
      <div class="modal-subtitle" id="metricModalSubtitle">Subhead</div>
      
      <div class="modal-body" id="metricModalBody">
        <!-- Injected by JS -->
      </div>
    </div>
  </div>

  <!-- MODAL: Generation Point Deep Dive -->
  <div class="modal-backdrop" id="genModal" onclick="closeModalOnBackdrop(event, 'genModal')">
    <div class="modal-box">
      <button class="modal-close" onclick="closeModal('genModal')">&times;</button>
      <div class="modal-title" id="genModalTitle">Generation Point</div>
      <div class="modal-subtitle" id="genModalSubtitle">Subhead</div>
      
      <div class="modal-body" id="genModalBody">
        <!-- Injected by JS -->
      </div>
    </div>
  </div>

  <script>
    const DATA = {dataset_json};
    let currentMetric = 'pass_at_1';
    let chartInstance = null;

    // Theme Toggle
    function toggleTheme() {
      const html = document.documentElement;
      const nextTheme = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      html.setAttribute('data-theme', nextTheme);
      localStorage.setItem('theme', nextTheme);
      if (chartInstance) {
        updateChartTheme();
      }
    }

    // Initialize Chart
    function initChart() {
      const ctx = document.getElementById('rsiChart').getContext('2d');
      const meta = DATA.metrics_metadata[currentMetric];

      const datasets = [
        {
          label: DATA.models.looped_84m.short_name,
          data: DATA.models.looped_84m[currentMetric],
          borderColor: DATA.models.looped_84m.color,
          backgroundColor: DATA.models.looped_84m.bg_color,
          borderWidth: 3,
          tension: 0.25,
          pointRadius: 4.5,
          pointHoverRadius: 8,
          pointBackgroundColor: DATA.models.looped_84m.color,
          fill: true
        },
        {
          label: DATA.models.flat_152m_codeparrot.short_name,
          data: DATA.models.flat_152m_codeparrot[currentMetric],
          borderColor: DATA.models.flat_152m_codeparrot.color,
          backgroundColor: 'transparent',
          borderWidth: 2,
          borderDash: [5, 4],
          tension: 0.2,
          pointRadius: 3.5,
          pointHoverRadius: 7,
          pointBackgroundColor: DATA.models.flat_152m_codeparrot.color
        },
        {
          label: DATA.models.flat_152m_edu.short_name,
          data: DATA.models.flat_152m_edu[currentMetric],
          borderColor: DATA.models.flat_152m_edu.color,
          backgroundColor: 'transparent',
          borderWidth: 2,
          borderDash: [2, 2],
          tension: 0.2,
          pointRadius: 3.5,
          pointHoverRadius: 7,
          pointBackgroundColor: DATA.models.flat_152m_edu.color
        }
      ];

      chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
          labels: DATA.generations,
          datasets: datasets
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: {
            mode: 'index',
            intersect: false
          },
          onClick: (e, elements) => {
            if (elements.length > 0) {
              const genIndex = elements[0].index;
              const dsIndex = elements[0].datasetIndex;
              const archKeys = ['looped_84m', 'flat_152m_codeparrot', 'flat_152m_edu'];
              const archId = archKeys[dsIndex] || 'looped_84m';
              openGenModal(genIndex, archId);
            }
          },
          plugins: {
            legend: {
              position: 'top',
              labels: {
                color: document.documentElement.getAttribute('data-theme') === 'dark' ? '#f8fafc' : '#0f172a',
                font: { size: 12, weight: '700' },
                padding: 16
              },
              onClick: (e, legendItem, legend) => {
                const archKeys = ['looped_84m', 'flat_152m_codeparrot', 'flat_152m_edu'];
                const archId = archKeys[legendItem.datasetIndex];
                // Click on any series legend item opens the Architecture modal!
                openArchModal(archId);
              }
            },
            tooltip: {
              padding: 14,
              cornerRadius: 10,
              titleFont: { size: 14, weight: '700' },
              bodyFont: { size: 13 },
              callbacks: {
                label: function(context) {
                  return ` ${context.dataset.label}: ${context.parsed.y}${meta.unit}`;
                },
                afterBody: function(contexts) {
                  const idx = contexts[0].dataIndex;
                  return `\n🌟 Milestone: ${DATA.models.looped_84m.milestone_tasks[idx]}`;
                }
              }
            }
          },
          scales: {
            x: {
              grid: { display: false },
              ticks: {
                color: '#94a3b8',
                font: { size: 12, weight: '600' }
              },
              title: {
                display: true,
                text: 'Autonomous RSI Traversal Generation (X-Axis)',
                color: '#94a3b8',
                font: { size: 12, weight: '600' }
              }
            },
            y: {
              beginAtZero: true,
              grid: {
                color: 'rgba(148, 163, 184, 0.1)'
              },
              ticks: {
                color: '#94a3b8',
                font: { size: 12 },
                callback: function(val) {
                  return val + meta.unit;
                }
              },
              title: {
                display: true,
                text: meta.name,
                color: '#94a3b8',
                font: { size: 12, weight: '600' }
              }
            }
          }
        }
      });
    }

    // Handle Metric Button Click: toggle view or open modal on second click
    function handleMetricClick(metricKey) {
      if (currentMetric === metricKey) {
        openMetricModal(metricKey);
      } else {
        selectMetric(metricKey);
      }
    }

    // Switch Selected Metric
    function selectMetric(metricKey) {
      currentMetric = metricKey;
      const meta = DATA.metrics_metadata[metricKey];

      // Update Chart
      document.getElementById('chartTitle').innerText = `📈 ${meta.name} Trajectory`;
      chartInstance.data.datasets[0].data = DATA.models.looped_84m[metricKey];
      chartInstance.data.datasets[1].data = DATA.models.flat_152m_codeparrot[metricKey];
      chartInstance.data.datasets[2].data = DATA.models.flat_152m_edu[metricKey];
      chartInstance.options.scales.y.title.text = meta.name;
      chartInstance.options.scales.y.ticks.callback = function(val) { return val + meta.unit; };
      chartInstance.update();

      // Update Pill Buttons
      document.querySelectorAll('.metric-pill').forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('onclick').includes(metricKey));
      });

      // Update KPI Cards
      document.querySelectorAll('.kpi-card').forEach(card => {
        card.classList.toggle('active-metric', card.id === `kpi_${metricKey}`);
      });
    }

    function updateChartTheme() {
      const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
      const textColor = isDark ? '#f8fafc' : '#0f172a';
      chartInstance.options.plugins.legend.labels.color = textColor;
      chartInstance.update();
    }

    // Open Architecture Details Modal
    function openArchModal(archId) {
      const arch = DATA.models[archId];
      document.getElementById('archModalTitle').innerText = `🏛️ ${arch.name}`;
      document.getElementById('archModalSubtitle').innerText = `${arch.status} • Weight Footprint: ${arch.weight_size_mb} MB`;

      const html = `
        <div class="modal-section-title">1. Core Architectural & Parameter Specifications</div>
        <div class="spec-grid">
          <div class="spec-item"><span class="spec-label">Total Parameters</span><span class="spec-val">${arch.total_params.toLocaleString()}</span></div>
          <div class="spec-item"><span class="spec-label">Computational Weights (Attention + FFN)</span><span class="spec-val" style="color: var(--primary);">${arch.non_emb_params.toLocaleString()} (${(100 - arch.emb_tax_pct).toFixed(1)}%)</span></div>
          <div class="spec-item"><span class="spec-label">Embedding Table Footprint (Tax)</span><span class="spec-val" style="color: ${arch.emb_tax_pct > 30 ? 'var(--danger)' : 'var(--success)'};">${arch.emb_params.toLocaleString()} (${arch.emb_tax_pct}%)</span></div>
          <div class="spec-item"><span class="spec-label">Vocabulary & Tokenizer</span><span class="spec-val">${arch.tokenizer} (${arch.vocab_size.toLocaleString()} tokens)</span></div>
          <div class="spec-item"><span class="spec-label">Physical Layers vs Virtual Depth</span><span class="spec-val">${arch.phys_layers} Physical Layers ➔ ${arch.recurrent_loops}</span></div>
          <div class="spec-item"><span class="spec-label">Autoregressive Decoding Speed</span><span class="spec-val" style="color: var(--success);">${arch.decode_tok_s} tok/s (Fast: ${arch.fast_reflex_tok_s} tok/s)</span></div>
        </div>

        <div class="modal-section-title">2. Recurrent Invariant & Weight Memory Bandwidth</div>
        <div class="math-block">
${archId === 'looped_84m' ? 
`// Inter-Loop RMSNorm Invariant Formulation:
h^(1) = TransformerBlocks_0..7(x)
h^(2) = TransformerBlocks_0..7( RMSNorm(h^(1)) )  <-- Resets activation variance to σ² = 1.0!

// Memory Bandwidth Advantage:
Weight Size = 168.9 MB (halved from 308.3 MB)
GB10 Unified LPDDR5X Bandwidth = 512 GB/s
Theoretical Upper Bound = 512 GB/s / 168.9 MB = 3,031 layers/s ➔ 189 tok/s!`
:
`// Standard Flat Sequential Transformer (No Recurrence):
h = TransformerBlocks_0..11(x)

// The Vocabulary Tax Failure Mode:
Vocabulary = 100,262 tokens * 768 hidden_dim = 77,001,216 parameters!
Over 50.5% of the total model budget was spent on static word embeddings.
Only 75.5M parameters were allocated to sequential attention reasoning blocks.
Autoregressive decode bandwidth was restricted to ~49.9 tok/s.`}
        </div>

        <div class="modal-section-title">3. Trajectory & Performance Summary</div>
        <div style="background: var(--bg); padding: 14px; border-radius: 12px; border: 1px solid var(--border); font-size: 0.88rem;">
          <b>Pass@1 Frontier:</b> ${arch.pass_at_1[0]}% at M0 ➔ <b>${Math.max(...arch.pass_at_1)}% peak</b><br/>
          <b>Pass@5 Frontier:</b> ${arch.pass_at_5[0]}% at M0 ➔ <b>${Math.max(...arch.pass_at_5)}% peak</b><br/>
          <b>RSI Optimization Loss:</b> ${arch.loss_masking}<br/>
          <b>Operational Status:</b> ${arch.status}
        </div>
      `;

      document.getElementById('archModalBody').innerHTML = html;
      document.getElementById('archModal').classList.add('open');
    }

    // Open Metric Details Modal
    function openMetricModal(metricKey) {
      const meta = DATA.metrics_metadata[metricKey];
      document.getElementById('metricModalTitle').innerText = `📊 ${meta.name}`;
      document.getElementById('metricModalSubtitle').innerText = `Unit: ${meta.unit} • Champion Score: ${meta.champion_value}`;

      const html = `
        <div class="modal-section-title">1. Metric Definition & Scientific Formulation</div>
        <div style="font-size: 0.9rem; margin-bottom: 12px; color: var(--text);">
          ${meta.description}
        </div>
        <div class="math-block" style="margin-bottom: 14px;">
${meta.formula}
        </div>

        <div class="modal-section-title">2. Why It Matters in Recursive Self-Improvement</div>
        <div style="background: var(--bg); padding: 14px; border-radius: 12px; border: 1px solid var(--border); font-size: 0.88rem; line-height: 1.6;">
          ${meta.why_it_matters}
        </div>

        <div class="modal-section-title" style="margin-top: 14px;">3. Comparative Outcome Across Architectures</div>
        <div class="spec-grid">
          <div class="spec-item">
            <span class="spec-label">Mirendil-84M Looped Multi-Depth</span>
            <span class="spec-val" style="color: var(--primary);">${meta.champion_value}</span>
          </div>
          <div class="spec-item">
            <span class="spec-label">Mirendil-152M Flat Baseline</span>
            <span class="spec-val" style="color: var(--danger);">${meta.baseline_value}</span>
          </div>
          <div class="spec-item">
            <span class="spec-label">Cumulative Performance Multiplier</span>
            <span class="spec-val" style="color: var(--success);">${meta.multiplier}</span>
          </div>
          <div class="spec-item">
            <span class="spec-label">Measurement Method</span>
            <span class="spec-val">Sandboxed Python Subprocess Verifier</span>
          </div>
        </div>
      `;

      document.getElementById('metricModalBody').innerHTML = html;
      document.getElementById('metricModal').classList.add('open');
    }

    // Open Generation Point Modal
    function openGenModal(genIndex, archId = 'looped_84m') {
      const genLabel = DATA.generations[genIndex];
      const looped = DATA.models.looped_84m;
      const flat = DATA.models.flat_152m_codeparrot;
      const milestone = looped.milestone_tasks[genIndex];
      const wandbRun = looped.wandb_runs[genIndex];
      const currentArch = DATA.models[archId] || looped;

      document.getElementById('genModalTitle').innerText = `🔍 Generation ${genLabel} Forensic Record`;
      document.getElementById('genModalSubtitle').innerText = `Active Series: ${currentArch.name} • Hardware: NVIDIA DGX Grace Blackwell GB10`;

      const html = `
        <div style="margin-bottom: 14px;">
          <button class="btn btn-primary" onclick="closeModal('genModal'); openArchModal('${archId}');" style="width: 100%; justify-content: center; font-size: 0.88rem; padding: 10px;">
            🏛️ Click Here to Inspect ${currentArch.name} Specifications ➔
          </button>
        </div>

        <div class="modal-section-title">1. Landmark Algorithmic Frontier Conquered</div>
        <div style="background: rgba(79, 70, 229, 0.1); border: 1px solid var(--primary); padding: 14px; border-radius: 12px; font-size: 0.95rem; font-weight: 600; color: var(--text);">
          ⭐ ${milestone}
        </div>

        <div class="modal-section-title" style="margin-top: 14px;">2. Comparative Metrics in Generation ${genLabel}</div>
        <div class="spec-grid">
          <div class="spec-item"><span class="spec-label">Looped 84M Pass@1 (Zero-Shot)</span><span class="spec-val" style="color: var(--primary);">${looped.pass_at_1[genIndex]}%</span></div>
          <div class="spec-item"><span class="spec-label">Flat 152M Baseline Pass@1</span><span class="spec-val" style="color: var(--danger);">${flat.pass_at_1[genIndex]}%</span></div>
          <div class="spec-item"><span class="spec-label">Looped 84M Pass@5</span><span class="spec-val" style="color: var(--success);">${looped.pass_at_5[genIndex]}%</span></div>
          <div class="spec-item"><span class="spec-label">Looped Pass@16 Solve Breadth</span><span class="spec-val">${looped.pass_at_16[genIndex]}%</span></div>
          <div class="spec-item"><span class="spec-label">Candidate Pass Density</span><span class="spec-val">${looped.candidate_density[genIndex]}%</span></div>
          <div class="spec-item"><span class="spec-label">Gold Verified Solutions Harvested</span><span class="spec-val">${looped.gold_harvested[genIndex]} programs</span></div>
          <div class="spec-item"><span class="spec-label">AST Legal Syntax Rate</span><span class="spec-val">${looped.syntax_valid_pct[genIndex]}%</span></div>
          <div class="spec-item"><span class="spec-label">Weights & Biases Run Artifact</span><span class="spec-val"><a href="https://forge.coreweave.com/wandb/ramakant_shankar/rsi-coding-pretrain/runs/${wandbRun}" target="_blank" style="color: var(--primary); font-family: monospace;">${wandbRun} ↗</a></span></div>
        </div>
      `;

      document.getElementById('genModalBody').innerHTML = html;
      document.getElementById('genModal').classList.add('open');
    }

    function closeModal(modalId) {
      document.getElementById(modalId).classList.remove('open');
    }

    function closeModalOnBackdrop(e, modalId) {
      if (e.target.id === modalId) {
        closeModal(modalId);
      }
    }

    // Populate Ledger Table
    function populateLedger() {
      const tbody = document.getElementById('ledgerTableBody');
      const looped = DATA.models.looped_84m;
      const flat = DATA.models.flat_152m_codeparrot;

      let rows = '';
      for (let i = DATA.generations.length - 1; i >= 0; i--) {
        const gen = DATA.generations[i];
        rows += `
          <tr onclick="openGenModal(${i})">
            <td><b style="color: var(--primary);">${gen}</b></td>
            <td><b style="color: var(--primary);">${looped.pass_at_1[i]}%</b></td>
            <td><b style="color: var(--success);">${looped.pass_at_5[i]}%</b></td>
            <td>${looped.candidate_density[i]}%</td>
            <td><b>${looped.gold_harvested[i]}</b></td>
            <td style="color: var(--text-muted);">${flat.pass_at_1[i]}%</td>
            <td style="color: var(--text-muted);">${flat.pass_at_5[i]}%</td>
            <td style="font-size: 0.82rem;">${looped.milestone_tasks[i]}</td>
            <td><a href="https://forge.coreweave.com/wandb/ramakant_shankar/rsi-coding-pretrain/runs/${looped.wandb_runs[i]}" target="_blank" style="color: var(--primary); font-family: monospace;" onclick="event.stopPropagation();">${looped.wandb_runs[i]} ↗</a></td>
          </tr>
        `;
      }
      tbody.innerHTML = rows;
    }

    // Init on DOM Load
    document.addEventListener('DOMContentLoaded', () => {
      const savedTheme = localStorage.getItem('theme') || 'dark';
      document.documentElement.setAttribute('data-theme', savedTheme);
      initChart();
      populateLedger();
    });
  </script>
</body>
</html>
"""

def generate_dashboard():
    json_str = json.dumps(DATASET)
    html_content = HTML_TEMPLATE.replace("{dataset_json}", json_str)
    
    workspace_file = ROOT_DIR / "DASHBOARD.html"
    gdrive_file = GDRIVE_DIR / "DASHBOARD.html"
    
    workspace_file.write_text(html_content, encoding="utf-8")
    print(f"Generated Workspace Dashboard: {workspace_file} ({len(html_content):,} bytes)")
    
    shutil.copy2(workspace_file, gdrive_file)
    print(f"Mirrored to Google Drive: {gdrive_file} ({gdrive_file.stat().st_size:,} bytes)")

if __name__ == "__main__":
    generate_dashboard()
