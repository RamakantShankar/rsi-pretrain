#!/usr/bin/env python3
"""
Mirendil Laboratory - Mirror Parity Verification Script
Compares all 11 core research deliverables between local workspace and Google Drive mirror.
"""

import os
import hashlib
from pathlib import Path

FILES = [
    "RESEARCH_JOURNAL.md",
    "RESEARCH_JOURNAL.html",
    "RESEARCH_COMPENDIUM.md",
    "RESEARCH_COMPENDIUM.html",
    "EXPERIMENT_LOG.md",
    "EXPERIMENT_LOG.html",
    "EXPERIMENT_AND_ABLATION_LOG.md",
    "DASHBOARD.html",
    "annotations.json",
    "notebooks/14_mirendil_154m_playground.ipynb",
    "notebooks/15_looped_transformer_explained.ipynb"
]

SRC_DIR = Path("M:/Rama/Study/llm/LLM/RSI")
DST_DIR = Path("D:/D/Google Drive/RSI_Pretraining_Lab")

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 80)
    print("MIRENDIL PRETRAINING LAB: 100% BIT-FOR-BIT MIRROR INTEGRITY AUDIT")
    print("=" * 80)
    print(f"Source: {SRC_DIR}")
    print(f"Mirror: {DST_DIR}\n")

    all_matched = True
    for rel in FILES:
        src = SRC_DIR / rel
        dst = DST_DIR / rel

        if not src.exists():
            print(f"[MISSING SRC] {rel}")
            all_matched = False
            continue

        if not dst.exists():
            print(f"[MISSING DST] {rel}")
            all_matched = False
            continue

        s_size = src.stat().st_size
        d_size = dst.stat().st_size
        s_hash = sha256_file(src)
        d_hash = sha256_file(dst)

        match = (s_size == d_size and s_hash == d_hash)
        if not match:
            all_matched = False
            print(f"[MISMATCH]    {rel:<45} (Src: {s_size}B != Dst: {d_size}B)")
        else:
            print(f"[MATCH 100%]  {rel:<45} ({s_size:,} bytes) SHA: {s_hash[:12]}...")

    print("=" * 80)
    if all_matched:
        print("RESULT: ALL 11 DELIVERABLES ARE 100% BIT-FOR-BIT SYNCHRONIZED ON GOOGLE DRIVE!")
    else:
        print("RESULT: DESYNCHRONIZATION DETECTED - PLEASE INVESTIGATE MISMATCHED FILES.")
    print("=" * 80)

if __name__ == "__main__":
    main()
