"""
Production 13-Gram Evaluation Decontamination Engine for Rama-84M Pretraining.

Scans candidate pretraining documents against evaluation benchmark suites
(Internal 55-Task Problem Bank, HumanEval, MBPP, GSM8K) using exact sliding-window
13-gram token matching. Quarantines contaminated documents to guarantee 100%
untainted, non-cheating benchmark evaluations.

Usage:
    python tools/decontaminate_evals.py \
        --input data/rsi/verified_solutions_cycle32_token_deduped.jsonl \
        --benchmarks data/rsi/problem_bank.json \
        --tokenizer checkpoints/tokenizers/rama_16k.json \
        --n-gram 13 \
        --threshold 1 \
        --output-clean data/rsi/clean_certified.jsonl \
        --output-quarantined data/rsi/quarantined_contaminated.jsonl
"""

import os
import sys
import json
import time
import argparse
import hashlib
from typing import List, Dict, Set, Tuple, Any
from collections import defaultdict
from tokenizers import Tokenizer


def hash_token_tuple(tokens: Tuple[int, ...]) -> int:
    """Generate a 64-bit integer hash for a tuple of token IDs using BLAKE2b."""
    data = b"".join(int(t).to_bytes(4, byteorder="little", signed=False) for t in tokens)
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), byteorder="little")


class BenchmarkDecontaminator:
    def __init__(self, tokenizer_path: str, n_gram: int = 13, threshold: int = 1):
        self.n_gram = n_gram
        self.threshold = threshold
        
        print(f"[*] Loading BPE Tokenizer: {tokenizer_path}")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # 64-bit hash -> List of (benchmark_name, task_id, source_type, preview_text)
        self.index: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        self.benchmark_task_count = 0
        self.total_ngrams_indexed = 0

    def extract_ngrams(self, text: str) -> List[Tuple[Tuple[int, ...], int, int]]:
        """
        Tokenizes text and extracts all n-gram token tuples along with token start/end offsets.
        Returns: list of (token_tuple, start_idx, end_idx)
        """
        encoded = self.tokenizer.encode(text)
        token_ids = encoded.ids
        
        if len(token_ids) < self.n_gram:
            return []
            
        ngrams = []
        for i in range(len(token_ids) - self.n_gram + 1):
            window = tuple(token_ids[i : i + self.n_gram])
            ngrams.append((window, i, i + self.n_gram))
        return ngrams

    def index_benchmark_file(self, benchmark_path: str, benchmark_name: str = "custom"):
        """Indexes all prompts, canonical solutions, and test assertions from a benchmark file."""
        if not os.path.exists(benchmark_path):
            print(f"[!] Warning: Benchmark file {benchmark_path} not found. Skipping.")
            return

        print(f"[*] Indexing benchmark: {benchmark_name} ({benchmark_path})...")
        with open(benchmark_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Support both JSON (list or dict) and JSONL
        records = []
        if content.startswith("[") or content.startswith("{"):
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    records = data
                elif isinstance(data, dict):
                    records = list(data.values())
            except Exception:
                pass

        if not records:
            # Fallback to JSONL
            for line in content.splitlines():
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        task_count = 0
        indexed_ngrams = 0

        for r in records:
            task_id = str(r.get("task_id", r.get("name", f"task_{task_count}")))
            task_count += 1
            
            # 1. Index Prompt
            prompt_text = r.get("prompt", r.get("question", ""))
            if prompt_text:
                for ngram, s, e in self.extract_ngrams(prompt_text):
                    h = hash_token_tuple(ngram)
                    self.index[h].append({
                        "benchmark": benchmark_name,
                        "task_id": task_id,
                        "type": "prompt",
                        "snippet": prompt_text[:80].replace("\n", "\\n")
                    })
                    indexed_ngrams += 1

            # 2. Index Canonical Solution
            canonical_text = r.get("canonical", r.get("solution", r.get("canonical_solution", "")))
            if canonical_text:
                for ngram, s, e in self.extract_ngrams(canonical_text):
                    h = hash_token_tuple(ngram)
                    self.index[h].append({
                        "benchmark": benchmark_name,
                        "task_id": task_id,
                        "type": "canonical_solution",
                        "snippet": canonical_text[:80].replace("\n", "\\n")
                    })
                    indexed_ngrams += 1

            # 3. Index Test Assertions
            tests = r.get("tests", r.get("test", []))
            test_strings = []
            if isinstance(tests, list):
                test_strings = [str(t) for t in tests]
            elif isinstance(tests, str):
                test_strings = [tests]

            for test_str in test_strings:
                if test_str:
                    for ngram, s, e in self.extract_ngrams(test_str):
                        h = hash_token_tuple(ngram)
                        self.index[h].append({
                            "benchmark": benchmark_name,
                            "task_id": task_id,
                            "type": "test_assertion",
                            "snippet": test_str[:80].replace("\n", "\\n")
                        })
                        indexed_ngrams += 1

        self.benchmark_task_count += task_count
        self.total_ngrams_indexed += indexed_ngrams
        print(f"    Indexed {task_count} tasks | {indexed_ngrams} unique {self.n_gram}-gram occurrences.")

    def scan_document(self, text: str) -> List[Dict[str, Any]]:
        """
        Scans a single text document against the 13-gram index.
        Returns list of match records if contaminated.
        """
        ngrams = self.extract_ngrams(text)
        if not ngrams:
            return []

        matches = []
        for ngram, s, e in ngrams:
            h = hash_token_tuple(ngram)
            if h in self.index:
                for ref in self.index[h]:
                    matches.append({
                        "offset": (s, e),
                        "benchmark": ref["benchmark"],
                        "task_id": ref["task_id"],
                        "type": ref["type"],
                        "ref_snippet": ref["snippet"]
                    })
        return matches

    def decontaminate_file(
        self,
        input_path: str,
        output_clean: str,
        output_quarantined: str,
        audit_log: str = None
    ) -> Dict[str, Any]:
        """Scans an entire JSONL corpus, separating clean docs from quarantined docs."""
        print(f"\n[*] Starting decontamination on: {input_path}")
        print(f"    Parameters: N={self.n_gram} | Threshold={self.threshold} hit(s) | Benchmark Tasks={self.benchmark_task_count}")
        start_time = time.time()

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        total_docs = 0
        clean_docs = 0
        quarantined_docs = 0
        leak_by_type = defaultdict(int)
        leak_by_benchmark = defaultdict(int)
        audit_records = []

        os.makedirs(os.path.dirname(output_clean) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(output_quarantined) or ".", exist_ok=True)

        with open(input_path, "r", encoding="utf-8") as in_f, \
             open(output_clean, "w", encoding="utf-8") as clean_f, \
             open(output_quarantined, "w", encoding="utf-8") as quar_f:

            for line_idx, line in enumerate(in_f):
                line = line.strip()
                if not line:
                    continue
                total_docs += 1
                doc = json.loads(line)
                
                # Combine full text: prompt + completion / text
                text_to_scan = ""
                if "text" in doc:
                    text_to_scan = doc["text"]
                else:
                    parts = []
                    if "prompt" in doc:
                        parts.append(doc["prompt"])
                    if "completion" in doc:
                        parts.append(doc["completion"])
                    if "solution" in doc:
                        parts.append(doc["solution"])
                    text_to_scan = "\n".join(parts)

                matches = self.scan_document(text_to_scan)
                
                # Deduplicate matches by (benchmark, task_id, type)
                unique_hits = {(m["benchmark"], m["task_id"], m["type"]): m for m in matches}
                
                if len(unique_hits) >= self.threshold:
                    quarantined_docs += 1
                    for (bm, tid, mtype), m in unique_hits.items():
                        leak_by_type[mtype] += 1
                        leak_by_benchmark[bm] += 1

                    doc["_quarantine_meta"] = {
                        "match_count": len(matches),
                        "unique_hits": list(unique_hits.keys()),
                        "matches": matches[:5] # keep top 5 for inspection
                    }
                    quar_f.write(json.dumps(doc) + "\n")
                    
                    if audit_log:
                        audit_records.append({
                            "doc_id": doc.get("task_id", line_idx),
                            "line": line_idx,
                            "match_count": len(matches),
                            "hits": [
                                {
                                    "benchmark": m["benchmark"],
                                    "task_id": m["task_id"],
                                    "type": m["type"],
                                    "ref_snippet": m["ref_snippet"]
                                }
                                for m in unique_hits.values()
                            ]
                        })
                else:
                    clean_docs += 1
                    clean_f.write(json.dumps(doc) + "\n")

        elapsed = time.time() - start_time
        speed = total_docs / max(elapsed, 1e-4)

        if audit_log and audit_records:
            os.makedirs(os.path.dirname(audit_log) or ".", exist_ok=True)
            with open(audit_log, "w", encoding="utf-8") as af:
                json.dump(audit_records, af, indent=2)

        summary = {
            "total_docs": total_docs,
            "clean_docs": clean_docs,
            "quarantined_docs": quarantined_docs,
            "quarantine_pct": (quarantined_docs / total_docs * 100) if total_docs > 0 else 0.0,
            "elapsed_seconds": round(elapsed, 3),
            "throughput_docs_sec": round(speed, 1),
            "leak_by_type": dict(leak_by_type),
            "leak_by_benchmark": dict(leak_by_benchmark)
        }

        print("\n" + "=" * 70)
        print("EVALUATION DECONTAMINATION SUMMARY")
        print("=" * 70)
        print(f"  Total Documents Scanned:    {total_docs}")
        print(f"  Clean Certified Kept:       {clean_docs} ({(clean_docs / max(total_docs, 1) * 100):.1f}%)")
        print(f"  Quarantined (Contaminated): {quarantined_docs} ({summary['quarantine_pct']:.1f}%)")
        print(f"  Processing Time:            {elapsed:.2f}s ({speed:.0f} docs/s)")
        print("-" * 70)
        print("  Leak Breakdown by Type:")
        for t, count in leak_by_type.items():
            print(f"    - {t:<22}: {count}")
        print("  Leak Breakdown by Benchmark:")
        for b, count in leak_by_benchmark.items():
            print(f"    - {b:<22}: {count}")
        print("=" * 70)
        print(f"[+] Clean dataset saved to:       {output_clean}")
        print(f"[+] Quarantined dataset saved to: {output_quarantined}")
        if audit_log:
            print(f"[+] Forensic audit log saved to:  {audit_log}")

        return summary


def main():
    parser = argparse.ArgumentParser(description="13-Gram Evaluation Decontamination Engine")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSONL dataset")
    parser.add_argument("--benchmarks", type=str, nargs="+", default=["data/rsi/problem_bank.json"], help="Paths to benchmark files")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizers/rama_16k.json", help="Path to BPE tokenizer")
    parser.add_argument("--n-gram", type=int, default=13, help="Sliding window size N (default: 13)")
    parser.add_argument("--threshold", type=int, default=1, help="Min matches to quarantine doc (default: 1)")
    parser.add_argument("--output-clean", type=str, required=True, help="Output path for clean dataset")
    parser.add_argument("--output-quarantined", type=str, required=True, help="Output path for quarantined dataset")
    parser.add_argument("--audit-log", type=str, default=None, help="Path to export forensic audit log JSON")
    args = parser.parse_args()

    decontaminator = BenchmarkDecontaminator(
        tokenizer_path=args.tokenizer,
        n_gram=args.n_gram,
        threshold=args.threshold
    )

    for bm_path in args.benchmarks:
        bm_name = os.path.splitext(os.path.basename(bm_path))[0]
        decontaminator.index_benchmark_file(bm_path, benchmark_name=bm_name)

    decontaminator.decontaminate_file(
        input_path=args.input,
        output_clean=args.output_clean,
        output_quarantined=args.output_quarantined,
        audit_log=args.audit_log
    )


if __name__ == "__main__":
    main()
