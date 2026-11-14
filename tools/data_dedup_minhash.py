#!/usr/bin/env python3
"""
MinHash LSH Deduplication Engine for LLM Pretraining Data.
Implements:
  1. k-Shingling (Word-level k=5, Code-level k=6)
  2. 128-Permutation MinHash Signatures using 64-bit Mersenne Prime (2^61 - 1)
  3. LSH Banding (16 bands x 8 rows = 128 slots) with Super-Bucket Cap
  4. 128-Slot RAM Signature Verification (Jaccard threshold >= 0.80)
  5. Connected Components Graph Clustering (Union-Find Disjoint Set)
  6. Quality-Weighted Representative Selection (Keeps highest quality doc)
"""

import os
import sys
import time
import json
import random
import argparse
from typing import List, Dict, Set, Tuple, Any
from collections import defaultdict


# 61-bit Mersenne Prime for 64-bit integer hashing
MERSENNE_PRIME = (1 << 61) - 1


class UnionFind:
    """Disjoint Set Union (Union-Find) with path compression and rank optimization."""
    def __init__(self, elements):
        self.parent = {x: x for x in elements}
        self.rank = {x: 0 for x in elements}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1


class MinHashLSHDeduplicator:
    def __init__(
        self,
        num_perm: int = 128,
        num_bands: int = 16,
        rows_per_band: int = 8,
        threshold: float = 0.80,
        k_shingle: int = 5,
        super_bucket_cap: int = 1000,
        seed: int = 42
    ):
        assert num_bands * rows_per_band == num_perm, "num_bands * rows_per_band must equal num_perm!"
        self.num_perm = num_perm
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        self.threshold = threshold
        self.k_shingle = k_shingle
        self.super_bucket_cap = super_bucket_cap

        # Generate deterministic universal hash parameters (a_i, b_i)
        random.seed(seed)
        self.a_params = [random.randint(1, MERSENNE_PRIME - 1) for _ in range(num_perm)]
        self.b_params = [random.randint(0, MERSENNE_PRIME - 1) for _ in range(num_perm)]

    def get_shingles(self, text: str) -> Set[int]:
        """Extracts word k-shingles and hashes each shingle to a 64-bit integer."""
        words = text.strip().split()
        if len(words) < self.k_shingle:
            # Fallback for very short texts: return single shingle
            return {hash(" ".join(words)) & 0xFFFFFFFF}
        
        shingles = set()
        for i in range(len(words) - self.k_shingle + 1):
            shingle_str = " ".join(words[i : i + self.k_shingle])
            # Hash string to 64-bit integer
            shingle_hash = hash(shingle_str) & 0xFFFFFFFFFFFFFFFF
            shingles.add(shingle_hash)
        return shingles

    def compute_minhash_signature(self, shingles: Set[int]) -> List[int]:
        """Computes 128-integer MinHash signature using universal hashing."""
        sig = []
        for a, b in zip(self.a_params, self.b_params):
            min_val = MERSENNE_PRIME
            for s in shingles:
                h_val = (a * s + b) % MERSENNE_PRIME
                if h_val < min_val:
                    min_val = h_val
            sig.append(min_val)
        return sig

    def deduplicate(self, documents: List[Dict[str, Any]], text_key: str = "text") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Full End-to-End MinHash LSH Deduplication Pipeline:
          1. Extract shingles & compute 128-int signatures
          2. Index signatures into 16 LSH band tables
          3. Generate candidate pairs from bucket collisions (with super-bucket cap)
          4. Verify 128-slot signature similarity (J >= 0.80)
          5. Cluster matching pairs using Union-Find
          6. Select highest-quality document per cluster
        """
        t0 = time.time()
        num_docs = len(documents)
        print(f"[*] Starting MinHash LSH deduplication on {num_docs:,} documents...")
        print(f"    Params: {self.num_perm} hashes | {self.num_bands} bands x {self.rows_per_band} rows | J >= {self.threshold} | k={self.k_shingle}")

        # 1. Compute MinHash Signatures
        print("[1/5] Computing MinHash signatures...")
        signatures = []
        for i, doc in enumerate(documents):
            text = doc.get(text_key, "")
            shingles = self.get_shingles(text)
            sig = self.compute_minhash_signature(shingles)
            signatures.append(sig)
            if (i + 1) % 1000 == 0 or (i + 1) == num_docs:
                print(f"      Signatures computed: {i+1:,}/{num_docs:,}", end="\r")
        print(f"\n      Signatures completed in {time.time() - t0:.2f}s")

        # 2. LSH Banding & Bucket Indexing
        t_lsh = time.time()
        print("[2/5] Indexing into LSH band hash tables...")
        # 16 band tables: band_idx -> {band_tuple_key: [doc_indices]}
        band_tables = [defaultdict(list) for _ in range(self.num_bands)]
        
        for doc_id, sig in enumerate(signatures):
            for band_idx in range(self.num_bands):
                start = band_idx * self.rows_per_band
                end = start + self.rows_per_band
                band_key = tuple(sig[start:end])
                band_tables[band_idx][band_key].append(doc_id)

        # 3. Candidate Pair Extraction with Super-Bucket Cap
        print("[3/5] Extracting candidate duplicate pairs from buckets...")
        candidate_pairs = set()
        super_bucket_count = 0
        instant_clusters = []

        for band_idx, table in enumerate(band_tables):
            for band_key, doc_ids in table.items():
                size = len(doc_ids)
                if size <= 1:
                    continue
                elif size > self.super_bucket_cap:
                    # Super-Bucket: treat entire group as instant duplicate cluster
                    super_bucket_count += 1
                    instant_clusters.append(doc_ids)
                else:
                    # Standard bucket: add all pairwise candidate pairs
                    for i in range(size):
                        for j in range(i + 1, size):
                            d1, d2 = min(doc_ids[i], doc_ids[j]), max(doc_ids[i], doc_ids[j])
                            candidate_pairs.add((d1, d2))

        print(f"      Candidate pairs identified: {len(candidate_pairs):,}")
        if super_bucket_count > 0:
            print(f"      Super-buckets capped: {super_bucket_count:,} (large boilerplate groups)")

        # 4. Signature Verification in RAM
        t_ver = time.time()
        print(f"[4/5] Verifying {len(candidate_pairs):,} candidate pairs in RAM (J >= {self.threshold})...")
        verified_edges = []
        for d1, d2 in candidate_pairs:
            sig1 = signatures[d1]
            sig2 = signatures[d2]
            # Fast vectorized equality count
            match_count = sum(1 for a, b in zip(sig1, sig2) if a == b)
            est_jaccard = match_count / self.num_perm
            if est_jaccard >= self.threshold:
                verified_edges.append((d1, d2))

        print(f"      True duplicates verified: {len(verified_edges):,}")

        # 5. Connected Components Graph Clustering (Union-Find)
        t_clust = time.time()
        print("[5/5] Clustering duplicates & pruning via Quality Scoring...")
        uf = UnionFind(range(num_docs))
        
        for d1, d2 in verified_edges:
            uf.union(d1, d2)
        for cluster in instant_clusters:
            anchor = cluster[0]
            for member in cluster[1:]:
                uf.union(anchor, member)

        # Group docs by root cluster
        clusters = defaultdict(list)
        for doc_id in range(num_docs):
            root = uf.find(doc_id)
            clusters[root].append(doc_id)

        # Quality-Weighted Selection
        # Score = text length (completeness) + comment bonus
        def quality_score(doc_idx: int) -> float:
            t = documents[doc_idx].get(text_key, "")
            score = float(len(t))
            if "#" in t or '"""' in t:
                score *= 1.2  # Bonus for documentation
            return score

        pruned_indices = set()
        duplicate_cluster_count = 0

        for root, members in clusters.items():
            if len(members) > 1:
                duplicate_cluster_count += 1
                # Pick document with highest quality score as winner
                winner = max(members, key=quality_score)
                for m in members:
                    if m != winner:
                        pruned_indices.add(m)

        # Build clean output list
        clean_documents = [doc for i, doc in enumerate(documents) if i not in pruned_indices]
        total_time = time.time() - t0

        stats = {
            "total_input_documents": num_docs,
            "candidate_pairs_checked": len(candidate_pairs),
            "verified_duplicate_edges": len(verified_edges),
            "duplicate_clusters_found": duplicate_cluster_count,
            "documents_pruned": len(pruned_indices),
            "clean_documents_remaining": len(clean_documents),
            "reduction_percent": (len(pruned_indices) / max(1, num_docs)) * 100.0,
            "elapsed_seconds": total_time,
            "docs_per_second": num_docs / max(1e-5, total_time)
        }

        print("\n" + "=" * 70)
        print("MINHASH LSH DEDUPLICATION SUMMARY")
        print("=" * 70)
        print(f"  Input Documents:        {stats['total_input_documents']:,}")
        print(f"  Candidate Pairs:        {stats['candidate_pairs_checked']:,}")
        print(f"  Duplicate Clusters:     {stats['duplicate_clusters_found']:,}")
        print(f"  Documents Pruned:       {stats['documents_pruned']:,} ({stats['reduction_percent']:.1f}% pruned)")
        print(f"  Clean Documents Kept:   {stats['clean_documents_remaining']:,}")
        print(f"  Processing Time:        {stats['elapsed_seconds']:.2f}s ({stats['docs_per_second']:.0f} docs/s)")
        print("=" * 70)

        return clean_documents, stats


def main():
    parser = argparse.ArgumentParser(description="MinHash LSH Deduplicator for LLM Training Corpora")
    parser.add_argument("--input", type=str, required=True, help="Input JSON or JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output deduplicated JSON or JSONL file")
    parser.add_argument("--text-key", type=str, default="solution", help="JSON key containing text to deduplicate")
    parser.add_argument("--k", type=int, default=5, help="Shingle size (default: 5)")
    parser.add_argument("--threshold", type=float, default=0.80, help="Jaccard threshold (default: 0.80)")
    parser.add_argument("--bands", type=int, default=16, help="Number of LSH bands (default: 16)")
    parser.add_argument("--rows", type=int, default=8, help="Rows per band (default: 8)")
    args = parser.parse_args()

    # Load Input Data
    print(f"Loading input data from: {args.input}")
    docs = []
    if args.input.endswith(".jsonl"):
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            docs = json.load(f)

    # Initialize Engine & Run
    engine = MinHashLSHDeduplicator(
        num_perm=args.bands * args.rows,
        num_bands=args.bands,
        rows_per_band=args.rows,
        threshold=args.threshold,
        k_shingle=args.k
    )

    clean_docs, stats = engine.deduplicate(docs, text_key=args.text_key)

    # Save Output Data
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Saving deduplicated dataset to: {args.output}")
    if args.output.endswith(".jsonl"):
        with open(args.output, "w", encoding="utf-8") as f:
            for d in clean_docs:
                f.write(json.dumps(d) + "\n")
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(clean_docs, f, indent=2)

    print("Deduplication Complete!")


if __name__ == "__main__":
    main()
