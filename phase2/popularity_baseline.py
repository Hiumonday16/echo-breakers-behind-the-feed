#!/usr/bin/env python
"""Rank each dev impression by how often that article was clicked in training.

Phase 2 groundwork, from the team's MIND/popularity_baseline.py (same ranking logic; paths are now
arguments so it runs on this repo's data/ layout).

Usage:
    python phase2/popularity_baseline.py --train data/MINDsmall_train --dev data/MINDsmall_dev --out data/phase2_eval

Writes <out>/ref/truth.txt and <out>/res/prediction.txt in the MIND competition format. Score them with
Microsoft's evaluate.py (https://github.com/msnews/MIND/blob/master/evaluate.py, needs scikit-learn):
    python evaluate.py data/phase2_eval data/phase2_eval/scores
"""
import argparse
import json
import os
from collections import Counter


def click_counts(path):
    counts = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            impressions = line.rstrip("\n").split("\t")[4]
            for token in impressions.split():
                news_id, label = token.rsplit("-", 1)
                if label == "1":
                    counts[news_id] += 1
    return counts


def write_split(path, counts, truth_path, pred_path):
    os.makedirs(os.path.dirname(truth_path), exist_ok=True)
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    n = 0
    with open(path, encoding="utf-8") as src, open(truth_path, "w", encoding="utf-8") as truth, open(
        pred_path, "w", encoding="utf-8"
    ) as pred:
        for line in src:
            imp_id, _, _, _, impressions = line.rstrip("\n").split("\t")
            pairs = []
            for token in impressions.split():
                news_id, label = token.rsplit("-", 1)
                pairs.append((news_id, int(label)))
            # Higher training click count is a better rank. Ties keep impression order.
            order = sorted(range(len(pairs)), key=lambda i: (-counts[pairs[i][0]], i))
            ranks = [0] * len(pairs)
            for rank, index in enumerate(order, start=1):
                ranks[index] = rank
            labels = [label for _, label in pairs]
            truth.write("{} {}\n".format(imp_id, json.dumps(labels, separators=(",", ":"))))
            pred.write("{} {}\n".format(imp_id, json.dumps(ranks, separators=(",", ":"))))
            n += 1
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="folder with the training behaviors.tsv")
    ap.add_argument("--dev", required=True, help="folder with the dev behaviors.tsv")
    ap.add_argument("--out", default="data/phase2_eval")
    args = ap.parse_args()
    counts = click_counts(os.path.join(args.train, "behaviors.tsv"))
    n = write_split(os.path.join(args.dev, "behaviors.tsv"), counts,
                    os.path.join(args.out, "ref", "truth.txt"),
                    os.path.join(args.out, "res", "prediction.txt"))
    print("impressions {}  clicked-articles {}".format(n, len(counts)))
