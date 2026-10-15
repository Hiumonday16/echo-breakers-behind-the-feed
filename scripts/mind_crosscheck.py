"""
Echo Breakers - cross-checks of the two MIND analyses (ours and MIND/analysis/phase1_diversity.py).

Per impression with a non-empty history, it tests two claims:
  1. "Shown lists are broader than click histories": compares entropy of history vs impression
     with both sides sampled down to the same size (entropy rises with sample size).
  2. "70% of slots stay inside historical categories": compares that share with what a
     non-personalized feed would give (platform-wide category mix over the same categories).
  3. Click rate on slots in a familiar category (already in the history) vs a new one,
     overall and within each category (the relevance side of the diversity trade-off).

Outputs (in --out, with a _dev suffix for the dev split):
    key_numbers_crosscheck[_dev].csv  incl. data-quality checks (dq_*), as in phase1_diversity.py
    ctr_familiar_vs_new[_dev].csv, history_vs_shown_mix[_dev].csv,
    09_ctr_familiar_vs_new[_dev].png

Usage:
    python scripts/mind_crosscheck.py --data data/MINDsmall_train --out results
"""
import argparse
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import chart_style
from mind_analysis import KeyNumbers, entropy, load, matched_entropy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    split = "dev" if "dev" in os.path.basename(os.path.normpath(args.data)).lower() else "train"

    news, beh = load(args.data)
    cat = dict(zip(news.news_id, news.category))
    rng = np.random.default_rng(args.seed)

    # Data-quality checks (carried over from the team's phase1_diversity.py)
    abstracts = pd.read_csv(os.path.join(args.data, "news.tsv"), sep="\t", header=None,
                            usecols=[0, 4], names=["news_id", "abstract"], quoting=3)
    tokens = [t.rsplit("-", 1) for imp in beh.impressions for t in str(imp).split()]
    hist_ids = [n for h in beh.history for n in h.split()]
    quality = {
        "news_rows": len(news),
        "duplicate_news_ids": news.news_id.duplicated().sum(),
        "missing_category": news.category.isna().sum(),
        "missing_abstract": abstracts.abstract.isna().sum(),
        "behavior_rows": len(beh),
        "duplicate_impression_ids": beh.impression_id.duplicated().sum(),
        "users": beh.user_id.nunique(),
        "empty_history": (beh.history == "").sum(),
        "slots": len(tokens),
        "clicks": sum(label == "1" for _, label in tokens),
        "bad_click_labels": sum(label not in ("0", "1") for _, label in tokens),
        "slots_missing_from_news": sum(n not in cat for n, _ in tokens),
        "history_articles_missing_from_news": sum(n not in cat for n in hist_ids),
    }
    del tokens, hist_ids

    # Platform-wide slot mix = what a feed with no personalization would look like
    platform = Counter()
    for imp in beh.impressions:
        platform.update(cat[n] for n, _ in (t.rsplit("-", 1) for t in imp.split()) if n in cat)
    total = sum(platform.values())
    platform_share = {c: n / total for c, n in platform.items()}

    rows = []
    fam = Counter()  # (category, familiar?, clicked?) -> slots
    hist_mix, shown_mix = Counter(), Counter()  # paired impressions only
    for row in beh.itertuples(index=False):
        h = Counter(cat[n] for n in row.history.split() if n in cat)
        if not h:
            continue
        toks = [t.rsplit("-", 1) for t in row.impressions.split()]
        toks = [(n, label) for n, label in toks if n in cat]
        s = Counter(cat[n] for n, _ in toks)
        hist_mix.update(h)
        shown_mix.update(s)
        for n, label in toks:
            fam[(cat[n], cat[n] in h, label == "1")] += 1
        nh, ns = sum(h.values()), sum(s.values())
        k = min(nh, ns)
        rows.append({
            "n_history": nh, "n_slots": ns,
            "entropy_history": entropy(h), "entropy_impression": entropy(s),
            "entropy_history_matched": matched_entropy(h, k, rng, draws=5),
            "entropy_impression_matched": matched_entropy(s, k, rng, draws=5),
            "share_in_history": sum(v for c, v in s.items() if c in h) / ns,
            "baseline_in_history": sum(platform_share.get(c, 0) for c in h),
        })
    d = pd.DataFrame(rows)

    sfx = "" if split == "train" else f"_{split}"
    first, last = pd.to_datetime(beh.time, format="%m/%d/%Y %I:%M:%S %p").agg(["min", "max"])
    dates = (f"{first:%b} {first.day}, {first.year}" if first.date() == last.date()
             else f"{first:%b} {first.day}-{last.day}, {last.year}")
    kn = KeyNumbers(f"MIND-small {split}, {dates}")
    sample = f"{len(d):,} impressions with a reading history (all users)"
    for k, v in quality.items():
        kn.add(f"dq_{k}", v, f"Data quality: {k.replace('_', ' ')}", "whole split", "-")
    kn.add("ctr_overall", quality["clicks"] / quality["slots"], "Clicks / shown slots",
           "whole split", "-")

    # Topic mix of what readers clicked before vs what they were shown
    mix = pd.DataFrame({"history_share": pd.Series(hist_mix) / sum(hist_mix.values()),
                        "shown_share": pd.Series(shown_mix) / sum(shown_mix.values())}).fillna(0)
    mix["gap_points"] = (mix.shown_share - mix.history_share) * 100
    mix.index.name = "category"
    mix.sort_values("history_share", ascending=False).round(5).to_csv(
        os.path.join(args.out, f"history_vs_shown_mix{sfx}.csv"))
    kn.add("entropy_history_mean_raw", d.entropy_history.mean(),
           "Mean diversity of the click history (bits), unmatched (as in phase1_diversity.py)",
           sample, "-")
    kn.add("entropy_impression_mean_raw", d.entropy_impression.mean(),
           "Mean diversity of the shown list (bits), unmatched", sample, "-")
    kn.add("entropy_history_mean_matched", d.entropy_history_matched.mean(),
           "Mean history diversity, both sides sampled to the smaller size", sample, "-")
    kn.add("entropy_impression_mean_matched", d.entropy_impression_matched.mean(),
           "Mean shown-list diversity, both sides sampled to the smaller size", sample, "-")
    kn.add("share_impressions_broader_than_history",
           (d.entropy_impression_matched > d.entropy_history_matched).mean(),
           "Share of impressions more diverse than the user's history (size-matched)", sample, "-")
    kn.add("share_slots_in_history_categories", d.share_in_history.mean(),
           "Mean share of shown slots in a category the user has clicked before", sample, "-")
    kn.add("baseline_share_if_unpersonalized", d.baseline_in_history.mean(),
           "Same share if the feed followed the platform-wide category mix (no personalization)",
           sample, "-")
    kn.add("share_impressions_above_baseline", (d.share_in_history > d.baseline_in_history).mean(),
           "Share of impressions that lean towards the user's history more than the baseline",
           sample, "-")

    # Click rate on familiar vs new categories, overall and within each category
    f = pd.Series(fam).rename_axis(["category", "familiar", "clicked"]).unstack("clicked",
                                                                              fill_value=0)
    f.columns = ["not_clicked", "clicked"]
    f["slots"] = f.clicked + f.not_clicked
    tot = f.groupby(level="familiar")[["clicked", "slots"]].sum()
    ctr_fam, ctr_new = (tot.clicked / tot.slots)[True], (tot.clicked / tot.slots)[False]
    within = (f.clicked / f.slots).unstack("familiar").rename(columns={True: "ctr_familiar",
                                                                     False: "ctr_new"})
    n = f.slots.unstack("familiar")
    within["slots_familiar"], within["slots_new"] = n[True], n[False]
    within = within[(within.slots_familiar >= 500) & (within.slots_new >= 500)]
    within = within.sort_values("ctr_familiar")
    within.round(5).to_csv(os.path.join(args.out, f"ctr_familiar_vs_new{sfx}.csv"))
    kn.add("ctr_familiar", ctr_fam, "Click rate on slots in a category already in the history",
           f"{int(tot.slots[True]):,} slots", f"ctr_familiar_vs_new{sfx}.csv")
    kn.add("ctr_new", ctr_new, "Click rate on slots in a category not in the history",
           f"{int(tot.slots[False]):,} slots", f"ctr_familiar_vs_new{sfx}.csv")
    kn.add("categories_familiar_ctr_higher", (within.ctr_familiar > within.ctr_new).sum(),
           f"Categories (of {len(within)} with 500+ slots on both sides) where familiar slots "
           "are clicked more", f"{len(within)} categories", f"ctr_familiar_vs_new{sfx}.csv")
    kn.print()
    kn.save(os.path.join(args.out, f"key_numbers_crosscheck{sfx}.csv"))

    # Chart 9: within-category click rate, familiar vs new (dumbbell)
    chart_style.apply()
    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(within))
    ax.hlines(y, within.ctr_new, within.ctr_familiar, color=chart_style.GRID, lw=3, zorder=1)
    ax.plot(within.ctr_new, y, "o", ms=8, color=chart_style.ORANGE, mec=chart_style.SURFACE,
            mew=2, label="Topic new to the reader", zorder=2)
    ax.plot(within.ctr_familiar, y, "o", ms=8, color=chart_style.BLUE, mec=chart_style.SURFACE,
            mew=2, label="Topic the reader already clicks", zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(within.index)
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.set_xlabel("Click-through rate")
    chart_style.value_grid(ax, "x")
    ax.set_title(f"Familiar topics get clicked more: {ctr_fam:.1%} vs {ctr_new:.1%} "
                 f"({(within.ctr_familiar > within.ctr_new).sum()} of {len(within)} topics)")
    ax.legend(loc="lower right")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.text(0.01, 0.01, f"Source: MIND-small {split} set (Microsoft News, {dates}). {sample}; "
             "topics with 500+ slots on both sides.\nObservational: not the effect of changing "
             "the recommender.", fontsize=7, color=chart_style.MUTED)
    fig.savefig(os.path.join(args.out, f"09_ctr_familiar_vs_new{sfx}.png"), dpi=200)
    plt.close(fig)
    print("Saved outputs to", args.out)


if __name__ == "__main__":
    main()
