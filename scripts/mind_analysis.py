"""
Echo Breakers - Phase 1 starter: MIND-small exposure analysis.

Usage:
    python scripts/mind_analysis.py --train data/MINDsmall_train --out results

Expects the unzipped MINDsmall_train folder containing news.tsv and behaviors.tsv.
Download: https://huggingface.co/datasets/Recommenders/MIND (MINDsmall_train.zip / MINDsmall_dev.zip;
the official Azure links on https://msnews.github.io/ are closed)

Outputs (in --out):
    user_metrics.csv       per-user diversity metrics
    category_summary.csv   catalog share, shown vs clicked share, CTR and catalog coverage per category
    group_exposure_ratio.csv  per reader group: category shown share / platform share
    group_boost.csv        per reader group: own-category boost (per-user median/mean, pooled)
    key_numbers_mind.csv   every headline figure with its definition and sample size
    01_category_exposure.png, 02_entropy_shown_vs_clicked.png, 03_user_groups.png,
    04_group_exposure_gap.png, 05_personalization_lift.png, 06_popularity_concentration.png
"""
import argparse
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

import chart_style


def entropy(counts):
    """Shannon entropy (bits) of a Counter or list of counts."""
    c = np.array(list(counts.values()) if isinstance(counts, dict) else counts, dtype=float)
    c = c[c > 0]
    if c.size == 0:
        return np.nan
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def gini(values):
    """Gini coefficient: 0 = equal exposure, 1 = one item gets everything."""
    v = np.sort(np.array(values, dtype=float))
    if v.size == 0 or v.sum() == 0:
        return np.nan
    n = v.size
    idx = np.arange(1, n + 1)
    return float((2 * (idx * v).sum()) / (n * v.sum()) - (n + 1) / n)


def matched_entropy(counts, k, rng, draws=20):
    """Mean entropy of k items drawn without replacement from `counts`.

    Entropy grows with sample size, so comparing a user's few clicks with
    hundreds of shown items is unfair. This shrinks the bigger pool to the
    same size as the clicks before comparing.
    """
    c = np.array(list(counts.values()), dtype=np.int64)
    if k < 1 or c.sum() < k:
        return np.nan
    samples = rng.multivariate_hypergeometric(c, k, size=draws).astype(float)
    p = samples / k
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(p > 0, p * np.log2(p), 0.0).sum(axis=1)
    return float(h.mean())


def top_share(values, frac=0.10):
    """Share of the total that goes to the top `frac` of items."""
    v = np.sort(np.array(values, dtype=float))[::-1]
    if v.size == 0 or v.sum() == 0:
        return np.nan
    n_top = max(1, int(np.ceil(frac * v.size)))
    return float(v[:n_top].sum() / v.sum())


SOURCE = "Source: MIND-small (Microsoft News). "  # set in main() from the data's dates


def save(fig, out, name, sample_note):
    """Save a chart with its data source and sample size underneath."""
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(0.01, 0.01, SOURCE + sample_note, fontsize=7, color=chart_style.MUTED)
    fig.savefig(os.path.join(out, name), dpi=200)
    plt.close(fig)


class KeyNumbers:
    """Every figure that may appear in the PDF, with its definition and sample."""

    def __init__(self, dataset):
        self.dataset, self.rows = dataset, []

    def add(self, key, value, definition, sample, source_file):
        self.rows.append({"key": key, "value": round(float(value), 4), "definition": definition,
                          "sample": sample, "dataset": self.dataset, "source_file": source_file})

    def print(self):
        for r in self.rows:
            v = r["value"]
            shown = f"{v:,.0f}" if v == int(v) and abs(v) >= 1 else f"{v:.4g}"
            print(f"  {r['key']:<40} {shown:>10}   [{r['sample']}]")

    def save(self, path):
        pd.DataFrame(self.rows).to_csv(path, index=False)


def load(train_dir):
    news = pd.read_csv(
        os.path.join(train_dir, "news.tsv"), sep="\t", header=None,
        names=["news_id", "category", "subcategory", "title", "abstract",
               "url", "title_entities", "abstract_entities"],
        usecols=["news_id", "category", "subcategory", "title"],
    )
    beh = pd.read_csv(
        os.path.join(train_dir, "behaviors.tsv"), sep="\t", header=None,
        names=["impression_id", "user_id", "time", "history", "impressions"],
    )
    beh["history"] = beh["history"].fillna("")
    return news, beh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--max_users", type=int, default=0,
                    help="random sample of users for speed; 0 = all users")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_group", type=int, default=200,
                    help="smallest reader group shown in group charts")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    news, beh = load(args.train)
    cat = dict(zip(news.news_id, news.category))

    # Sample users for speed (document this in your methodology)
    users = beh.user_id.drop_duplicates()
    n_users = len(users)
    # Label charts and key numbers with the split and its actual dates
    global SOURCE
    split = "dev" if "dev" in os.path.basename(os.path.normpath(args.train)).lower() else "train"
    t = pd.to_datetime(beh.time, format="%m/%d/%Y %I:%M:%S %p")
    first, last = t.min(), t.max()
    dates = (f"{first:%b} {first.day}, {first.year}" if first.date() == last.date()
             else f"{first:%b} {first.day}-{last.day}, {last.year}")
    SOURCE = f"Source: MIND-small {split} set (Microsoft News, {dates}). "
    if 0 < args.max_users < len(users):
        keep = users.sample(args.max_users, random_state=args.seed)
        beh = beh[beh.user_id.isin(keep)]

    # Per-user counters: history (past reading), shown, clicked
    hist, shown, clicked = {}, {}, {}
    item_shown, item_clicked = Counter(), Counter()
    seen_hist = set()
    for row in beh.itertuples(index=False):
        u = row.user_id
        if u not in seen_hist:  # history is repeated across impressions; count once
            seen_hist.add(u)
            hist[u] = Counter(cat.get(n) for n in row.history.split() if n in cat)
        shown.setdefault(u, Counter())
        clicked.setdefault(u, Counter())
        for tok in str(row.impressions).split():
            nid, label = tok.rsplit("-", 1)
            c = cat.get(nid)
            if c is None:
                continue
            shown[u][c] += 1
            item_shown[nid] += 1
            if label == "1":
                clicked[u][c] += 1
                item_clicked[nid] += 1

    # Per-user metrics
    rng = np.random.default_rng(args.seed)
    rows = []
    for u in shown:
        h, s, k = hist.get(u, Counter()), shown[u], clicked[u]
        n_k = sum(k.values())
        top = h.most_common(1)[0] if h else (None, 0)
        rows.append({
            "user_id": u,
            "n_history": sum(h.values()),
            "n_shown": sum(s.values()),
            "n_clicked": n_k,
            "entropy_history": entropy(h),
            "entropy_shown": entropy(s),
            "entropy_clicked": entropy(k),
            # Same sample size as clicks, so the comparison is fair
            "entropy_shown_matched": matched_entropy(s, n_k, rng),
            "entropy_history_matched": matched_entropy(h, n_k, rng),
            # Fixed 10-article sample so groups with more clicks/history aren't favoured
            "entropy_history_k10": matched_entropy(h, 10, rng),
            "top_history_category": top[0],
            "top_history_share": top[1] / sum(h.values()) if h else np.nan,
            "feed_top_share": s[top[0]] / sum(s.values()) if top[0] else np.nan,
        })
    um = pd.DataFrame(rows)

    # Category summary: shown vs clicked share, click-through rate, catalog coverage
    tot_shown, tot_clicked = Counter(), Counter()
    for u in shown:
        tot_shown.update(shown[u])
        tot_clicked.update(clicked[u])
    cs = pd.DataFrame({"shown": pd.Series(tot_shown), "clicked": pd.Series(tot_clicked)}).fillna(0)
    cs["shown_share"] = cs.shown / cs.shown.sum()
    cs["clicked_share"] = cs.clicked / cs.clicked.sum()
    cs["ctr"] = cs.clicked / cs.shown
    cs["catalog_articles"] = news.category.value_counts()
    cs["catalog_share"] = cs.catalog_articles / len(news)
    cs["articles_shown"] = pd.Series(Counter(cat[n] for n in item_shown))
    cs["catalog_coverage"] = cs.articles_shown / cs.catalog_articles
    cs = cs.sort_values("shown_share", ascending=False)
    cs.to_csv(os.path.join(args.out, "category_summary.csv"))

    # Personalization strength: how much more of a user's favourite category
    # their feed contains, compared with the platform-wide feed (1 = no boost)
    um["feed_top_lift"] = um.feed_top_share / um.top_history_category.map(cs.shown_share)
    um.to_csv(os.path.join(args.out, "user_metrics.csv"), index=False)

    # What each reader group sees more / less of than the platform average
    g = um.dropna(subset=["top_history_category"])
    g = g[g.n_history >= 10]
    sizes = g.top_history_category.value_counts()
    groups = sizes[sizes >= args.min_group].index
    gx = {}
    for grp_name in groups:
        tot = Counter()
        for u in g.user_id[g.top_history_category == grp_name]:
            tot.update(shown[u])
        gx[grp_name] = pd.Series(tot) / sum(tot.values())
    main_cats = cs.index[cs.shown >= 1000]  # drop near-empty categories (kids, northamerica)
    gx = pd.DataFrame(gx).T.reindex(columns=main_cats).fillna(0)
    ratio = gx / cs.shown_share[main_cats]
    ratio.index.name = "reader_group"
    ratio.round(3).to_csv(os.path.join(args.out, "group_exposure_ratio.csv"))

    # Boost of each group's own category, two definitions:
    #   pooled   = group's total feed share of the category / platform share (heatmap diagonal)
    #   per-user = median / mean of each member's own feed_top_lift (more conservative)
    gm = g[g.top_history_category.isin(groups)]
    gs = gm.groupby("top_history_category").agg(
        users=("user_id", "count"),
        boost_per_user_median=("feed_top_lift", "median"),
        boost_per_user_mean=("feed_top_lift", "mean"),
    )
    gs["boost_pooled"] = [ratio.at[c, c] if c in ratio.columns else np.nan for c in gs.index]
    gs = gs.sort_values("boost_per_user_median", ascending=False)
    gs.index.name = "reader_group"
    gs.round(3).to_csv(os.path.join(args.out, "group_boost.csv"))

    # Headline numbers: printed, and saved with definitions so every figure in the
    # PDF can be checked against one table
    kn = KeyNumbers(f"MIND-small {split}, {dates}")
    n_shown_art, n_clicked_art = len(item_shown), len(item_clicked)
    m = um[um.n_clicked >= 3].dropna(subset=["entropy_shown_matched", "entropy_clicked"])
    mh = m.dropna(subset=["entropy_history_matched"])
    p = um[um.n_history >= 10].dropna(subset=["feed_top_lift"])
    catalog_imps = list(item_shown.values()) + [0] * (len(news) - n_shown_art)
    users_all = (f"{len(um):,} sampled users (seed {args.seed})" if 0 < args.max_users < n_users
                 else f"all {len(um):,} users")
    users_10 = f"{len(p):,} users with 10+ articles read"

    kn.add("users_sampled", len(um), "Users in the random sample", users_all, "user_metrics.csv")
    kn.add("popularity_share_top10pct_shown", top_share(list(item_shown.values())),
           f"Share of all impressions going to the top 10% of articles shown at least once "
           f"({int(np.ceil(0.1 * n_shown_art)):,} of {n_shown_art:,} articles)", users_all, "-")
    kn.add("popularity_share_top10pct_catalog", top_share(catalog_imps),
           f"Same, over the whole {len(news):,}-article catalog (never-shown = 0). Do not headline:"
           " inflated by old history-only articles", users_all, "-")
    kn.add("popularity_share_top10pct_clicked", top_share(list(item_clicked.values())),
           f"Share of all clicks going to the top 10% of the {n_clicked_art:,} clicked articles",
           users_all, "-")
    kn.add("gini_shown", gini(list(item_shown.values())),
           f"Gini of impressions across the {n_shown_art:,} articles shown at least once",
           users_all, "-")
    kn.add("gini_clicked", gini(list(item_clicked.values())),
           f"Gini of clicks across the {n_clicked_art:,} articles clicked at least once",
           users_all, "-")
    # news.tsv also lists history-only articles, so this is below 1 by construction
    kn.add("catalog_coverage", n_shown_art / len(news),
           "Share of news.tsv articles shown at least once (news.tsv includes history-only articles)",
           users_all, "category_summary.csv")
    kn.add("entropy_feed_matched_median", m.entropy_shown_matched.median(),
           "Median feed diversity (bits), feed sampled down to each user's click count",
           f"{len(m):,} users with 3+ clicks", "user_metrics.csv")
    kn.add("entropy_clicked_median", m.entropy_clicked.median(),
           "Median click diversity (bits)", f"{len(m):,} users with 3+ clicks", "user_metrics.csv")
    kn.add("share_clicks_narrower_than_feed",
           (m.entropy_clicked < m.entropy_shown_matched).mean(),
           "Share of users whose clicks are less diverse than their size-matched feed",
           f"{len(m):,} users with 3+ clicks", "user_metrics.csv")
    kn.add("entropy_history_matched_median", mh.entropy_history_matched.median(),
           "Median reading-history diversity (bits), sampled down to each user's click count",
           f"{len(mh):,} users with 3+ clicks and enough history", "user_metrics.csv")
    kn.add("boost_median", p.feed_top_lift.median(),
           "Median ratio: favourite category's share of the user's feed / its platform-wide share",
           users_10, "user_metrics.csv (feed_top_lift)")
    kn.add("share_users_boosted", (p.feed_top_lift > 1).mean(),
           "Share of users whose feed over-represents their favourite category", users_10,
           "user_metrics.csv")
    kn.add("share_users_favourite_never_shown", (p.feed_top_lift == 0).mean(),
           "Share of users never shown their favourite category", users_10, "user_metrics.csv")
    for grp_name, row in gs.iterrows():
        kn.add(f"boost_median_{grp_name}", row.boost_per_user_median,
               f"Per-user median boost for {grp_name} readers (pooled ratio: "
               f"{row.boost_pooled:.2f})" + (" - small group, do not headline"
                                            if row.users < 250 else ""),
               f"{int(row.users):,} {grp_name} readers with 10+ articles read", "group_boost.csv")
    hk = gm.groupby("top_history_category").entropy_history_k10.median()
    for grp_name, v in hk.items():
        kn.add(f"history_entropy_k10_{grp_name}", v,
               f"Median reading-history diversity of {grp_name} readers (bits, 10-article samples)",
               f"{(gm.top_history_category == grp_name).sum():,} {grp_name} readers",
               "user_metrics.csv")
    kn.print()
    kn.save(os.path.join(args.out, "key_numbers_mind.csv"))
    print(f"Per-group boost of own category (n={len(gm)}):")
    print(gs.round(2).to_string())
    print(f"Group exposure gaps (n={len(gm)}; shown share vs platform, <0.8 = under-shown):")
    for grp_name, row in ratio.iterrows():
        low = row[row < 0.8].sort_values()
        print(f"  {grp_name:<14} under-shown: "
              + (", ".join(f"{c} {v:.2f}" for c, v in low.items()) or "none"))

    chart_style.apply()

    # Chart 1: shown vs clicked share by category
    c1 = cs[cs.shown >= 1000]  # drop near-empty categories (kids, northamerica)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(c1))
    ax.bar(x - 0.2, c1.shown_share, 0.4, label="Shown", color=chart_style.BLUE)
    ax.bar(x + 0.2, c1.clicked_share, 0.4, label="Clicked", color=chart_style.ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels(c1.index, rotation=45, ha="right")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_ylabel("Share of all impressions / clicks")
    chart_style.value_grid(ax)
    ax.set_title(f"News is {c1.shown_share.iloc[0]:.0%} of what readers are shown "
                 f"and {c1.clicked_share.iloc[0]:.0%} of what they click")
    ax.legend()
    save(fig, args.out, "01_category_exposure.png", f"{len(um):,} sampled users, all impressions.")

    # Chart 2: entropy of shown (size-matched to clicks) vs clicked per user
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 3.5, 36)
    ax.hist(m.entropy_shown_matched, bins=bins, histtype="step", lw=2,
            color=chart_style.BLUE, label="Feed (sampled to the same size as clicks)")
    ax.hist(m.entropy_clicked, bins=bins, histtype="step", lw=2,
            color=chart_style.ORANGE, label="Clicks")
    ax.set_xlabel("Topic diversity (bits; higher = more varied)")
    ax.set_ylabel("Users")
    chart_style.value_grid(ax)
    ax.set_title(f"Clicks are only slightly narrower than the feed (median "
                 f"{m.entropy_clicked.median():.2f} vs {m.entropy_shown_matched.median():.2f} bits)")
    ax.legend(loc="upper right")
    save(fig, args.out, "02_entropy_shown_vs_clicked.png",
         f"{len(m):,} users with 3+ clicks; feed diversity sampled to each user's click count.")

    # Chart 3: behaviour-based groups, reading-diet breadth on a fixed 10-article sample
    grp = gm.groupby("top_history_category").agg(
        users=("user_id", "count"),
        median_entropy=("entropy_history_k10", "median"),
    ).sort_values("median_entropy")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [chart_style.BLUE if i < 2 else chart_style.GREY for i in range(len(grp))]
    bars = ax.barh(grp.index[::-1], grp.median_entropy[::-1], color=colors[::-1], height=0.7)
    for b, v in zip(bars, grp.median_entropy[::-1]):
        ax.text(v + 0.02, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center",
                fontsize=8, color=chart_style.INK_2)
    ax.set_xlabel("Median reading-history diversity, 10-article samples (bits)")
    ax.set_ylabel("Reader group (most-read category)")
    ax.set_title(f"{grp.index[0].capitalize()} and {grp.index[1]} readers have the "
                 "narrowest reading habits")
    save(fig, args.out, "03_user_groups.png",
         f"{len(gm):,} users with 10+ articles read, in groups of {args.min_group}+; 10-article samples.")

    # Chart 4: heatmap of what each group is shown vs the platform average
    r = ratio.loc[gs.index]  # strongest per-user boost at the top
    lr = np.log2(r.clip(lower=1 / 8))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    im = ax.imshow(lr, cmap=chart_style.DIVERGING, vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(r.columns)))
    ax.set_xticklabels(r.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(r.index)))
    ax.set_yticklabels(r.index)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            ax.text(j, i, f"{r.iat[i, j]:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(lr.iat[i, j]) > 1.2 else chart_style.INK)
    cb = fig.colorbar(im, ax=ax, ticks=[-2, -1, 0, 1, 2], shrink=0.8)
    cb.ax.set_yticklabels(["¼× (less)", "½×", "1× average", "2×", "4× (more)"])
    cb.outline.set_visible(False)
    ax.set_xlabel("Topic shown in the feed")
    ax.set_ylabel("Reader group (most-read category)")
    ax.set_title("Each group's feed leans towards its own topic; other topics shrink slightly")
    save(fig, args.out, "04_group_exposure_gap.png",
         f"{len(gm):,} users with 10+ articles read, in groups of {args.min_group}+. Cells = group's combined "
         "feed share / platform share (pooled).")

    # Chart 5: personalization lift of each user's favourite category
    zero = (p.feed_top_lift == 0).sum()  # favourite category never shown at all
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(np.log2(p.feed_top_lift[p.feed_top_lift > 0].clip(1 / 16, 16)), bins=40,
            color=chart_style.BLUE)
    ax.axvline(0, color=chart_style.INK_2, lw=1, ls="--")
    ax.text(0.02, 0.95, f"{zero:,} users ({zero / len(p):.1%}) were never shown\n"
            "their favourite topic (not plotted;\nmostly readers with very small feeds)",
            transform=ax.transAxes, va="top", fontsize=9, color=chart_style.INK_2)
    ax.set_xticks([-2, -1, 0, 1, 2, 3])
    ax.set_xticklabels(["¼×", "½×", "1× (no boost)", "2×", "4×", "8×"])
    ax.set_xlabel("Favourite topic's share of the reader's feed vs. platform average")
    ax.set_ylabel("Users")
    chart_style.value_grid(ax)
    ax.set_title(f"{(p.feed_top_lift > 1).mean():.0%} of feeds lean towards the reader's "
                 f"favourite topic (median {p.feed_top_lift.median():.2f}×)")
    save(fig, args.out, "05_personalization_lift.png", f"{users_10}.")

    # Chart 6: concentration curve of article exposure
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for label, counter, color in [("Impressions", item_shown, chart_style.BLUE),
                                  ("Clicks", item_clicked, chart_style.ORANGE)]:
        v = np.sort(np.array(list(counter.values()), dtype=float))[::-1]
        xs, ys = np.arange(1, v.size + 1) / v.size, np.cumsum(v) / v.sum()
        ax.plot(xs, ys, label=label, color=color)
        y10 = top_share(v)
        ax.plot([0.1], [y10], "o", ms=7, color=color, mec=chart_style.SURFACE, mew=2)
        ax.text(0.12, y10 - 0.035, f"{y10:.0%} of {label.lower()}", color=chart_style.INK,
                fontsize=9, va="top")
    ax.plot([0, 1], [0, 1], color=chart_style.GREY, ls="--", lw=1, label="Equal exposure")
    ax.axvline(0.1, color=chart_style.AXIS, lw=0.8, ls=":")
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("Share of articles (most exposed first)")
    ax.set_ylabel("Cumulative share of exposure")
    chart_style.value_grid(ax, "both")
    ax.set_title(f"The top 10% of articles shown get {top_share(list(item_shown.values())):.0%}"
                 " of all impressions")
    ax.legend(loc="lower right")
    save(fig, args.out, "06_popularity_concentration.png",
         f"{n_shown_art:,} articles shown at least once; {n_clicked_art:,} clicked; {len(um):,} users.")

    print("Saved outputs to", args.out)


if __name__ == "__main__":
    main()
