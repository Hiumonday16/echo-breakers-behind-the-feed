"""
Echo Breakers - Phase 1: GDELT country coverage, to set beside the MIND analysis.

Usage:
    python scripts/gdelt_analysis.py --raw data/gdelt/raw --lookups data/gdelt/lookups --out results

Expects GDELT 1.0 daily event files (YYYYMMDD.export.CSV.zip) in --raw, and in --lookups:
    CSV.header.dailyupdates.txt  https://www.gdeltproject.org/data/lookups/CSV.header.dailyupdates.txt
    FIPS.country.txt             https://www.gdeltproject.org/data/lookups/FIPS.country.txt
Daily files: https://data.gdeltproject.org/events/YYYYMMDD.export.CSV.zip

Outputs (in --out):
    gdelt_country_coverage.csv     events and articles per country (FIPS code + name)
    gdelt_population_ratio.csv     coverage share vs 2019 population share (with --population)
    key_numbers_gdelt.csv          every headline figure with its definition and sample size
    07_gdelt_country_coverage.png  top countries vs. everyone else
    08_gdelt_coverage_vs_population.png  most populous countries (with --population)

Population (optional, --population folder):
    countryInfo.txt   https://download.geonames.org/export/dump/countryInfo.txt (FIPS -> ISO3)
    wb_pop_2019.json  https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL?date=2019&format=json&per_page=500
"""
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.ticker import PercentFormatter

import chart_style
from mind_analysis import KeyNumbers, gini, top_share

COLS = ["GLOBALEVENTID", "SQLDATE", "NumArticles", "ActionGeo_CountryCode"]


def load_header(lookups):
    with open(os.path.join(lookups, "CSV.header.dailyupdates.txt"), encoding="utf-8") as f:
        return f.read().strip().split("\t")


def check_columns(path, header):
    """Fail loudly if the file layout doesn't match the codebook header."""
    sample = pd.read_csv(path, sep="\t", header=None, nrows=2000, dtype=str)
    if sample.shape[1] != len(header):
        raise ValueError(f"{path}: {sample.shape[1]} columns, header has {len(header)}")
    sample.columns = header
    if not pd.to_numeric(sample.NumArticles, errors="coerce").notna().all():
        raise ValueError(f"{path}: NumArticles is not numeric - columns may be shifted")
    codes = sample.ActionGeo_CountryCode.dropna()
    if not codes.str.fullmatch(r"[A-Z]{2}").all():
        raise ValueError(f"{path}: ActionGeo_CountryCode is not 2-letter FIPS - columns may be shifted")


def save(fig, path, footer):
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(0.01, 0.01, footer, fontsize=7, color=chart_style.MUTED, wrap=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def population_view(cc, pop_dir):
    """Coverage share vs 2019 population share, per country.

    GDELT uses FIPS 10-4 codes; GeoNames' countryInfo.txt maps FIPS -> ISO3,
    which the World Bank population file uses.
    """
    geo = pd.read_csv(os.path.join(pop_dir, "countryInfo.txt"), sep="\t", comment="#",
                      header=None, dtype=str, keep_default_na=False, usecols=[1, 3, 4])
    geo.columns = ["iso3", "fips", "name"]
    fips_to_iso3 = dict(zip(geo.fips, geo.iso3))
    iso3_to_name = dict(zip(geo.iso3, geo.name))  # cleaner labels than GDELT's lookup
    with open(os.path.join(pop_dir, "wb_pop_2019.json"), encoding="utf-8") as f:
        wb = json.load(f)[1]
    pop_by_iso3 = {r["countryiso3code"]: r["value"] for r in wb if r["value"]}

    df = cc[["country", "articles"]].copy()
    df["iso3"] = df.index.map(fips_to_iso3)
    df["population"] = df.iso3.map(pop_by_iso3)
    matched = df.dropna(subset=["population"])
    matched_share = matched.articles.sum() / df.articles.sum()
    missing = df[df.population.isna()].sort_values("articles", ascending=False)
    print(f"Population match: {len(matched)} of {len(df)} countries/territories, "
          f"{matched_share:.1%} of articles. Largest unmatched: "
          + ", ".join(f"{c} ({a:,})" for c, a in zip(missing.country.head(5), missing.articles.head(5))))
    matched = matched.copy()
    matched["country"] = matched.iso3.map(iso3_to_name)
    matched["coverage_share"] = matched.articles / matched.articles.sum()
    matched["pop_share"] = matched.population / matched.population.sum()
    matched["coverage_ratio"] = matched.coverage_share / matched.pop_share
    return matched, matched_share


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--lookups", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--top", type=int, default=15, help="countries to show in the chart")
    ap.add_argument("--population", help="folder with countryInfo.txt and wb_pop_2019.json "
                    "(optional: adds coverage vs population)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    header = load_header(args.lookups)
    idx = [header.index(c) for c in COLS]
    print("Columns used (1-indexed):", {c: i + 1 for c, i in zip(COLS, idx)},
          f"of {len(header)}")
    names = pd.read_csv(os.path.join(args.lookups, "FIPS.country.txt"), sep="\t",
                        header=None, names=["fips", "country"], dtype=str,
                        keep_default_na=False)  # "NA" is Netherlands Antilles, not missing
    names = dict(zip(names.fips.str.strip(), names.country.str.strip()))

    files = sorted(glob.glob(os.path.join(args.raw, "*.export.CSV.zip")))
    parts = []
    for path in files:
        check_columns(path, header)
        df = pd.read_csv(path, sep="\t", header=None, usecols=idx, dtype=str)
        df.columns = [header[i] for i in sorted(idx)]
        parts.append(df)
    ev = pd.concat(parts, ignore_index=True)
    print(f"Files: {len(files)} ({os.path.basename(files[0])[:8]} to "
          f"{os.path.basename(files[-1])[:8]}, by date added)")

    # Drop duplicates and rows without a usable country, and say how many
    n_raw = len(ev)
    ev = ev.drop_duplicates("GLOBALEVENTID")
    n_dupes = n_raw - len(ev)
    no_geo = ev.ActionGeo_CountryCode.isna()
    unknown = ~no_geo & ~ev.ActionGeo_CountryCode.isin(names)
    print(f"Rows: {n_raw:,} | duplicate event IDs dropped: {n_dupes:,} "
          f"| no country code: {no_geo.sum():,} ({no_geo.mean():.1%}) "
          f"| code not in FIPS lookup: {unknown.sum():,} ({unknown.mean():.2%})")
    ev = ev[~no_geo & ~unknown]
    ev["NumArticles"] = ev.NumArticles.astype(int)
    print(f"Events kept: {len(ev):,}")

    cc = ev.groupby("ActionGeo_CountryCode").agg(
        events=("GLOBALEVENTID", "count"), articles=("NumArticles", "sum"))
    cc.index.name = "fips"
    cc.insert(0, "country", cc.index.map(names))
    cc["article_share"] = cc.articles / cc.articles.sum()
    cc["event_share"] = cc.events / cc.events.sum()
    cc = cc.sort_values("articles", ascending=False)
    cc.to_csv(os.path.join(args.out, "gdelt_country_coverage.csv"))

    # Same concentration metrics as MIND, on both counts so the result
    # doesn't hinge on one choice
    window = f"{os.path.basename(files[0])[:8]}-{os.path.basename(files[-1])[:8]}"
    kn = KeyNumbers(f"GDELT 1.0 daily events, {window}")
    n = len(cc)
    sample = f"{len(ev):,} events with a known location, {n} countries/territories"
    kn.add("rows_no_country_share", no_geo.mean(), "Share of event rows dropped: no country code",
           f"{n_raw:,} event rows", "-")
    kn.add("rows_unknown_code_share", unknown.mean(),
           "Share of event rows dropped: code not in GDELT's FIPS lookup", f"{n_raw:,} event rows", "-")
    for col in ["articles", "events"]:
        v = cc[col].values
        kn.add(f"top10_countries_share_{col}", np.sort(v)[::-1][:10].sum() / v.sum(),
               f"Share of all {col} about the 10 most-covered countries", sample,
               "gdelt_country_coverage.csv")
        kn.add(f"top10pct_countries_share_{col}", top_share(v),
               f"Share of all {col} about the top 10% of countries/territories "
               f"({int(np.ceil(0.1 * n))} of {n})", sample, "gdelt_country_coverage.csv")
        kn.add(f"bottom_half_share_{col}", np.sort(v)[: n // 2].sum() / v.sum(),
               f"Share of all {col} about the least-covered half ({n // 2}) of countries/territories",
               sample, "gdelt_country_coverage.csv")
        kn.add(f"gini_countries_{col}", gini(v), f"Gini of {col} across countries/territories",
               sample, "gdelt_country_coverage.csv")
    kn.add("us_article_share", cc.article_share.get("US", np.nan),
           "United States' share of all articles", sample, "gdelt_country_coverage.csv")

    pop = None
    if args.population:
        pop, matched_share = population_view(cc, args.population)
        pop.sort_values("coverage_ratio").round(6).to_csv(
            os.path.join(args.out, "gdelt_population_ratio.csv"))
        big = pop[pop.population >= 1e6]
        top10 = cc.index[:10]
        kn.add("articles_matched_to_population", matched_share,
               "Share of all articles whose country matched a 2019 World Bank population",
               sample, "gdelt_population_ratio.csv")
        kn.add("top10_countries_population_share",
               pop.population[pop.index.isin(top10)].sum() / pop.population.sum(),
               "Share of world population (matched countries) living in the 10 most-covered "
               "countries", f"{len(pop)} matched countries", "gdelt_population_ratio.csv")
        under = big.coverage_ratio < 0.5
        kn.add("population_share_in_undercovered",
               big.population[under].sum() / pop.population.sum(),
               "Share of world population living in countries (1M+ people) that get less than "
               "half their population share of coverage", f"{len(big)} countries with 1M+ people",
               "gdelt_population_ratio.csv")
        for f in ["US", "IN", "CH", "NI", "ID", "BR"]:
            if f in pop.index:
                kn.add(f"coverage_ratio_{f}", pop.coverage_ratio[f],
                       f"{pop.country[f]}: coverage share / population share",
                       f"{len(pop)} matched countries", "gdelt_population_ratio.csv")
    kn.print()
    kn.save(os.path.join(args.out, "key_numbers_gdelt.csv"))
    print("Top 10 by articles:")
    print(cc.head(10)[["country", "articles", "article_share", "event_share"]]
          .round(3).to_string())

    chart_style.apply()
    footer = (f"Source: GDELT 1.0 daily events, {window} ({len(files)} days). {len(ev):,} events "
              "with a known location; articles = NumArticles; country = where the event took place.")

    # Chart 7: top countries vs. all the rest
    top = cc.head(args.top)
    rest = cc.article_share.iloc[args.top:].sum()
    labels = list(top.country) + [f"All other {n - args.top} countries/territories"]
    vals = list(top.article_share) + [rest]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = [chart_style.GREY] + [chart_style.BLUE] * args.top
    bars = ax.barh(labels[::-1], vals[::-1], color=colors, height=0.7)
    for b, v in zip(bars, vals[::-1]):
        ax.text(b.get_width() + 0.005, b.get_y() + b.get_height() / 2, f"{v:.1%}",
                va="center", fontsize=8, color=chart_style.INK_2)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("Share of all news articles about events, by country of the event")
    top10 = cc.article_share.head(10).sum()
    ax.set_title(f"10 countries get {top10:.0%} of the news coverage GDELT tracks")
    ax.set_xlim(0, max(vals) * 1.15)
    save(fig, os.path.join(args.out, "07_gdelt_country_coverage.png"), footer)

    # Chart 8: coverage share vs population share, most populous countries
    if pop is not None:
        pp = pop.sort_values("population", ascending=False).head(args.top)
        fig, ax = plt.subplots(figsize=(8, 5.5))
        y = np.arange(len(pp))[::-1]
        ax.barh(y + 0.2, pp.pop_share, 0.4, color=chart_style.GREY, label="Share of world population")
        ax.barh(y - 0.2, pp.coverage_share, 0.4, color=chart_style.BLUE, label="Share of news coverage")
        for yi, (_, row) in zip(y, pp.iterrows()):
            ax.text(max(row.pop_share, row.coverage_share) + 0.005, yi,
                    f"{row.coverage_ratio:.2f}×", va="center", fontsize=8, color=chart_style.INK_2)
        ax.set_yticks(y)
        ax.set_yticklabels(pp.country)
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_xlabel("Share (labels: coverage share ÷ population share)")
        chart_style.value_grid(ax, "x")
        ax.set_title(f"{pp.country.iloc[0]} has {pp.pop_share.iloc[0]:.0%} of the world's people "
                     f"but {pp.coverage_share.iloc[0]:.0%} of its news coverage")
        ax.legend(loc="lower right")
        save(fig, os.path.join(args.out, "08_gdelt_coverage_vs_population.png"),
             footer + f" Population: World Bank 2019 (SP.POP.TOTL); {len(pop)} matched countries.")

    print("Saved outputs to", args.out)


if __name__ == "__main__":
    main()
