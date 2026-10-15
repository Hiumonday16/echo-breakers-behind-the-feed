# Echo Breakers: Behind The Feed

Team Echo Breakers' entry for the **Behind The Feed Data Challenge**: who gets exposed to what
news, what they miss, and how to measure whether a feed is diverse and fair.

Phase 1 deliverable: a 5-page PDF visual summary (deadline Oct 18, 2026, 11:59 PM ET).

## Data (not in the repo)

| Dataset | Use | Where to get it |
|---|---|---|
| MIND-small, train (Nov 9–14, 2019) and dev (Nov 15) | Main analysis; dev = replication check | Official Azure links are closed. Use the Hugging Face mirror [Recommenders/MIND](https://huggingface.co/datasets/Recommenders/MIND) (`MINDsmall_train.zip`, `MINDsmall_dev.zip`) → `data/MINDsmall_train/`, `data/MINDsmall_dev/` (`news.tsv`, `behaviors.tsv`). Byte-identical to [huyva/MIND-small](https://huggingface.co/datasets/huyva/MIND-small). Microsoft Research License (research use). |
| GDELT 1.0 daily events, Oct 16 – Nov 14, 2019 | Geographic coverage | `https://data.gdeltproject.org/events/YYYYMMDD.export.CSV.zip` → `data/gdelt/raw/`; lookups (`CSV.header.dailyupdates.txt`, `FIPS.country.txt`) from https://www.gdeltproject.org/data/lookups/ → `data/gdelt/lookups/` |
| World Bank population 2019 + GeoNames country table | Coverage vs population (FIPS → ISO3) | `wb_pop_2019.json` from `https://api.worldbank.org/v2/country/all/indicator/SP.POP.TOTL?date=2019&format=json&per_page=500` and `https://download.geonames.org/export/dump/countryInfo.txt` → `data/population/` |

`data/` is git-ignored because the files are large. Each teammate downloads them locally.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install pandas numpy matplotlib
```

## Run

```powershell
python scripts/mind_analysis.py --train data/MINDsmall_train --out results
python scripts/mind_analysis.py --train data/MINDsmall_dev --out results/dev
python scripts/mind_crosscheck.py --data data/MINDsmall_train --out results
python scripts/mind_crosscheck.py --data data/MINDsmall_dev --out results/dev
python scripts/gdelt_analysis.py --raw data/gdelt/raw --lookups data/gdelt/lookups --population data/population --out results
```

Prints the headline numbers and writes to `results/`:

- `user_metrics.csv`: per-user diversity metrics
- `category_summary.csv`: shown vs. clicked share and CTR per category
- `group_exposure_ratio.csv`: per reader group, how often each category is shown vs. the platform average
- `gdelt_country_coverage.csv`: GDELT events and articles per country
- `gdelt_population_ratio.csv`: coverage share vs population share per country
- `key_numbers_mind.csv`, `key_numbers_gdelt.csv`: **every figure for the PDF**, with definition and sample size (reviewer checks against these)
- `key_numbers_crosscheck.csv`, `ctr_familiar_vs_new.csv`: history vs feed and familiar vs new topics
- Charts `01`–`09`: category exposure, entropy, reader groups, group exposure gaps, personalization lift, popularity concentration, GDELT country coverage, coverage vs population, familiar vs new click rate

Defaults: all 50,000 users (`--max_users N` samples N users with `--seed`, for speed). Dev outputs go to `results/dev/`.

`mind_crosscheck.py` merges in the team's first MIND analysis (formerly `MIND/analysis/phase1_diversity.py`): data-quality checks (`dq_*` rows), history-vs-shown topic mix, and familiar-vs-new click rates (chart `09`), plus size-matched entropy and a no-personalization baseline. All overlapping numbers were checked against that analysis's outputs (353 values, 0 mismatches).

## Metrics

| Metric | Question | Note |
|---|---|---|
| Shannon entropy of categories | How varied is one user's news diet? | Shown/history entropy is also computed **size-matched** to the number of clicks, because entropy rises with sample size |
| Gini coefficient | Is exposure spread evenly across articles? | 0 = equal, 1 = one article gets everything |
| Catalog coverage | How much of the catalog is ever shown? | `news.tsv` includes history-only articles, so this is below 1 by design |
| Popularity share | How much exposure goes to the top 10% of articles? | Computed over articles shown at least once (20,288 in train); the script also prints the whole-catalog version |
| Personalization boost | How much more of a reader's favourite category is in their feed than in the average feed? | Users with 10+ articles read. Per-user median is the headline; the pooled group ratio (heatmap diagonal) is higher |

## Layout

```
data/       raw datasets (git-ignored)
docs/       team plan (Oct 14 snapshot) and the MIND data licence (MSR_License_Data.pdf)
scripts/    analysis code (chart_style.py = shared palette and fonts)
results/    charts and CSVs produced by the scripts
report/     storyline draft and final PDF
phase2/     groundwork for Phase 2 (popularity-ranking baseline)
```
