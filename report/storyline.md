# Echo Breakers – Phase 1 storyline and page copy (draft v2, merged)

**Status:** draft for team review, Oct 14, 2026. This version merges our analysis (`scripts/`)
with the team's MIND analysis (`MIND/analysis/`), and is rerun on **all 50,000 users**. The
**dev day (Nov 15)** is used as a replication check. Every number comes from a `results/`
key-numbers file, and its key is shown in `[brackets]` for the reviewer:

- `key_numbers_mind.csv` (MIND train), `dev/key_numbers_mind.csv` (dev)
- `key_numbers_crosscheck.csv`, `dev/key_numbers_crosscheck_dev.csv` (history vs feed, familiar vs new)
- `key_numbers_gdelt.csv` (GDELT and population)

Remove the brackets in the final PDF.

## The story in one sentence

Feeds are broad but tilted: they show readers more topics than they read, yet a few articles
and a few countries take most of the attention, and each reader's favourite topic gets
extra space. Familiar topics also get more clicks, so the system has little reason to
change on its own.

## Page plan

| Page | Headline (the finding) | Charts | Backup |
|---|---|---|---|
| 1. Hook | The top 10% of articles get 91% of what people see | Hero number; small `06` | |
| 2. Data & method | How we measured diversity and fairness | Metrics table; small `02` | |
| 3. Who sees what (MIND) | Broad but tilted: feeds add topics, but lean towards each reader's favourite, and familiar topics win the clicks | `04` (main), `09` (small) | `05`, `01`, `03` |
| 4. Where news looks (GDELT) | 10% of countries get 83% of coverage; 70% of the world lives in under-covered countries | `08` (main), `07` (small) | |
| 5. Takeaways & limitations | Three findings, honest limits, what we build next | None (text) | |

---

## Page 1 – Hook

**Headline:** The top 10% of articles get 91% of what people see.

**Body (≈40 words):**
On Microsoft News in November 2019, 1 in 10 articles took 91% of all impressions
[popularity_share_top10pct_shown]. The next day showed the same thing (92%, dev). Recommendation
systems decide what we see. We asked: who gets exposed to what, and what are they missing?

**Visual:** "91%" as a hero number, with a small version of chart `06`.

**Small print:** Top 10% of the 20,288 articles shown at least once to 50,000 users.
MIND-small, Nov 9–14, 2019.

---

## Page 2 – Data and method

**Headline:** Two public datasets, four simple measures, one replication.

**Data:**
- **MIND-small (Microsoft News):** all 50,000 users. The train week (Nov 9–14, 2019) is the
  main sample; the dev day (Nov 15) is a replication check. Source: Wu et al., ACL 2020, used
  under the Microsoft Research License. Files from the `Recommenders/MIND` mirror on Hugging
  Face, byte-identical to a second mirror we checked.
- **GDELT 1.0 daily events:** 4.45 million news-reported events, Oct 16 – Nov 14, 2019, by
  the country where each event happened.
- **World Bank population, 2019:** joined to GDELT through GeoNames' FIPS-to-ISO table.

**Measures (plain words):**

| Measure | Question | Read it as |
|---|---|---|
| Diversity (Shannon entropy) | How varied is a news diet? | Higher = more varied |
| Inequality (Gini) / popularity share | Is attention spread evenly? How much goes to the top 10%? | Higher = more concentrated |
| Personalization boost | How much more of your favourite topic is in your feed than in the average feed? | 1× = no boost |
| Familiar vs new click rate | Do people click topics they already read more often? | The relevance side of the trade-off |

**Method note (small chart `02`):** Diversity rises with the number of items counted, so
every comparison samples the larger side down to the same size. Measured that way:
- Shown lists are more varied than readers' click histories in 67% of impressions
  [share_impressions_broader_than_history] (76% on dev).
- Clicks are slightly narrower than the feed (median 1.58 vs 1.83 bits
  [entropy_clicked_median, entropy_feed_matched_median]).

---

## Page 3 – Who sees what (MIND)

**Headline:** Broad but tilted: feeds add new topics, but lean towards each reader's
favourite, and familiar topics win the clicks.

**Body:**
- **Broad:** the typical shown list covers more topics than the reader's history (67% of
  impressions [share_impressions_broader_than_history]).
- **Tilted:** 69% of readers get a feed that over-represents their favourite topic
  [share_users_boosted]. The typical boost is 1.21× [boost_median]. For niche interests it is
  stronger: car readers get 2.6× the average share of car stories [boost_median_autos], and
  food readers 2.3× [boost_median_foodanddrink]. On the dev day the tilt is the same direction
  but weaker (1.07× overall; cars 1.5×), so we report it as a range.
- **Mostly familiar:** 70% of shown slots are in topics the reader already clicks
  [share_slots_in_history_categories]. A feed with no personalization would give 64%
  [baseline_share_if_unpersonalized], so personalization adds about 6 points.
- **Familiar wins clicks:** familiar-topic slots are clicked at 4.5% and new-topic slots at
  2.8% [ctr_familiar, ctr_new]. This holds in 14 of 14 topics, and again on dev (4.8% vs
  2.6%). This is why relevance-optimised feeds drift towards the familiar.
- The topics squeezed most are video and general news for niche readers (about 0.6× and 0.8×).

**Charts:** `04` heatmap (main), `09` familiar vs new (small).

**Small print:** 27,087 users with at least 10 articles read; "readers of X" = most-read
category (behaviour-based, not demographic). Boosts in the text are per-user medians;
heatmap cells are pooled group ratios, so they run higher (cars 3.3×). Click rates are
observational, not the effect of changing the recommender.

**Do not headline:** small groups. Charts only show groups of 200+ users (video, 116, and
music, 119, are left out), and movies (231) and travel (231) are still small. Do not headline "never shown their favourite topic" (3.2%) either: it comes
almost entirely from users with very small feeds.

---

## Page 4 – Where news looks (GDELT)

**Headline:** 10% of countries get 83% of news coverage, and 70% of the world lives in
countries that are under-covered.

**Body:**
- The 10 most-covered countries get 67% of all articles [top10_countries_share_articles].
  The US alone gets 38% [us_article_share], 8.9 times its share of world population
  [coverage_ratio_US].
- The least-covered half of countries (123) share 1.4% of coverage
  [bottom_half_share_articles].
- China and India hold 36% of the world's people but get about 6% of coverage (0.15× and
  0.19× [coverage_ratio_CH, coverage_ratio_IN]). The Democratic Republic of the Congo, with
  93 million people, gets 0.015×.
- 70% of the world's population lives in countries that get less than half their
  population share of coverage [population_share_in_undercovered].
- The pattern holds whether we count articles or events (top 10 countries: 67% vs 61%).

**Charts:** `08` coverage vs population (main), `07` top-15 countries (small, or as a stat).

**Small print:** GDELT 1.0, 30 days ending Nov 14, 2019. Country = where the event happened,
not where it was published. 2.6% of events had no location and were dropped. Population
from the World Bank (2019), matched for 99.6% of articles. Taiwan is not in World Bank data.

---

## Page 5 – Takeaways, limitations, next step

**Three findings:**
1. **Attention is concentrated.** 10% of articles get 91% of impressions; 10% of countries
   get 83% of coverage.
2. **Feeds are broad but tilted.** They show more topics than people read, but lean towards
   each reader's favourite (1.2× typical; 2–2.6× for niche readers).
3. **Relevance pulls towards the familiar.** Familiar topics are clicked 1.6 times as often
   as new ones, in every topic. A system that optimises clicks alone has a built-in reason to
   narrow.

**Limitations:**
- MIND has no demographics, so "communities" are behaviour-based reader groups.
- One platform, one week (plus one replication day) in Nov 2019. It shows a pattern, not
  today's feeds.
- We see what was shown, not why. We cannot separate the algorithm from editors or the news
  cycle. The tilt was weaker on the dev day than in the train week.
- Clicks are not the same as interest, and the click-rate gap is observational.
- Topics are 17 coarse categories, so diversity inside "news" or "sports" is invisible.
- **The time windows differ:** MIND covers Nov 9–15, while GDELT covers the 30 days to
  Nov 14. We compare patterns, not the same period.
- GDELT measures media volume, not importance. Its sources skew English-language and
  US-centred, its location tagging is automatic, and the news cycle shapes the ranking.

**Next step (Phase 2), one line:**
The team's working title is "Beyond the Bubble: a diversity-aware news recommender". It could
pair with a reader-facing "feed diversity label" showing diversity, topic boost and
popularity share, with a "show me what I'm missing" option.

*(Team to confirm before it goes in the PDF.)*
