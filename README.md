# CuStorA — Customer Intelligence and Retention Decision-Support System

A Flask app that takes customer data (CSV/XLSX), cleans it, and runs it
through a pipeline combining unsupervised segmentation, supervised
churn prediction, explainability, relationship-health scoring, and
rule-based retention recommendations — all shown on a dashboard.

CuStorA answers one question, broken into steps: **which customers
need attention, and why?**

```
CUSTOMER DATA
      ↓
WHO ARE THEY?            RFM + K-Means segmentation
      ↓
WHAT MIGHT HAPPEN?       Churn prediction (LogReg / Random Forest / MLP)
      ↓
WHY?                     SHAP explanations, translated to plain English
      ↓
HOW HEALTHY IS THE
RELATIONSHIP?            Relationship health score (independent of churn)
      ↓
WHAT DOES THAT MEAN
OPERATIONALLY?           Customer state (health × churn combined)
      ↓
WHAT SHOULD WE DO?       Rule-based retention recommendation
```

## Two ways to feed it

**1. A transaction history (recommended — this is what companies really have).**
Upload one row per order line with a customer id, a date and an amount
(any column names; CuStorA detects them and you confirm). From it CuStorA:

- builds a profile per customer (recency, frequency, spend, buying
  trend, rhythm, returns) using only information available on the
  snapshot date;
- **defines churn as "no purchase in the next N days"** (you choose
  60/90/120/180) — it is computed from the ledger, never guessed;
- trains the churn model on past snapshots and **tests it on the most
  recent period it never saw**, comparing against a simple
  "longest since last order" baseline;
- scores every currently-active customer, then runs the same
  segmentation, health, explanation and recommendation stages.

Fields a ledger cannot supply (age, gender, complaints, discount use)
are reported as *not available* on the **Model evidence** page and are
left out of scoring rather than invented. The upload page has a
one-click **real-data demo** (UCI Online Retail, 398k real purchase lines from 4,338 identified customers after guest rows are removed):
held-out ROC-AUC ≈ 0.74 (95% interval 0.72–0.76) against a 0.60 baseline;
see *What the model upgrade changed* below for the full, honest picture.

**2. A ready-made customer table** (the original 13-column schema with a
`Churn` column), or the built-in **synthetic generator**. Synthetic
data only demonstrates that the software runs; the dashboard labels it
as such and its accuracy figures are not evidence.

The **Model evidence** page (`/evidence`) shows, from files the pipeline
wrote: the time-split test of the current analysis, three independent
real-dataset checks (Telco, E-Commerce, Online Retail), what drives the
predictions, which fields were observed vs. derived vs. missing, and
the limitations.

## Why so many components?

It can look like CuStorA does the same thing five different ways.
It doesn't — each piece answers a different question, and the later
stages consume the earlier ones:

| Component | Question it answers |
|---|---|
| RFM | What is this customer's value/behaviour profile? |
| K-Means (on top of RFM + engineered features) | Which behavioural group does this customer resemble? |
| Churn model | How likely are they to leave, going forward? |
| SHAP | Why did the model predict that risk, for this specific customer? |
| Relationship health | How healthy is the relationship *right now*, independent of the churn model's prediction? |
| Customer state | What does health + churn risk mean operationally (e.g. "Hidden Risk" = looks healthy but the model disagrees)? |
| Recommendation | Given all of the above, what should the business actually do? |

Health and churn risk are deliberately kept as two separate signals
rather than merged into one score — a customer can look healthy on
recency/engagement/friction right now while the churn model still
flags rising risk from patterns those three numbers don't capture
("Hidden Risk"), and the reverse also happens. Collapsing them into
one number would throw that disagreement away.

## Methodology and design decisions

This section exists because most of the parameters below are
**judgement calls, not statistically fitted values** — CuStorA doesn't
have enough real transactional history (with confirmed retention
outcomes) to calibrate them. Being explicit about that is more honest
than presenting fixed thresholds as if they were derived.

**Relationship health score** (`src/health/relationship_health.py`)
is a weighted blend of recency (40%), engagement (35%), and service
friction (25%) — recency weighted highest because a customer who
stopped buying is the clearest single warning sign, engagement close
behind as an earlier-warning signal, friction weighted least because
complaints/returns are sparser per customer and shouldn't let one
incident dominate. The score also blends a smooth numeric calculation
(70%) with fuzzy "good/bad" membership (30%), so a customer who's
solidly good on all three metrics scores a bit higher than a straight
average would give them. All of the weights and the fuzzy membership
breakpoints (30/180 days, 30/70 engagement, 1/5 friction incidents)
are named constants at the top of that file with the reasoning
written out — if this were pointed at a real business, they should be
re-derived from that business's actual repurchase cycle and complaint
rate, or fit against real outcomes if enough history exists.

**Customer state thresholds** (`src/state/customer_state.py`) split
predicted churn probability into low/medium/high bands at 40% and
60% — round numbers chosen as a starting point in the absence of
outcome data to calibrate against, not statistically derived cutoffs.

**Churn model selection** (`src/churn/churn_models.py`) picks the
best of Logistic Regression / Random Forest / MLP by **Recall first,
F1 as a tiebreaker** — not the more common "highest F1" or "highest
accuracy" rule. The reasoning: CuStorA's goal is retention, and in
that framing a false negative (missing a real churner) is more costly
than a false positive (flagging someone who was actually fine). F1 as
a tiebreaker exists so the policy doesn't just pick whichever model
flags the most people as at-risk — among similar-Recall models, it
favors the one not achieving that Recall by over-flagging. On the
current dataset this rule picks Logistic Regression over Random
Forest, which would win on F1 alone — see
`reports/churn/churn_model_comparison.csv`.

The training pipeline avoids leaking segmentation outputs and
duplicate-signal columns (e.g. `TotalPurchases` and `Frequency` are
the same value under two names) into the churn model — see the
comment block above `columns_to_drop` in `churn_models.py` for the
full list and why each one is excluded.

### Train / validation / test split — the test set is untouched until final evaluation

An earlier version of this pipeline split the data into just train
and test, then used the test set both to pick the operating threshold
*and* to report the resulting Precision/Recall/F1 — using the same
data to make a decision and then to evaluate that decision, which
overstates how good the numbers actually are. That's fixed now.

Every run of `src/churn/churn_models.py` splits the data three ways
(stratified by churn, 60/20/20):

* **Train** — fits every candidate model, and is also what
  `CalibratedClassifierCV` uses internally to calibrate the winning
  model's probabilities.
* **Validation** — the three candidate models (Logistic Regression /
  Random Forest / MLP) are compared on validation, not test; the
  operating threshold and the Medium/High risk percentile bands are
  also chosen on validation. Every diagnostic artifact that feeds a
  decision — `churn_model_comparison.csv`, `threshold_table.csv`,
  `calibration_table.csv`, `precision_recall_curve.png`,
  `calibration_plot.png` — is built from validation.
* **Test** — used exactly once, at the very end of the run, to
  produce `reports/churn/final_test_evaluation.csv` and
  `final_test_confusion_matrix.png`. Nothing above that point reads
  from the test set, and nothing after it feeds back into an earlier
  decision.

`reports/churn/decision_thresholds.json` records which set the
operating threshold was chosen on (`operating_threshold_selected_on`)
and which population the risk bands were computed over
(`risk_band_population`), so this is checkable from the output files
themselves rather than only from this README.

**Caveat this split surfaces on the current 1,000-row demo dataset:**
a 20% validation split is only ~200 rows (~34 churners), which is a
small sample to pick a threshold from — on the shipped synthetic
data this actually picks an aggressive low threshold (flags almost
everyone) because it's chasing Recall on a small, noisy sample. That
instability is a real limitation of trying to do threshold selection
properly on a small dataset, not something the leakage fix caused —
the old version had the same-size sample, it just called it "test"
and used it twice. It's expected to stabilize with more data (see
"Run CuStorA on real data" below).

### Current demo results

Numbers from the most recent pipeline run on the shipped synthetic
dataset. Model comparison (`reports/churn/churn_model_comparison.csv`)
is measured on the **validation** set — treat these as "the pipeline
and evaluation methodology work end-to-end", not as a claim about
real-world predictive accuracy (see *Data and limitations* below):

| Model               | Accuracy | Precision | Recall |     F1 | ROC-AUC |  Brier |
|----------------------|--------:|----------:|-------:|-------:|--------:|-------:|
| Logistic Regression  |   0.555 |     0.177 |  0.441 |  0.252 |   0.498 |  0.255 |
| MLP Neural Network   |   0.825 |     0.333 |  0.029 |  0.054 |   0.469 |  0.177 |
| Random Forest        |   0.810 |     0.000 |  0.000 |  0.000 |   0.524 |  0.154 |

**Best model:** Logistic Regression
**Selection criterion:** Recall → F1 (see rationale above)

**Final, one-time test evaluation** (`reports/churn/final_test_evaluation.csv`,
`final_test_confusion_matrix.png`) — the calibrated Logistic
Regression model at its validation-chosen threshold, scored once on
the untouched test set:

| Accuracy | Precision | Recall |     F1 | ROC-AUC |  Brier |
|---------:|----------:|-------:|-------:|--------:|-------:|
|    0.170 |     0.170 |  1.000 |  0.291 |   0.595 |  0.141 |

This is a good illustration of the small-validation-set caveat above
in action: the threshold picked on ~200 validation rows (0.05) flags
almost every test customer, so Recall is perfect but Precision is
poor. That's a real, checkable result of the methodology on this
dataset size — not something smoothed over.

**Is the predicted probability trustworthy?** `reports/churn/calibration_table.csv`
and `calibration_plot.png` bin the best model's predictions and
compare the average predicted probability in each bin against the
actual observed churn rate in that bin. On the current demo dataset
there's a real gap between the two in most bins (see
`CalibrationGap` in the table) — the model's numbers should be read
as a *risk ranking*, not yet as a calibrated probability, until that
gap closes on real outcome data.

**Are the 40%/60% churn risk bands defensible?** They're round-number
starting points, but they're no longer just asserted — every pipeline
run now produces `reports/churn/threshold_table.csv`, showing Recall,
Precision, and customers-flagged at candidate thresholds (0.30 through
0.70) for the best model, alongside a precision-recall curve
(`reports/churn/precision_recall_curve.png`). That table is what makes
the 40%/60% choice checkable: it shows the actual recall/precision
trade-off near those cut points on this dataset, rather than asking
you to take the round numbers on faith.

**Feature overlap in clustering.** `Complaints`/`Returns` and their
rate versions (`ComplaintRate`/`ReturnRate`) measure the same
underlying idea — service friction — as a raw count vs. a
purchase-volume-normalized rate. Including both in K-Means would let
"friction" quietly outweigh every other concept in the distance
calculation simply because it's represented twice. `CLUSTER_FEATURES`
in `src/clustering/kmeans.py` keeps the rate versions only; the raw
counts are still available elsewhere (churn model, customer profile
page) where duplication isn't an issue.

**Segment naming reflects each cluster's own characteristics, not a
forced value ladder.** Earlier versions of `segment_naming.py` ranked
clusters by a combined Frequency+Monetary-Recency score and assigned
names from a single low-to-high "value ladder" (Low-Activity →
VIP). That breaks down the moment a cluster is high on monetary value
but low on engagement — it isn't meaningfully "above" or "below" a
cluster that's the opposite, it's just different. Segments are now
named from each cluster's own normalized scores on Monetary,
Frequency, Engagement, and Recency (e.g. "High-Value, Dormant
Customers", "Frequent, Low-Engagement Customers"), capped at the two
strongest descriptors for readability, with disambiguation if two
clusters land on the same name.

**K selection.** `choose_k()` in `src/clustering/kmeans.py` picks the
candidate K with the highest Silhouette Score. Davies-Bouldin is also
computed and reported, but purely as a secondary diagnostic to sanity-
check the chosen K against — not a second vote in the decision. If a
viva question is "why did only one of the two metrics decide K", the
answer is: that's intentional, and it's documented in the function's
docstring.

## Data and limitations

**The synthetic dataset is a demonstration tool, not a benchmark.**
The built-in generator (`src/data/demo_generator.py`) creates churn
labels from rules over the same behavioural fields the model is
trained on. That means strong-looking metrics on synthetic data
mostly show the pipeline works end-to-end — they aren't evidence the
model would predict real churn well, because a real churn model has
to work against causes it never directly observes. On the dataset
currently shipped with this repo, model performance is actually
fairly modest (see `reports/churn/churn_model_comparison.csv`), which
is closer to what you'd expect from a real dataset than a suspiciously
perfect score would be.

**The upload page can now accept close-but-not-exact column names.**
An uploaded CSV/XLSX ideally matches CuStorA's own schema exactly
(`CustomerID`, `Age`, `TotalSpent`, `EngagementScore`, etc. — see
`REQUIRED_COLUMNS` in `app.py`), but if it doesn't, `src/data/column_mapper.py`
(a real, unit-tested rule-based mapper, built and validated against a
genuine external dataset — see the next section) suggests a mapping
from the file's actual column names onto CuStorA's schema. The Upload
page shows that suggestion as a coverage table (which fields matched,
which didn't, and why) and lets you apply it with one click instead of
renaming columns and re-uploading. Fields with no plausible source
column are reported as unavailable rather than guessed at or filled
with a placeholder, per the module's design.

**External validation confirmed the schema is the main remaining
gap.** The pipeline was run against the real IBM/Kaggle "Telco
Customer Churn" dataset (7,043 real customers, genuine churn labels).
Only 4 of 13 canonical fields honestly mapped (Telco is a subscription
business with no discrete "orders," dates, complaints, or returns to
map from), and cleaning/feature engineering failed immediately on the
rest, confirming the schema-coupling problem on real data rather than
just in theory. The churn-modeling *methodology* itself, however,
doesn't hardcode CuStorA's column names — run unmodified on Telco's
own native columns, it reached ROC-AUC 0.83 on a genuinely held-out
real test set, evidence the train/validation/test methodology finds
real signal rather than just fitting synthetic labels back to their
own generating rules.

**A second, independent real dataset confirmed the same picture.**
The same exercise was repeated against the real Kaggle "E-Commerce
Customer Churn" dataset (5,630 real customers, genuine 0/1 churn
labels). This dataset is e-commerce like CuStorA's own schema, but is
a last-month *snapshot* rather than a lifetime purchase ledger, so it
still only mapped 4 of 13 canonical fields honestly, and cleaning/
feature engineering failed the same way Telco's did. Run unmodified on
this dataset's own native columns, the churn methodology reached
ROC-AUC 0.89 on the held-out test set — a second independent
confirmation, from a different business type, that the methodology
finds real signal on data CuStorA had no hand in generating. Because
only 4 of 13 fields honestly mapped, this dataset was **not** used to
replace `data/raw/customers_messy.csv` — doing so would mean
fabricating the other 9 fields, which would reintroduce the exact
label-independent-of-features problem this section exists to flag.

**A third real dataset, with real dates, made real temporal
validation possible.** Both datasets above are snapshots — one row per
customer, no invoice-level dates — so they could only test the
modeling *methodology*, not a real forward-looking churn label or
trend features. A deeper exercise was run against the real UCI "Online
Retail" dataset (530,103 real purchase-line transactions, before guest-checkout rows were excluded, from 4,340
real customers, Dec 2010-Dec 2011, with genuine per-invoice dates).
Because that dataset has real dates, it supported a proper 90-day
forward-looking churn label (not "is inactive right now"), 30
leakage-safe temporal/RFM/trend/slope/volatility features computed
only from information available as of each cutoff date, and a genuine
time-based train/validation/test split instead of a random shuffle.
Compared against LogReg/Random Forest/MLP plus XGBoost and LightGBM,
the best models reached **ROC-AUC 0.77, PR-AUC 0.53, and Lift@10% of
~2.1-2.2x** on a held-out test month that comes chronologically
*after* all training data — a real, believable, and much more
actionable number than the synthetic pipeline's ~0.60, and
intentionally not pushed toward 0.90+ (which would most likely mean
the label or split had started leaking future information).

*(The reference datasets, validation scripts, temporal model, and full
write-ups behind these three findings were development/research-only
material — not read by `app.py`, `pipeline.py`, or anything under
`src/` — and have been removed from this package to keep it small; see
"Development vs. application artifacts" below.)*

**The rule-based recommendation engine is intentionally rule-based,
not a model.** `src/recommendations/retention_recommendations.py`
maps customer state + a few signals to a recommendation via `if`/`elif`
rules. That's a design choice for explainability, not a limitation to
apologize for — but it should be described as what it is ("Rule-Based
Retention Recommendation Engine"), not marketed as AI-generated
recommendations. Each recommendation is stored with the specific
signals that triggered it (`RecommendationEvidence`) and a
`RecommendationConfidence` label (High/Medium/Low, based on how many
independent signals fired) — shown on the customer profile page under
"Why this action" — so a recommendation is traceable back to the
actual numbers that produced it, not just a name.

**Missing values are reported, not silently guessed.** Cleaning
(`src/data/cleaning.py`) now returns a "Data Health" report
(`reports/data_health.csv`) — duplicates removed, negative/invalid
values cleared, values imputed, and (importantly) how many `Churn`
labels were missing. A missing `Churn` value is left missing rather
than filled with the column mode; churn_models.py already excludes
unlabeled rows from training (`prepare_features`), so a manufactured
label is never used to teach the model something that wasn't actually
observed. `Gender` is likewise treated as a generic category — any
non-blank value is kept as-is rather than being restricted to a fixed
Male/Female list or silently overwritten with whichever value happens
to be most common in the file.

## How it works

1. **Get data** — on the Upload page, either:
   - upload your own CSV/XLSX (must match CuStorA's schema, shown on
     the page), or
   - type a number of customers and click **Generate Dataset** to
     create a synthetic dataset instead (`src/data/demo_generator.py`)
2. **Clean it** — clicking "Run Analysis" kicks off `pipeline.py`,
   which always cleans the data first (`src/data/cleaning.py`):
   dedupes customers, fixes bad/negative values, fills missing data,
   parses dates.
3. **Get insights** — the rest of the pipeline runs automatically on
   the cleaned data and produces the dashboard: RFM segments, customer
   clusters, churn risk, relationship health, customer state,
   retention recommendations, and estimated CLV per customer.

## Project structure

```
app.py                 Flask routes (dashboard, upload, generate, run-analysis)
pipeline.py            Runs all pipeline stages in-process, in order
src/
  data/                Cleaning, synthetic data generation, column_mapper.py
  features/            Feature engineering (recency, tenure, etc.)
  rfm/                 Recency/Frequency/Monetary scoring
  clustering/          PCA + K-Means + naming the resulting segments,
                        cluster_investigation.py (why K=2)
  churn/               Trains and compares churn prediction models
  explainability/      SHAP explanations for churn predictions
  health/              Relationship health scoring
  state/               Overall customer state (e.g. "Critical Retention")
  recommendations/     Retention recommendation per customer
  business/            3-year risk-adjusted value + business-priority metrics
templates/              Dashboard, upload, and customer profile pages
tests/                  Pytest suite for the pure business-logic functions
data/                   Raw uploads, demo data, pipeline outputs
reports/                Charts and CSV summaries the pipeline produces
models/                 The saved (best) churn model
```

## Development vs. application artifacts

This package contains only what the running app actually needs. An
earlier development pass of this repo also included larger reference
material used solely to validate the methodology against real,
independently-sourced data: ~47 MB of real Telco/e-commerce/UCI Online
Retail reference datasets, a ~7.2 MB model trained only on the UCI
dataset, the standalone validation/investigation scripts that produced
them, and their write-ups. Nothing in `app.py`, `pipeline.py`, or `src/`
ever read any of it, so it's been removed to keep this package small —
see *Data and limitations* above for what those experiments found.


## Running it

```bash
pip install -r requirements.txt
python app.py          # then open http://127.0.0.1:5000
```

There is **no login** — this is a single-user local demo, so don't
expose it to a network. Set `FLASK_DEBUG=1` only on a trusted local
machine (the Werkzeug debugger allows code execution from the browser).

Open **Upload**, then either click **Analyse real retail data** (about
under a minute), upload your own transaction file or customer table, or
generate synthetic customers; then click **Run analysis**. The pipeline
runs on a background thread with a live progress page. One run at a
time. The repo ships with a finished real-data run, so the dashboard
and Model evidence page work immediately on first start.

Uploads are capped at 100 MB (`MAX_CONTENT_LENGTH`).

Command line (no web app): `python pipeline.py` uses an already-uploaded
customer table, or asks how many synthetic customers to generate.

Running the tests needs pytest: `pip install -r requirements-dev.txt`.

Run the test suite:

```bash
pip install -r requirements.txt -r requirements-dev.txt   # dev file adds pytest
pytest
```

## Known limitations / future improvements

Roughly in the order they'd be tackled:

- **Upload validation and the dataset-health checklist** (`validate_uploaded_data`,
  `build_health_checks` in `app.py`) still live inline in the Flask
  app; they belong in a reusable `src/validation/` module, same as any
  other pipeline stage.
- **Cleaning's "what changed" summary is aggregate, not row-level.**
  `reports/data_health.csv` now reports counts (rows deduped, values
  imputed, missing labels found, etc.) — see *Methodology* above —
  but it doesn't yet show which specific rows/cells changed. A
  row-level diff would be the next step.
- **Test coverage is a start, not comprehensive.** `tests/` currently
  covers the pure business-logic functions (cleaning, feature
  engineering, RFM, health scoring, churn feature prep and model
  selection, customer state, recommendations, CLV, clustering/naming,
  column mapping, guest-id cleaning, BG/NBD, leakage and evaluation guarantees,
  the dashboard JSON/CSV endpoints) with 121 tests — it does not yet cover
  `pipeline.py`'s end-to-end orchestration with integration tests.
- **Config/secrets** — `SECRET_KEY` and `FLASK_DEBUG` now come from
  the environment (see *Running it*) rather than being hardcoded.
- **Page-specific CSS still lives inside some templates** (upload,
  evidence, profile); shared tokens, sidebar, top bar and the dashboard are
  centralised in `static/css/` and `templates/_sidebar.html` / `_topbar.html`.
- **Priority queue ranking (`BusinessPriority`) is itself a rule**,
  same caveat as the recommendation engine: it's an explainable
  heuristic (churn risk × customer value × health severity), not a
  model fit to real outcome data.
- **No persistent database, model/version tracking, or intervention
  outcome tracking yet** — everything is file-based and stateless
  across pipeline runs. Deliberately out of scope for the current
  demo; would matter for anything running in production over time.
- **Concurrent pipeline runs are now isolated** (previously the
  biggest concurrency risk on this list). Each run gets its own
  workspace under `data/runs/<run_id>/`, and `/run-analysis` only ever
  has one run in flight at a time (single-user demo, guarded by a
  lock). A proper task queue (RQ/Celery) would be the next step if
  this ever needed to support several people running analyses at once.

## The interface

The dashboard is a **retention command center**, built to answer "who do I
call, and is it worth it?" rather than to look impressive:

- **Key figures**: expected revenue loss over the churn window, expected
  number of leavers, how many customers deserve a personal touch, and how far
  to trust the ranking (the share of real leavers found in the riskiest 20% of
  the held-out period).
- **Risk × value matrix** (own page, `/risk-value`): every customer as a dot (chance of leaving vs.
  revenue at stake). Four groups — *Save now*, *Protect*, *Win back cheaply*,
  *Leave alone* — each with a suggested play and a link that opens that group in the queue.
- **Outreach planner** (own page, `/outreach-planner`): set how many customers your team can contact, an
  assumed win-back rate and a cost per contact. It shows expected leavers
  reached, loss covered, revenue you could keep and net benefit, can pick the
  best number for your cost, shows what the held-out test says about that
  coverage, and exports the list as a formatted Excel file (`/customers/export.xlsx`; a plain CSV is still available at `/customers/export.csv`).
- **Where the loss is concentrated** (by segment) and **what the model looks
  at** (global SHAP drivers).
- **Action queue** over *all* customers: search, filter by state, sort by
  expected loss / priority / risk / revenue at stake, page through, and open
  any customer in a side drawer with plain-English reasons.

Light and dark themes follow the system preference until you toggle; the
sidebar has its own light and dark colours. All styling is in
`static/css/app.css` (tokens, layout, sidebar), `static/css/theme.css`
(palette + refinements) and `static/css/command.css` (dashboard); the sidebar
and top bar are shared partials (`templates/_sidebar.html`, `_topbar.html`).
Dashboard behaviour is in `static/js/command.js` (no external JS libraries).

## What the model upgrade changed (real data, honestly measured)

Held-out period: the most recent 90-day window of the UCI data
(snapshot 2011-09-10 → 2011-12-09), never used to choose or fit anything.

| | Previous model | This model |
|---|---|---|
| ROC-AUC | 0.736 (0.715–0.756) | 0.739 (0.716–0.761) |
| Brier score (lower = better) | 0.191 | 0.184 |
| Average predicted risk (actual: 31%) | 39% | 34% |
| Accuracy at the new cut-off (0.33) | 60% | 64% |
| Balanced accuracy at that cut-off | 0.67 | 0.67 |
| Customers flagged at that cut-off | 61% | 52% |

(The previous model *as shipped* used a cut-off of 0.25 and scored 59% accuracy.
Figures come from `reports/churn/backtest_summary.json`; the model type picked
on validation can change between runs on other data.)

* **The ranking improvement is within noise** (ROC-AUC +0.003, 95% interval −0.006 to +0.011). On one year of one retailer's data, ranking quality sits near
  a ceiling of roughly 0.75–0.78 whatever model or features are used; many
  model families, regularisation, time-decay weights and monotonic constraints
  were tried on development folds and all landed inside the same ±0.012 band.
  A score far above that on this kind of data would indicate leakage.
* **The largest real fix was data cleaning.** `CustomerID = 0` in this file is
  133,358 anonymous guest-checkout rows (25% of rows, 16% of revenue) that were
  being treated as one customer with 1,427 orders who never churns — the top
  "customer" on the dashboard. Placeholder ids (`0`, `-1`, `guest`, `unknown`…)
  are now removed and counted (`rows_dropped_anonymous`), with a safeguard so
  a normal-sized real customer numbered 0 is kept.
* **The operating threshold is now robust to the season.** It used to maximise
  F1 on a spring validation period where ~43% of customers left; at that churn
  rate F1 rewards flagging nearly everyone, so the autumn test (31% left)
  got a flag-almost-everyone cut-off and ~59% accuracy. It now maximises
  balanced accuracy (Youden's J), which does not depend on the churn rate;
  accuracy on the test period is about 64% (a Gradient Boosting model, statistically tied
  with the Logistic Regression that validated best, reached 68% in one run).
* **Evaluation is stricter.** The model scored on validation is trained only on
  snapshots whose outcomes were known *before* the validation date (a 90-day
  embargo), because weekly/fortnightly snapshots of the same customer are
  near-duplicates that otherwise leak the answer. Test is touched once; the
  non-selected models are never scored on it. Calibration uses Platt scaling
  with customer-grouped folds (monotone, so it never creates ties in the
  ranking). The Evidence page shows confidence intervals, the previous-model
  comparison, calibration by risk decile and a gain curve.
* **Features**: 34 leakage-safe features in five groups (level, momentum,
  rhythm, basket, BG/NBD "still alive?" model — `src/churn/bgnbd.py`).
  On this retailer BG/NBD fits almost no dropout, so `PAlive` carries little
  signal here (`ExpectedOrders` does); it can matter on other businesses.
* **Logistic Regression** is range-clipped (`RangeClipper`) so customers far
  outside the older training snapshots' range cannot get absurd scores.
* **Business metrics**: "Critical" priority was unreachable because the
  3-year value is already discounted by churn probability, so a high-risk
  customer could never be high-value. Priority now uses **revenue at stake**
  (normal spend per window) and **expected loss** (probability × stake), and
  the top 1% by expected loss are always Critical.
* **Explanations are sanity-checked**: a reason whose wording contradicts the
  customer's real value (e.g. "spend is on the lower end" for a £168K
  customer, which a linear model's correlated coefficients could produce) is
  skipped.

## What changed most recently

- **Interface update**:
  - *Sidebar*: Risk × value matrix, Outreach planner and a new **Customer clusters** map (`/clusters`) now each have their own page under *Analyse*. **Model evidence** moved to the bottom of the sidebar.
  - *Reached customers* (`/reached`): mark a customer as reached (customer panel, profile page, or the queue row) and they leave the action queue, matrix, planner and call-list export. Undo any time. The list is stored in `data/reached_customers.json` (not tracked) and belongs to the dataset it was made for (`src/outreach/reached_store.py`).
  - *Help desk*: a small button in the sidebar opens a plain-language guide to every page, term and pipeline stage.
  - *Short text, full on hover*: explanatory text shows a short line and expands when you hover or tap it.
  - *Click-a-stage after analysis*: on the progress page, click "stay here", then click any of the 12 stages to see what happened to your data. Read from the files the run left behind (`src/orchestration/stage_report.py`, `/run-analysis/stage.json`).
  - Tests: `tests/test_reached_and_pages.py`.

- **Model upgrade, cleaner data, command-center UI, light-theme sidebar fix**: see the two sections above.

- **Transaction-ledger mode** (`src/data/transactions.py`,
  `src/churn/temporal_churn.py`): upload order history, map columns,
  churn is derived from future purchases, the model is tested forward
  in time against a baseline. Leakage-safety and the no-invention rule
  are covered by `tests/test_transactions.py`.
- **Model evidence page** and a data-source banner on the dashboard
  (real vs. synthetic is always visible).
- **One-click real-data demo** on UCI Online Retail.
- **Segment naming** now ranks clusters by percentile instead of
  rescaling between clusters, so one huge-spend cluster no longer makes
  every other segment read "Low-Spend"; K-Means searches K=3–6 for
  segments a person can act on.
- **Relationship health** re-weights recency/engagement when the data
  has no complaints/returns, instead of handing everyone free points.
- Downstream stages ask the saved model for its own input columns
  (`src/churn/feature_contract.py`) instead of three hard-coded lists.
- **Interface** (see "The interface"): the 3D constellation and spotlight effects were replaced by the retention command center; the sidebar now follows the light/dark theme (it was forced dark in light mode).
- **Fixed**: state buttons only filtered the top 20 customers, so most states looked empty; they now query all customers (`/customers/list.json`), and search covers everyone.
- Currency symbol follows the data (₹ default, £ for the real demo).
- Removed login/CSRF text that no longer matched the code; removed a
  stray unrelated file and build clutter.

Earlier work (still in place): background pipeline execution with a
progress page, per-stage status (`reports/pipeline_status.csv`),
probability calibration, recommendation evidence + confidence,
honest handling of missing churn labels, the Data Health report.
