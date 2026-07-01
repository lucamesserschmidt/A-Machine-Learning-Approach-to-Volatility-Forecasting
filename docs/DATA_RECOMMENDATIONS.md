# Data recommendations: should you invest more in finding RV data?

**Short answer: yes — and it is cheaper than you think.** The single biggest
fidelity gap between this replication and Christensen, Siggaard & Veliyev (2023)
is the volatility *target*. They use high-quality 5-minute realized variance
(RV5) built from NYSE TAQ. You currently have one noisy free 1-minute SPY file
(2008–2021, ~3,350 days, and only ~86 days in 2021). Constructing RV from raw
1-minute bars is doable (this package does it) but it is the weakest link:
microstructure noise, irregular timestamps, and a short, single-asset panel.

There is a much better free option that gives you *ready-made, research-grade*
realized measures for ~30 assets, and it is trivial to load in R.

---

## 1. Oxford-Man Institute (OMI) Realized Library — the headline recommendation

The OMI Realized Library is the canonical free source of daily realized
measures (it underlies a large fraction of the RV-forecasting literature). It
provides, per day and per asset, a whole menu of estimators already computed
from cleaned high-frequency data: **5-minute RV (`rv5`)**, 5-minute
sub-sampled RV, realized kernel (`rk_parzen`), bipower variation, realized
semivariance, open/close prices and open-to-close returns — i.e. exactly the
inputs the paper uses, including the semivariance you need for SHAR and the
quantities behind HARQ.

**Important status note.** The library was *discontinued*: the official Oxford
page now states it is no longer maintained and will not be replaced. The
**archived dataset is still widely available and perfectly citable** for a
replication, covering ~31 global equity indices from **Jan 2000 to ~end-2021**
(≈5,600+ daily obs each, spanning the GFC and COVID — ideal for the paper's
crisis-period analysis). Access routes, easiest first:

* **R package `bvhar`** (CRAN). Ships the OMI data as built-in datasets:
  `oxford_rv` (5-minute RV) and `oxford_rk` (realized kernel), already widened
  to a date × asset matrix with missing values interpolated. As an R user this
  is a one-liner:
  ```r
  install.packages("bvhar")
  data(oxford_rv, package = "bvhar")   # daily rv5 for ~30 indices, 2000-2021
  ```
* **R package `highfrequency`** (CRAN): `data(realized_library)` — the older
  1996–2009 slice (returns, RV, realized kernels) if you want the early sample.
* **GitHub / Kaggle mirrors** of `oxfordmanrealizedvolatilityindices.csv`
  (the full long-format CSV with every estimator and every index). Search
  "oxford man realized library csv"; several research repos redistribute it.

**Why this matters for your grade.** The paper's results are *cross-sectional*
(many indices) and crisis-aware. With OMI you can move from "one self-built SPY
RV series" to "the same panel design as the literature" — RV5 headline target,
multiple indices, 20+ years, robust estimators — which directly addresses the
seminar's "ambition of the data effort" criterion. The `.SPX` series in OMI is
the natural headline asset and lines up with the paper.

> Caveat to state in your write-up: OMI ends in 2021, so you cannot extend the
> out-of-sample window past then with it. That is fine for a replication; note
> it explicitly and, if you want a post-2021 robustness slice, bridge with one
> of the sources below.

---

## 2. Recent / higher-frequency single-asset data (post-2021 or stock-level)

If you want stocks (the paper also studies the Dow components) or a sample that
runs past 2021, build RV yourself from intraday bars:

* **FirstRate Data** (firstratedata.com): free samples of 1-minute equity/ETF/
  index data; widely used in recent RV papers (e.g. multivariate rough-vol work
  using 1-min SPX/NDX/DJI, 2015–2025). Good for a modern out-of-sample slice.
* **Dukascopy** (historical tick downloader): free tick/1-min FX, indices, some
  equities; high quality but you must aggregate to 5-minute yourself.
* **Kaggle / Alpha Vantage / Polygon free tiers**: convenient but short
  histories and rate limits; fine for a robustness check, not a main sample.
* **Your existing Kaggle SPY 1-min file**: keep it as a *self-built-RV*
  robustness layer (this package already turns it into RV5 + semivariance +
  realized quarticity). It demonstrates you can construct realized measures
  from scratch — worth keeping precisely as a methodological contrast to the
  ready-made OMI series.

---

## 3. Predictors (the M_ALL information set) — already in good shape

Your FRED pulls (VIX `VIXCLS`, EPU `USEPUINDXD`, 3-month T-bill `DTB3`) are the
right free sources and need no upgrade. If you want to enrich M_ALL toward the
paper's spirit, all free from FRED/Fama-French:

* term spread (10Y–3M, `T10Y3M`), credit spread (`BAA10Y`),
* ADS Business Conditions index (Philadelphia Fed),
* Fama-French / momentum factors (Kenneth French data library),
* aggregate dollar volume (you already build this from the price files).

---

## Bottom line / recommended plan

1. **Adopt the OMI Realized Library via `bvhar` as the headline RV5 source**
   for SPX + a cross-section of indices, 2000–2021. Biggest fidelity gain for
   the least effort, and native to your R workflow.
2. **Keep your self-built SPY-1min RV5** as a "we can also construct realized
   measures ourselves" robustness layer (it also yields the semivariance and
   realized quarticity that power SHAR/HARQ).
3. **Optionally bridge post-2021** with FirstRate/Dukascopy 1-min for a modern
   out-of-sample robustness slice.
4. Leave the FRED predictor set as is.

This Python package consumes either source through the same daily-RV interface
(`add_intraday_realized_variance_features` for self-built RV; for OMI you would
load the wide `rv5` matrix and feed each column as a daily RV series), so
swapping in OMI is a thin data-loading change, not a redesign.
