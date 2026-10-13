#!/usr/bin/env python3
"""
SafeSteps Keychain study - statistical analysis (consolidated).

Reproduces every number reported in the manuscript from the row-level survey
export (the tab-separated paste with TWO header rows: a block-label row
"Adult Neurology Responses ... Child Neurology Responses", then a column-name
row, then one row per resident).

Requirements: python >= 3.9, pandas, numpy, scipy
    pip install pandas numpy scipy

Usage:
    python safesteps_analysis.py raw_data.tsv

Layout of the export (35 tab-separated columns):
    cols  0-17  adult neurology block   (col 18 is a blank spacer)
    cols 19-34  child neurology block
A row belongs to the adult block if col 0 is non-empty, and to the child block
if col 19 (PGY) is non-empty; one physical row can hold one of each.
"""
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------
# 1. Parsing
# --------------------------------------------------------------------------
ADULT_COLS = ["clarity", "carry", "remind_freq", "new_onset_use",
              "recommend_colleague", "pgy_raw", "free_text", "past_discussion",
              "emergency", "head", "body", "mouth_a", "mouth_b", "comfort",
              "confidence", "chronic_use", "recommend_trainee",
              "recommend_patient"]                                  # cols 0-17
CHILD_COLS = ["pgy_raw", "past_discussion", "comfort", "confidence",
              "emergency", "head", "body", "mouth", "clarity", "carry",
              "remind_freq", "new_onset_use", "chronic_use",
              "recommend_colleague", "recommend_trainee",
              "recommend_patient"]                                  # cols 19-34


def parse_pgy(x):
    """'PGY 3', 'pgy2', 'PGY4', '6' -> int; blank -> NaN."""
    m = re.search(r"(\d+)", str(x))
    return int(m.group(1)) if m else np.nan


def read_rows(path):
    """Read the export into a list of 35-field rows (header rows removed).
    Tolerates (a) extra lines before the block-label row, e.g. "Here are the
    raw data", and (b) copies in which each TAB was turned into 4 spaces."""
    with open(path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    start = next(i for i, ln in enumerate(lines) if ln.lstrip().startswith("Adult Neurology"))
    lines = [ln for ln in lines[start:] if ln.strip()]
    rows = []
    for ln in lines:
        if "\t" not in ln:                      # tabs were flattened to 4 spaces
            ln = ln.replace("    ", "\t")
        rows.append([f.strip() for f in ln.split("\t")])
    return rows[2:]                                 # drop the 2 header rows


def load(path):
    data_rows = read_rows(path)
    recs = []
    for r in data_rows:
        r = r + [""] * (35 - len(r))
        if r[0].strip() != "":                              # adult block present
            d = dict(zip(ADULT_COLS, r[0:18]))
            d["mouth"] = d["mouth_b"] if d["mouth_b"].strip() else d["mouth_a"]
            d["group"] = "Adult"
            recs.append(d)
        if r[19].strip() != "":                             # child block present
            d = dict(zip(CHILD_COLS, r[19:35]))
            d["group"] = "Child"
            recs.append(d)
    df = pd.DataFrame(recs)
    df["pgy"] = df["pgy_raw"].map(parse_pgy)
    for c in ["comfort", "confidence", "clarity", "recommend_colleague",
              "recommend_trainee", "recommend_patient"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # One adult respondent wrote "On my bag"; the survey tool's aggregate
    # counts (26 doctor bag / 3 backpack) treat this as "doctor bag".
    df["carry"] = df["carry"].replace({"On my bag": "In my doctor bag"})
    df["child"] = (df["group"] == "Child").astype(int)
    return df


# --------------------------------------------------------------------------
# 2. Recodes
# --------------------------------------------------------------------------
CORRECT = {
    "emergency": "If the seizure lasts longer than 5 minutes",
    "head": "Protect the head by placing a soft object between the head and the ground",
    "body": "Facing sideways with mouth pointing toward the ground",
    "mouth": "Nothing",
}

NEW_ONSET_MAP = {  # 1-4 ordinal intensity
    "Rarely, almost never": 1,
    "Occasionally, if I believe a visual description of seizure safety will benefit the patient": 2,
    "Often, if I am discussing seizure safety with new patients": 3,
    "Every new patient appointment": 4,
}
CHRONIC_MAP = {  # 1-4 ordinal intensity; top two categories merged
    "Rarely, almost never": 1,
    "Occasionally, if I am reminding a patient of seizure safety": 2,
    "Often, if I believe the patient will find it interesting to discuss": 3,
    "At all upcoming appointments, until I have shown it to all of my existing patients": 4,
    "Every time, even if I have already shown it to them": 4,
}


def add_scores(df):
    for k, correct in CORRECT.items():
        df[f"{k}_correct"] = (df[k] == correct).astype(int)
    df["knowledge_score"] = df[[f"{k}_correct" for k in CORRECT]].sum(axis=1)
    df["new_onset_score"] = df["new_onset_use"].map(NEW_ONSET_MAP)
    df["chronic_score"] = df["chronic_use"].map(CHRONIC_MAP)
    return df


# --------------------------------------------------------------------------
# 3. Helpers
# --------------------------------------------------------------------------
def pct(n, d):
    return f"{n} ({100 * n / d:.1f}%)"


def mean_sd(s):
    s = s.dropna()
    return f"{s.mean():.2f} (SD {s.std(ddof=1):.2f}), n={len(s)}"


def mann_whitney(a, c):
    """Two-sided Mann-Whitney U (U of the first sample) with z and r.
    p: scipy asymptotic (tie + continuity corrected).
    z: same normal approximation, tie-corrected, continuity-corrected.
    r = |z| / sqrt(N)."""
    a, c = np.asarray(a, float), np.asarray(c, float)
    n1, n2 = len(a), len(c)
    N = n1 + n2
    U, p = stats.mannwhitneyu(a, c, alternative="two-sided", method="asymptotic")
    _, counts = np.unique(np.concatenate([a, c]), return_counts=True)
    tie = np.sum(counts ** 3 - counts) / (N * (N - 1))
    mu = n1 * n2 / 2
    sigma = np.sqrt(n1 * n2 / 12 * ((N + 1) - tie))
    z = 0.0 if U == mu else (U - mu - 0.5 * np.sign(U - mu)) / sigma
    return dict(Mdn_A=np.median(a), n_A=n1, Mdn_C=np.median(c), n_C=n2,
                U=U, z=z, p=p, r=abs(z) / np.sqrt(N))


def ols(df, outcome, predictors):
    """OLS with unstandardized b (SE, t, p, 95% CI) and standardized beta
    (beta = b * SD(x) / SD(y), identical to refitting on z-scored variables)."""
    s = df.dropna(subset=[outcome] + predictors)
    y = s[outcome].to_numpy(float)
    X = np.column_stack([np.ones(len(s))] + [s[p].to_numpy(float) for p in predictors])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ b
    dof = len(y) - X.shape[1]
    mse = resid @ resid / dof
    se = np.sqrt(np.diag(mse * np.linalg.inv(X.T @ X)))
    t = b / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), dof))
    tcrit = stats.t.ppf(0.975, dof)
    r2 = 1 - resid @ resid / np.sum((y - y.mean()) ** 2)
    out = []
    for i, name in enumerate(["Intercept"] + predictors):
        beta = np.nan if i == 0 else b[i] * X[:, i].std(ddof=1) / y.std(ddof=1)
        out.append(dict(term=name, b=b[i], se=se[i], t=t[i], p=p[i],
                        ci_lo=b[i] - tcrit * se[i], ci_hi=b[i] + tcrit * se[i],
                        std_beta=beta))
    return pd.DataFrame(out), r2, dof, len(y)


# --------------------------------------------------------------------------
# 4. Analyses
# --------------------------------------------------------------------------
def main(path):
    df = add_scores(load(path))
    N = len(df)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")

    print("=" * 78, "\nTABLE 1 - Demographics\n", "=" * 78, sep="")
    print(f"N = {N}; Adult n={int((df.group=='Adult').sum())}, "
          f"Child n={int((df.group=='Child').sum())}")
    ct = pd.crosstab(df["pgy"], df["group"], margins=True)
    print(ct)
    print((pd.crosstab(df["pgy"], df["group"], normalize="columns") * 100).round(1))
    print(df.groupby("group")["pgy"].agg(["count", "mean", "std", "min", "max"]))
    print("Overall mean PGY:", round(df["pgy"].mean(), 2))

    print("\n" + "=" * 78, "\nTABLE 2 - Comfort / confidence distributions\n", "=" * 78, sep="")
    for c in ["comfort", "confidence"]:
        vc = df[c].value_counts().sort_index()
        print(c, {int(k): pct(int(v), N) for k, v in vc.items()},
              "| mean/SD:", mean_sd(df[c]),
              "| rated 4-5:", pct(int((df[c] >= 4).sum()), N))

    print("\n" + "=" * 78, "\nTABLE 4 - Knowledge items and composite score\n", "=" * 78, sep="")
    for k in CORRECT:
        print(f"{k:10s} correct: {pct(int(df[f'{k}_correct'].sum()), N)}")
    print("Composite score distribution:",
          {int(k): pct(int(v), N) for k, v in df['knowledge_score'].value_counts().sort_index().items()},
          "| mean/SD:", mean_sd(df["knowledge_score"]))

    print("\n" + "=" * 78, "\nCORRELATIONS (Spearman, individual level)\n", "=" * 78, sep="")
    pairs = [("comfort", "confidence"), ("pgy", "comfort"), ("pgy", "confidence"),
             ("pgy", "knowledge_score"), ("knowledge_score", "comfort"),
             ("knowledge_score", "confidence"), ("pgy", "clarity"),
             ("pgy", "recommend_colleague"), ("pgy", "recommend_trainee"),
             ("pgy", "recommend_patient")]
    for x, y in pairs:
        s = df.dropna(subset=[x, y])
        rho, p = stats.spearmanr(s[x], s[y])
        print(f"{x:16s} vs {y:20s} rho={rho:6.3f}  p={p:.4g}  n={len(s)}")

    print("\n" + "=" * 78, "\nTABLE 3 - Mann-Whitney U, Adult vs Child\n", "=" * 78, sep="")
    rows = []
    for v in ["comfort", "confidence", "knowledge_score", "clarity",
              "recommend_colleague", "recommend_trainee", "recommend_patient"]:
        a = df.loc[df.group == "Adult", v].dropna()
        c = df.loc[df.group == "Child", v].dropna()
        res = mann_whitney(a, c)
        res.update(variable=v, mean_A=a.mean(), mean_C=c.mean())
        rows.append(res)
    print(pd.DataFrame(rows)[["variable", "Mdn_A", "n_A", "Mdn_C", "n_C", "mean_A",
                              "mean_C", "U", "z", "p", "r"]].to_string(index=False))

    print("\n" + "=" * 78,
          "\nREGRESSION - outcome ~ PGY + Child (unstandardized b and standardized beta)\n",
          "=" * 78, sep="")
    for outcome in ["comfort", "confidence"]:
        tab, r2, dof, n = ols(df, outcome, ["pgy", "child"])
        print(f"\n{outcome}: n={n}, df={dof}, R^2={r2:.3f}")
        print(tab.to_string(index=False))

    print("\n" + "=" * 78, "\nKEYCHAIN ACCEPTABILITY (Tables 5-7)\n", "=" * 78, sep="")
    print("Clarity:", {int(k): pct(int(v), N) for k, v in df['clarity'].value_counts().sort_index().items()},
          "| mean/SD:", mean_sd(df["clarity"]),
          "| agree/strongly agree:", pct(int((df['clarity'] >= 4).sum()), N))
    print("\nCarry method:\n", df["carry"].value_counts(dropna=False).to_string())
    print("\nPersonal-reference use (n excludes missing):\n",
          df["remind_freq"].replace("", np.nan).value_counts(dropna=False).to_string())
    print("\nNew-onset use:\n", df["new_onset_use"].value_counts().to_string())
    print("\nChronic use:\n", df["chronic_use"].value_counts().to_string())
    for c in ["recommend_colleague", "recommend_trainee", "recommend_patient"]:
        print(f"\n{c}:", {int(k): pct(int(v), N) for k, v in df[c].value_counts().sort_index().items()},
              "| mean/SD:", mean_sd(df[c]),
              "| likely/very likely:", pct(int((df[c] >= 4).sum()), N))

    print("\n" + "=" * 78, "\nWITHIN-SUBJECT TESTS\n", "=" * 78, sep="")
    pair = df.dropna(subset=["new_onset_score", "chronic_score"])
    w, p = stats.wilcoxon(pair["new_onset_score"], pair["chronic_score"])
    print(f"Wilcoxon, new-onset vs chronic use intensity: n={len(pair)}, "
          f"means {pair['new_onset_score'].mean():.2f} vs {pair['chronic_score'].mean():.2f}, "
          f"W={w:.1f}, p={p:.4g}")

    rec = df.dropna(subset=["recommend_colleague", "recommend_trainee", "recommend_patient"])
    chi, p = stats.friedmanchisquare(rec["recommend_colleague"], rec["recommend_trainee"],
                                     rec["recommend_patient"])
    print(f"Friedman, recommend colleague/trainee/patient: n={len(rec)}, "
          f"chi2={chi:.2f}, p={p:.4g}")
    post = [("colleague", "trainee"), ("colleague", "patient"), ("trainee", "patient")]
    for a_, b_ in post:
        w, p = stats.wilcoxon(rec[f"recommend_{a_}"], rec[f"recommend_{b_}"])
        print(f"  post hoc {a_} vs {b_}: W={w:.1f}, p={p:.4g}, "
              f"Bonferroni-adjusted p={min(1, p * len(post)):.4g}")

    print("\n" + "=" * 78, "\nFIGURE 1 DATA - mean comfort/confidence by PGY\n", "=" * 78, sep="")
    g = df.groupby("pgy")[["comfort", "confidence"]].agg(["count", "mean", "std"])
    for c in ["comfort", "confidence"]:
        g[(c, "sem")] = g[(c, "std")] / np.sqrt(g[(c, "count")])
    print(g.sort_index(axis=1).to_string())


# --------------------------------------------------------------------------
# 5. Early exploratory tests on the aggregate (count-only) data
#    (run before the row-level data were available; NOT used in the manuscript)
# --------------------------------------------------------------------------
def aggregate_exploratory():
    def gof(counts, label):
        counts = np.array(counts)
        chi2, p = stats.chisquare(counts, np.full(len(counts), counts.sum() / len(counts)))
        print(f"{label}: chi2={chi2:.2f}, df={len(counts) - 1}, p={p:.3g}")

    gof([50, 1, 0, 0, 0, 0], "Emergency call timing")
    gof([47, 3, 1, 0, 0], "Head positioning")
    gof([42, 5, 1, 3, 0, 0], "Body positioning")
    gof([50, 1, 0, 0, 0, 0, 0], "Mouth item")
    gof([2, 2, 6, 17, 24], "Comfort 1-5")
    gof([1, 3, 11, 16, 20], "Confidence 1-5")
    gof([0, 0, 5, 6, 40], "Clarity 1-5")

    def two_prop(x1, n1, x2, n2, label):
        p1, p2 = x1 / n1, x2 / n2
        pp = (x1 + x2) / (n1 + n2)
        z = (p1 - p2) / np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
        print(f"{label}: z={z:.2f}, p={2 * (1 - stats.norm.cdf(abs(z))):.3g}")

    two_prop(18 + 23, 51, 13 + 4, 51, "High-intensity use, new-onset vs chronic (independent-samples approximation)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python safesteps_analysis.py raw_data.tsv")
    main(sys.argv[1])
    print("\n" + "=" * 78, "\nAGGREGATE-ONLY EXPLORATORY TESTS (not used in manuscript)\n", "=" * 78, sep="")
    aggregate_exploratory()
