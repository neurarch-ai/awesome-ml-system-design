import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = "/private/tmp/claude-501/-Users-xingao-Projects-Neurarch/65adcdb4-d882-4642-bf5c-d42bf1be1f61/scratchpad/awesome-ml-system-design/book/tabular/assets/"

plt.rcParams.update({
    'figure.dpi': 130, 'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False, 'figure.autolayout': True
})
BLUE, ORANGE, GREEN, RED, GRAY = '#2563eb', '#ea7317', '#16a34a', '#dc2626', '#64748b'

# ── 1. GBDT vs neural accuracy on tabular benchmarks ──────────────────────────
fig, ax = plt.subplots(figsize=(7.5, 4.2))
datasets = [
    "Adult\n(income)", "Bank\n(telemarket)", "Higgs\n(physics)", "Covertype\n(forest)",
    "Poker\n(hand)", "Year\n(music)"
]
gbdt_auc  = [0.928, 0.932, 0.820, 0.993, 0.994, None]
neural_auc= [0.913, 0.918, 0.811, 0.985, 0.987, None]
gbdt_rmse = [None,  None,  None,  None,  None,  8.91]
neural_rmse=[None,  None,  None,  None,  None,  9.40]

x = np.arange(len(datasets))
w = 0.35

# classification datasets (AUC, higher is better)
auc_idx = [i for i in range(len(datasets)) if gbdt_auc[i] is not None]
ax.bar([x[i] - w/2 for i in auc_idx], [gbdt_auc[i] for i in auc_idx],
       w, label='GBDT (LightGBM/XGBoost)', color=BLUE, alpha=0.87)
ax.bar([x[i] + w/2 for i in auc_idx], [neural_auc[i] for i in auc_idx],
       w, label='Deep neural net', color=ORANGE, alpha=0.87)

# regression dataset (RMSE bars on secondary axis, lower is better)
ax2 = ax.twinx()
ax2.bar(x[-1] - w/2, gbdt_rmse[-1], w, color=BLUE, alpha=0.87)
ax2.bar(x[-1] + w/2, neural_rmse[-1], w, color=ORANGE, alpha=0.87)
ax2.set_ylabel("RMSE (Year, lower better)", color=GRAY)
ax2.tick_params(axis='y', labelcolor=GRAY)
ax2.spines['top'].set_visible(False)
ax2.set_ylim(0, 14)

ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=9)
ax.set_ylabel("AUC (classification, higher better)")
ax.set_ylim(0.78, 1.05)
ax.set_title("GBDT vs deep neural net on tabular benchmarks\n(illustrative, based on published results)")
ax.legend(fontsize=9, frameon=False, loc='lower right')

# annotation
ax.annotate("GBDT matches or beats neural\non clean tabular columns",
            xy=(x[1] - w/2, gbdt_auc[1]),
            xytext=(x[1] + 0.5, gbdt_auc[1] - 0.025),
            fontsize=8, color=GRAY,
            arrowprops=dict(arrowstyle='->', color=GRAY))

fig.savefig(OUT + "fig-gbdt-vs-neural.png")
plt.close(fig)

# ── 2. Calibration reliability curve ─────────────────────────────────────────
rng = np.random.default_rng(42)
n = 2000
bins = np.linspace(0, 1, 11)

fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

for ax, title, skew in zip(axes, ["Well-calibrated model", "Miscalibrated model (overconfident)"], [0, 1]):
    true_p = rng.uniform(0, 1, n)
    y = rng.binomial(1, true_p)
    if skew == 0:
        pred_p = true_p + rng.normal(0, 0.05, n)
        pred_p = np.clip(pred_p, 0.01, 0.99)
    else:
        # overconfident: predictions pushed away from 0.5
        pred_p = 0.5 + 1.5 * (true_p - 0.5)
        pred_p = np.clip(pred_p, 0.01, 0.99)

    bin_midpoints, frac_pos = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (pred_p >= lo) & (pred_p < hi)
        if mask.sum() > 0:
            bin_midpoints.append((lo + hi) / 2)
            frac_pos.append(y[mask].mean())

    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Perfect calibration', zorder=2)
    ax.plot(bin_midpoints, frac_pos, 'o-', color=BLUE if skew == 0 else RED, lw=2,
            ms=6, label='Model', zorder=3)
    ax.fill_between(bin_midpoints, bin_midpoints, frac_pos,
                    alpha=0.18, color=BLUE if skew == 0 else RED, label='Calibration gap')
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, frameon=False)

fig.suptitle("Reliability (calibration) curves: predicted probability vs observed rate", fontsize=11)
fig.savefig(OUT + "fig-calibration-curve.png")
plt.close(fig)

# ── 3. Uplift / Qini curve ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

pop_frac = np.linspace(0, 1, 200)

# Qini: cumulative incremental conversions vs population fraction
# Random (flat uplift), Uplift model, Propensity model, Perfect
qini_random   = pop_frac * 0.04 * 1000   # baseline: 4% incremental rate
qini_uplift   = np.where(pop_frac <= 0.4,
                         pop_frac * 0.10 * 1000,
                         0.4 * 0.10 * 1000 + (pop_frac - 0.4) * 0.005 * 1000)
qini_propensity = np.where(pop_frac <= 0.6,
                           pop_frac * 0.060 * 1000,
                           0.6 * 0.060 * 1000 + (pop_frac - 0.6) * 0.010 * 1000)
qini_perfect  = np.where(pop_frac <= 0.35,
                         pop_frac * 0.114 * 1000,
                         0.35 * 0.114 * 1000 + (pop_frac - 0.35) * 0.0 * 1000)

ax = axes[0]
ax.plot(pop_frac, qini_random,     '--', color=GRAY,   lw=1.5, label='Random baseline')
ax.plot(pop_frac, qini_propensity, '-',  color=ORANGE,  lw=2,   label='Propensity model')
ax.plot(pop_frac, qini_uplift,     '-',  color=BLUE,    lw=2,   label='Uplift model')
ax.plot(pop_frac, qini_perfect,    ':',  color=GREEN,   lw=2,   label='Perfect (oracle)')
ax.set_xlabel("Population fraction targeted")
ax.set_ylabel("Cumulative incremental conversions")
ax.set_title("Qini curve: uplift vs propensity")
ax.legend(fontsize=9, frameon=False)

# Four-quadrant persuadability diagram
ax = axes[1]
ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
ax.axhline(0.5, color=GRAY, lw=1, ls='--')
ax.axvline(0.5, color=GRAY, lw=1, ls='--')
ax.set_xlabel("P(convert | no treatment)")
ax.set_ylabel("P(convert | treatment)")
ax.set_title("Persuadability quadrants")

quads = [
    (0.25, 0.75, BLUE,   "Persuadables\n(target)", 'center'),
    (0.75, 0.75, ORANGE, "Sure things\n(waste)", 'center'),
    (0.25, 0.25, RED,    "Lost causes\n(waste)", 'center'),
    (0.75, 0.25, GRAY,   "Do-not-disturb\n(backfire)", 'center'),
]
for xc, yc, col, label, ha in quads:
    ax.scatter(xc, yc, s=600, color=col, alpha=0.35, zorder=2)
    ax.text(xc, yc, label, ha='center', va='center', fontsize=8.5, color=col, fontweight='bold')

diag = np.array([0, 1])
ax.plot(diag, diag, 'k--', lw=1, alpha=0.4)

fig.suptitle("Uplift targeting: Qini curve and persuadability segments", fontsize=11)
fig.savefig(OUT + "fig-uplift-qini.png")
plt.close(fig)

# ── 4. Survival curve ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
t = np.linspace(0, 24, 300)   # months

def weibull_survival(t, scale, shape):
    return np.exp(-(t / scale) ** shape)

ax = axes[0]
curves = [
    ("High risk",   6,  1.4, RED),
    ("Medium risk", 12, 1.1, ORANGE),
    ("Low risk",    22, 0.9, BLUE),
]
for label, scale, shape, col in curves:
    S = weibull_survival(t, scale, shape)
    ax.plot(t, S, color=col, lw=2.2, label=label)

# Censored points
rng2 = np.random.default_rng(7)
for scale, shape, col in [(6, 1.4, RED), (12, 1.1, ORANGE), (22, 0.9, BLUE)]:
    tc = rng2.uniform(4, 20, 6)
    Sc = weibull_survival(tc, scale, shape)
    ax.scatter(tc, Sc, marker='|', s=100, color=col, zorder=4, alpha=0.7)

ax.set_xlabel("Time (months)")
ax.set_ylabel("Survival probability S(t)")
ax.set_title("Per-customer survival curves\n(| marks censored / still-active)")
ax.legend(fontsize=9, frameon=False)
ax.set_ylim(0, 1.02)

# Hazard rate
ax = axes[1]
t2 = np.linspace(0.1, 24, 300)
for label, scale, shape, col in curves:
    # hazard = (shape/scale) * (t/scale)^(shape-1)
    h = (shape / scale) * (t2 / scale) ** (shape - 1)
    ax.plot(t2, h, color=col, lw=2.2, label=label)

ax.set_xlabel("Time (months)")
ax.set_ylabel("Hazard rate h(t)")
ax.set_title("Hazard rates by risk segment\n(peak = highest churn risk moment)")
ax.legend(fontsize=9, frameon=False)

fig.suptitle("Survival analysis: curves and hazard rates for churn / default modeling", fontsize=11)
fig.savefig(OUT + "fig-survival-curves.png")
plt.close(fig)

print("wrote 4 figures to", OUT)
