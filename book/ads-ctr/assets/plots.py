import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = "/private/tmp/claude-501/-Users-xingao-Projects-Neurarch/65adcdb4-d882-4642-bf5c-d42bf1be1f61/scratchpad/awesome-ml-system-design/book/ads-ctr/assets/"

plt.rcParams.update({
    'figure.dpi': 130,
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.autolayout': True,
})

BLUE  = '#2563eb'
ORANGE = '#ea7317'
GREEN  = '#16a34a'
RED    = '#dc2626'
GRAY   = '#64748b'

# -----------------------------------------------------------------------
# 1) Calibration reliability curve
#    A well-calibrated model vs an over-confident raw DNN head
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 4.6))

rng = np.random.default_rng(42)
bins = np.linspace(0, 1, 11)
midpoints = (bins[:-1] + bins[1:]) / 2

# Perfect calibration
ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Perfect calibration', zorder=1)

# Raw DNN: over-confident (probabilities pulled toward extremes)
raw_frac_pos = midpoints ** 0.55  # systematic shift
ax.plot(midpoints, raw_frac_pos, 'o-', color=RED, lw=2, markersize=6,
        label='Raw DNN head (over-confident)', zorder=3)

# After Platt scaling: much closer to diagonal
platt_frac_pos = midpoints ** 0.92 + rng.normal(0, 0.015, len(midpoints))
platt_frac_pos = np.clip(platt_frac_pos, 0, 1)
ax.plot(midpoints, platt_frac_pos, 's-', color=GREEN, lw=2, markersize=6,
        label='After Platt scaling', zorder=3)

# Shade between raw and perfect to show calibration gap
ax.fill_between(midpoints, midpoints, raw_frac_pos,
                alpha=0.12, color=RED, label='Calibration gap (mis-pricing zone)')

ax.set_xlabel('Mean predicted pCTR (confidence)')
ax.set_ylabel('Observed click rate (fraction positive)')
ax.set_title('Reliability curve: raw DNN vs calibrated\n(mis-calibration = mis-priced eCPM)')
ax.legend(fontsize=9, frameon=False, loc='upper left')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

fig.savefig(OUT + 'fig-calibration-reliability.png')
plt.close(fig)


# -----------------------------------------------------------------------
# 2) AUC vs log loss tradeoff
#    Illustrates how a model can have high AUC but poor calibration
#    (log loss drops more when probabilities are honest)
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))

# Left panel: AUC separates models by rank order only
# Two score distributions: positive and negative
rng2 = np.random.default_rng(7)
neg_scores = rng2.beta(2, 6, 2000)
pos_scores = rng2.beta(4, 3, 2000)

ax = axes[0]
ax.hist(neg_scores, bins=40, alpha=0.6, color=RED,   density=True, label='negatives (no click)')
ax.hist(pos_scores, bins=40, alpha=0.6, color=GREEN,  density=True, label='positives (click)')
ax.axvline(0.5, color=GRAY, ls='--', lw=1.2)
ax.text(0.52, ax.get_ylim()[1] * 0.85, 'threshold', color=GRAY, fontsize=9)
ax.set_xlabel('Model score')
ax.set_ylabel('Density')
ax.set_title('AUC cares only about rank order\n(order can be right, scale still wrong)')
ax.legend(fontsize=9, frameon=False)

# Right panel: log loss punishes miscalibration
# Show that multiplying all scores by a constant hurts log loss but not AUC
ax = axes[1]
true_p = np.array([0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80])
scale_factors = [0.4, 0.7, 1.0, 1.5, 2.2]
colors_sf = [RED, ORANGE, GREEN, ORANGE, RED]

for sf, c in zip(scale_factors, colors_sf):
    p_pred = np.clip(true_p * sf, 1e-6, 1 - 1e-6)
    ll = -(true_p * np.log(p_pred) + (1 - true_p) * np.log(1 - p_pred))
    label = f'scale x{sf:.1f}'
    ax.plot(true_p, ll, 'o-', color=c, lw=1.8, markersize=5, label=label)

ax.set_xlabel('True click rate')
ax.set_ylabel('Per-example log loss')
ax.set_title('Log loss punishes probability scale errors\n(AUC is identical for all scale factors)')
ax.legend(fontsize=8.5, frameon=False, ncol=2)

fig.savefig(OUT + 'fig-auc-vs-logloss.png')
plt.close(fig)


# -----------------------------------------------------------------------
# 3) Feature-hashing collision rate vs table size
#    Shows the birthday-paradox style collision curve for typical id counts
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.2))

id_counts_millions = [1, 5, 10, 50, 100, 500, 1000]
table_sizes = np.array([2**17, 2**18, 2**19, 2**20, 2**21, 2**22, 2**23, 2**24, 2**25, 2**26])

for n_ids_m, c in zip([1, 10, 100, 1000], [GREEN, BLUE, ORANGE, RED]):
    n_ids = n_ids_m * 1_000_000
    # Approximate collision probability: 1 - exp(-n*(n-1)/(2*B)) for n balls, B bins
    collision = 1 - np.exp(-n_ids * (n_ids - 1) / (2.0 * table_sizes))
    collision = np.clip(collision, 0, 1)
    ax.semilogx(table_sizes, collision, 'o-', color=c, lw=2, markersize=5,
                label=f'{n_ids_m}M distinct ids')

ax.axhline(0.05, color=GRAY, ls='--', lw=1.2)
ax.text(table_sizes[1] * 1.1, 0.07, '5% collision threshold', color=GRAY, fontsize=9)
ax.set_xlabel('Hash table size (number of buckets, log scale)')
ax.set_ylabel('Expected collision probability')
ax.set_title('Feature hashing: collision rate vs table size\n(controlled quality cost for bounded memory)')
ax.legend(fontsize=9, frameon=False)
ax.set_ylim(0, 1.05)

fig.savefig(OUT + 'fig-feature-hashing-collisions.png')
plt.close(fig)


# -----------------------------------------------------------------------
# 4) Delayed-feedback timeline
#    Shows when clicks and conversions land relative to an impression
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 3.8))

rng3 = np.random.default_rng(99)

# Impression at t=0
# Clicks: exponential with mean ~30 seconds
# Conversions: log-normal with median ~1.5 days

impression_time = 0

click_delays_sec = rng3.exponential(30, 300)
conv_delays_hours = np.exp(rng3.normal(np.log(36), 0.8, 200))  # median ~36 hours

click_color = GREEN
conv_color = ORANGE

# Convert to hours for the x axis
click_delays_h = click_delays_sec / 3600

# Plot event cloud (jitter vertically for readability)
jitter_c = rng3.uniform(-0.08, 0.08, len(click_delays_h))
jitter_v = rng3.uniform(-0.08, 0.08, len(conv_delays_hours))

ax.scatter(click_delays_h, 1 + jitter_c, s=15, color=click_color, alpha=0.4, label='click events')
ax.scatter(conv_delays_hours, 2 + jitter_v, s=15, color=conv_color, alpha=0.4, label='conversion events')

ax.axvline(0, color=GRAY, lw=1.5, ls='--')
ax.text(0.01, 2.45, 'impression\n(t = 0)', color=GRAY, fontsize=9)

# Attribution window boundary
attr_window = 72  # hours
ax.axvline(attr_window, color=RED, lw=1.5, ls='--')
ax.text(attr_window + 1, 2.45, '72-hour\nattribution\nwindow', color=RED, fontsize=8.5)

# Shade the "unresolved" zone: inside window, no label yet
ax.axvspan(0, attr_window, alpha=0.06, color=BLUE)
ax.text(attr_window / 2, 0.6, 'unresolved zone\n(not-yet-converted != confirmed negative)',
        ha='center', color=BLUE, fontsize=8.5)

ax.set_yticks([1, 2])
ax.set_yticklabels(['clicks', 'conversions'])
ax.set_xlabel('Time after impression (hours)')
ax.set_title('Delayed feedback: clicks arrive in seconds, conversions in days\n'
             'Treat unlabeled samples as unresolved, not negative')
ax.legend(fontsize=9, frameon=False, loc='upper right')
ax.set_xlim(-2, 160)

fig.savefig(OUT + 'fig-delayed-feedback-timeline.png')
plt.close(fig)

print("wrote 4 figures to", OUT)
