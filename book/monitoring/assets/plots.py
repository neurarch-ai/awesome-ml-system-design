import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = "/private/tmp/claude-501/-Users-xingao-Projects-Neurarch/65adcdb4-d882-4642-bf5c-d42bf1be1f61/scratchpad/awesome-ml-system-design/book/monitoring/assets/"

plt.rcParams.update({
    'figure.dpi': 130, 'font.size': 11,
    'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.autolayout': True
})
BLUE, ORANGE, GREEN, RED, GRAY = '#2563eb', '#ea7317', '#16a34a', '#dc2626', '#64748b'

# -----------------------------------------------------------------------
# 1) Drifting feature distribution over time
#    Three snapshots of a feature histogram: training, 3 months, 6 months
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
rng = np.random.default_rng(42)
x = np.linspace(-3, 7, 300)

def gauss(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))

ax.fill_between(x, gauss(x, 1.0, 1.0), alpha=0.35, color=BLUE,  label='training window (reference)')
ax.fill_between(x, gauss(x, 1.8, 1.1), alpha=0.35, color=ORANGE, label='3 months later')
ax.fill_between(x, gauss(x, 3.0, 1.3), alpha=0.35, color=RED,   label='6 months later')
ax.plot(x, gauss(x, 1.0, 1.0), color=BLUE,   lw=1.5)
ax.plot(x, gauss(x, 1.8, 1.1), color=ORANGE, lw=1.5)
ax.plot(x, gauss(x, 3.0, 1.3), color=RED,    lw=1.5)

ax.set_xlabel('feature value')
ax.set_ylabel('density')
ax.set_title('Data drift: a feature distribution shifts over time')
ax.legend(fontsize=9, frameon=False)
fig.savefig(OUT + 'fig-drift-distributions.png')
plt.close(fig)
print('wrote fig-drift-distributions.png')

# -----------------------------------------------------------------------
# 2) PSI over time with thresholds
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
weeks = np.arange(0, 26)
rng2 = np.random.default_rng(7)
# PSI stays low, then rises after a simulated event at week 14
psi_base = rng2.uniform(0.02, 0.07, 14)
psi_after = np.linspace(0.09, 0.32, 12) + rng2.uniform(-0.02, 0.02, 12)
psi = np.concatenate([psi_base, psi_after])

ax.plot(weeks, psi, color=BLUE, lw=2, marker='o', markersize=4, label='PSI (feature A)')
ax.axhline(0.10, color=ORANGE, ls='--', lw=1.5, label='warn threshold (0.10)')
ax.axhline(0.25, color=RED,    ls='--', lw=1.5, label='alert threshold (0.25)')
ax.fill_between(weeks, 0, psi, where=(psi >= 0.25), color=RED,    alpha=0.15)
ax.fill_between(weeks, 0, psi, where=(psi >= 0.10) & (psi < 0.25), color=ORANGE, alpha=0.15)
ax.annotate('traffic shift event', xy=(14, psi[14]), xytext=(10, 0.28),
            fontsize=9, color=GRAY,
            arrowprops=dict(arrowstyle='->', color=GRAY))
ax.set_xlabel('weeks since deploy')
ax.set_ylabel('PSI')
ax.set_title('PSI over time: stable, then breaching warn and alert thresholds')
ax.legend(fontsize=9, frameon=False)
ax.set_ylim(0, 0.38)
fig.savefig(OUT + 'fig-psi-over-time.png')
plt.close(fig)
print('wrote fig-psi-over-time.png')

# -----------------------------------------------------------------------
# 3) Model quality decay curve
#    AUC declining over months; a retrain event restores quality
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
months = np.linspace(0, 12, 200)
# Exponential decay, with a retrain bump at month 7
auc_base = 0.82
decay = 0.025  # per month

def auc_curve(t, t_retrain=7.0):
    # pre-retrain segment
    pre = auc_base * np.exp(-decay * np.minimum(t, t_retrain))
    # post-retrain: starts fresh from auc_base again
    post = auc_base * np.exp(-decay * np.maximum(t - t_retrain, 0))
    return np.where(t <= t_retrain, pre, post)

auc = auc_curve(months)
rng3 = np.random.default_rng(99)
noise = rng3.normal(0, 0.003, len(months))
auc_noisy = auc + noise

ax.plot(months, auc_noisy, color=BLUE, lw=2, label='model AUC (weekly eval)')
ax.axvline(7, color=GREEN, ls='--', lw=1.5, label='retrain + promote')
ax.axhline(0.79, color=RED, ls=':', lw=1.5, label='minimum acceptable AUC')
ax.annotate('retrain\nrestores\nquality', xy=(7, auc_curve(7)), xytext=(8.2, 0.815),
            fontsize=9, color=GREEN,
            arrowprops=dict(arrowstyle='->', color=GREEN))
ax.set_xlabel('months after last training')
ax.set_ylabel('AUC')
ax.set_title('Model quality decays without retraining; triggered retrain recovers it')
ax.legend(fontsize=9, frameon=False)
ax.set_ylim(0.76, 0.85)
fig.savefig(OUT + 'fig-quality-decay.png')
plt.close(fig)
print('wrote fig-quality-decay.png')

# -----------------------------------------------------------------------
# 4) Label delay timeline
#    Shows events, label-arrival windows, and monitoring layers
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))

ax.set_xlim(-0.5, 32)
ax.set_ylim(-0.5, 4.5)
ax.axis('off')

# Horizontal timeline axis
ax.annotate('', xy=(31.5, 0), xytext=(-0.3, 0),
            arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.5))
ax.text(31.8, 0, 'time', va='center', color=GRAY, fontsize=10)
ax.text(0, -0.4, 't=0\n(event)', ha='center', fontsize=8, color=GRAY)

# Tick marks
for x_tick, label in [(0, ''), (1, '~1s'), (3, '~hours'), (14, '~weeks'), (30, '~months')]:
    ax.axvline(x_tick, ymin=0.07, ymax=0.13, color=GRAY, lw=1)
    if label:
        ax.text(x_tick, -0.4, label, ha='center', fontsize=8, color=GRAY)

# Row 1: Click labels (very fast)
y = 1
ax.barh(y, 1.5, left=0, height=0.4, color=BLUE, alpha=0.8)
ax.text(1.7, y, 'click / watch: labels in ~1 s', va='center', fontsize=9)

# Row 2: Conversion / purchase (hours)
y = 2
ax.barh(y, 4, left=0, height=0.4, color=ORANGE, alpha=0.8)
ax.text(4.2, y, 'purchase / sign-up: hours', va='center', fontsize=9)

# Row 3: Churn (weeks)
y = 3
ax.barh(y, 14, left=0, height=0.4, color=RED, alpha=0.8)
ax.text(14.2, y, 'churn / cancellation: weeks', va='center', fontsize=9)

# Row 4: Loan default / dispute (months)
y = 4
ax.barh(y, 30, left=0, height=0.4, color=GRAY, alpha=0.8)
ax.text(30.2, y, 'loan default / fraud dispute: months', va='center', fontsize=8)

# Proxy monitoring marker
ax.axvline(0.01, ymin=0.18, ymax=0.92, color=GREEN, lw=2, ls='--')
ax.text(0.2, 4.3, 'drift proxies available immediately', fontsize=8, color=GREEN)

ax.set_title('Label delay: how long until truth arrives', pad=12)
fig.savefig(OUT + 'fig-label-delay.png')
plt.close(fig)
print('wrote fig-label-delay.png')

print('All 4 figures written.')
