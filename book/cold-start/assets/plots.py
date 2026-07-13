import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from math import lgamma, exp

def beta_pdf(x, a, b):
    """Evaluate Beta(a,b) pdf at x using lgamma for numerical stability."""
    log_norm = lgamma(a + b) - lgamma(a) - lgamma(b)
    result = np.empty_like(x)
    for i, xi in enumerate(x):
        if xi <= 0 or xi >= 1:
            result[i] = 0.0
        else:
            result[i] = exp(log_norm + (a - 1) * np.log(xi) + (b - 1) * np.log(1 - xi))
    return result

OUT = "/private/tmp/claude-501/-Users-xingao-Projects-Neurarch/65adcdb4-d882-4642-bf5c-d42bf1be1f61/scratchpad/awesome-ml-system-design/book/cold-start/assets/"

plt.rcParams.update({
    'figure.dpi': 130,
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.autolayout': True,
})
BLUE, ORANGE, GREEN, RED, GRAY = '#2563eb', '#ea7317', '#16a34a', '#dc2626', '#64748b'

# ---- 1) Cumulative regret: epsilon-greedy vs UCB vs Thompson -----------------
rng = np.random.default_rng(42)
T = 2000
t = np.arange(1, T + 1)

# Epsilon-greedy: linear regret (constant epsilon tax)
eps_regret = 0.18 * t + rng.normal(0, 3, T).cumsum() * 0.3
eps_regret = np.maximum.accumulate(eps_regret)

# UCB: O(log t) regret
ucb_regret = 4.0 * np.log(t) + rng.normal(0, 1, T).cumsum() * 0.15
ucb_regret = np.maximum.accumulate(ucb_regret)

# Thompson: similar log growth but slightly lower constant
ts_regret = 3.2 * np.log(t) + rng.normal(0, 1, T).cumsum() * 0.12
ts_regret = np.maximum.accumulate(ts_regret)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t, eps_regret, color=RED, lw=2, label='epsilon-greedy (linear)')
ax.plot(t, ucb_regret, color=BLUE, lw=2, label='UCB (log t)')
ax.plot(t, ts_regret, color=GREEN, lw=2, label='Thompson sampling (log t)')
ax.set_xlabel('time steps (requests served)')
ax.set_ylabel('cumulative regret')
ax.set_title('Cumulative regret: epsilon-greedy vs UCB vs Thompson\n(illustrative; directed methods grow logarithmically)')
ax.legend(fontsize=9, frameon=False)
ax.annotate(
    'flat tax\nevery step',
    xy=(1600, eps_regret[1599]),
    xytext=(1200, eps_regret[1599] - 120),
    fontsize=9, color=RED,
    arrowprops=dict(arrowstyle='->', color=RED),
)
ax.annotate(
    'spend where\nuncertain',
    xy=(1600, ts_regret[1599]),
    xytext=(1100, ts_regret[1599] + 60),
    fontsize=9, color=GREEN,
    arrowprops=dict(arrowstyle='->', color=GREEN),
)
fig.savefig(OUT + 'fig-regret-comparison.png')
plt.close(fig)

# ---- 2) UCB confidence bounds narrowing over pulls --------------------------
pulls_axis = np.arange(1, 201)
mu_true = 0.6
alpha_ucb = 1.0

# Mean estimate converges, bound shrinks
mean_est = mu_true + rng.normal(0, 1, 200) / np.sqrt(pulls_axis) * 0.4
ucb_upper = mean_est + alpha_ucb * np.sqrt(np.log(500) / pulls_axis)
ucb_lower = mean_est - alpha_ucb * np.sqrt(np.log(500) / pulls_axis)

fig, ax = plt.subplots(figsize=(7, 4))
ax.fill_between(pulls_axis, ucb_lower, ucb_upper, color=BLUE, alpha=0.2, label='confidence interval')
ax.plot(pulls_axis, mean_est, color=BLUE, lw=2, label='estimated mean')
ax.axhline(mu_true, color=GRAY, ls='--', lw=1.5, label='true mean (0.6)')
ax.axhline(ucb_upper[0], color=ORANGE, ls=':', lw=1.5)
ax.annotate(
    'wide bound\npromotes early exploration',
    xy=(5, ucb_upper[4]),
    xytext=(30, ucb_upper[4] + 0.08),
    fontsize=9, color=ORANGE,
    arrowprops=dict(arrowstyle='->', color=ORANGE),
)
ax.annotate(
    'bound narrows\nas arm is pulled',
    xy=(150, ucb_upper[149]),
    xytext=(100, ucb_upper[149] + 0.15),
    fontsize=9, color=BLUE,
    arrowprops=dict(arrowstyle='->', color=BLUE),
)
ax.set_xlabel('number of pulls for this arm')
ax.set_ylabel('reward estimate')
ax.set_title('UCB confidence bounds narrow with each pull\n(arm is explored less as uncertainty drops)')
ax.legend(fontsize=9, frameon=False)
ax.set_ylim(0.0, 1.3)
fig.savefig(OUT + 'fig-ucb-bounds.png')
plt.close(fig)

# ---- 3) Beta posteriors for Thompson sampling --------------------------------
x = np.linspace(0.001, 0.999, 400)

configs = [
    ('Arm A: prior (1 pull, 0 wins)', 1, 1, GRAY, '--'),
    ('Arm B: 5 pulls, 1 win', 2, 5, RED, '-'),
    ('Arm C: 20 pulls, 12 wins', 13, 9, BLUE, '-'),
    ('Arm D: 50 pulls, 40 wins (near warm)', 41, 11, GREEN, '-'),
]

fig, ax = plt.subplots(figsize=(7, 4))
for label, a, b, color, ls in configs:
    y = beta_pdf(x, a, b)
    ax.plot(x, y, color=color, lw=2, ls=ls, label=label)

ax.set_xlabel('estimated success probability')
ax.set_ylabel('posterior density')
ax.set_title('Beta posteriors for Thompson sampling\n(wide = uncertain = explored often; narrow = confident)')
ax.legend(fontsize=8.5, frameon=False, loc='upper left')
ax.set_xlim(0, 1)
ax.set_ylim(0, None)
fig.savefig(OUT + 'fig-beta-posteriors.png')
plt.close(fig)

# ---- 4) Exploration rate vs short-term vs long-term reward -------------------
eps_range = np.linspace(0, 1, 200)

# Short-term reward falls as epsilon rises (more random serves)
short_term = 0.85 * (1 - eps_range) + 0.3 * eps_range

# Long-term reward has an optimum at moderate epsilon (better model from exploration)
long_term = 0.55 + 0.4 * np.exp(-((eps_range - 0.15) ** 2) / (2 * 0.07 ** 2))

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(eps_range, short_term, color=RED, lw=2, label='short-term reward (this session)')
ax.plot(eps_range, long_term, color=GREEN, lw=2, label='long-term reward (corpus growth)')
ax.axvline(0.15, color=GRAY, ls='--', lw=1.2)
ax.text(0.17, 0.35, 'optimal\nexploration rate', color=GRAY, fontsize=9)
ax.set_xlabel('exploration rate (fraction of impressions exploring)')
ax.set_ylabel('reward')
ax.set_title('Explore-exploit tradeoff\n(exploration costs short-term; pays off long-term)')
ax.legend(fontsize=9, frameon=False)
ax.set_ylim(0.2, 1.0)
fig.savefig(OUT + 'fig-explore-exploit-tradeoff.png')
plt.close(fig)

print("wrote 4 figures")
