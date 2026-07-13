import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = "/private/tmp/claude-501/-Users-xingao-Projects-Neurarch/65adcdb4-d882-4642-bf5c-d42bf1be1f61/scratchpad/awesome-ml-system-design/book/forecasting/assets/"

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

# ---------------------------------------------------------------
# 1) Pinball loss per quantile
#    x-axis: residual r = y - q_hat
#    Positive residual (r > 0): y > q_hat => forecast was too LOW => UNDER-predict
#    Negative residual (r < 0): y < q_hat => forecast was too HIGH => OVER-predict
# ---------------------------------------------------------------
r = np.linspace(-3, 3, 500)

def pinball(r, tau):
    """Pinball loss as a function of residual r = y - q_hat and quantile tau."""
    return np.where(r >= 0, tau * r, (tau - 1) * r)

fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=False)

configs = [
    (0.10, RED,    'P10 (tau=0.10)\nover-predict penalized heavily'),
    (0.50, GRAY,   'P50 (tau=0.50)\nsymmetric = MAE'),
    (0.90, BLUE,   'P90 (tau=0.90)\nunder-predict penalized heavily'),
]

for ax, (tau, col, title) in zip(axes, configs):
    loss = pinball(r, tau)
    ax.plot(r, loss, color=col, lw=2.5)
    ax.set_xlabel('Residual r = y - q_hat')
    ax.set_ylabel('Pinball loss')
    ax.set_title(title, fontsize=9)

    # Shade regions
    ax.fill_between(r[r >= 0], 0, pinball(r[r >= 0], tau), alpha=0.15, color=GREEN,
                    label='under-predict region')
    ax.fill_between(r[r <= 0], 0, pinball(r[r <= 0], tau), alpha=0.15, color=RED,
                    label='over-predict region')

    # Vertical line at r=0
    ax.axvline(0, color=GRAY, lw=1, ls='--')

    # Annotations: positive residual (right) = under-predict; negative (left) = over-predict
    ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 2.5
    ax.text(1.5, ymax * 0.65, 'under-predict\n(r > 0)', ha='center', fontsize=8,
            color=GREEN, fontweight='bold')
    ax.text(-1.5, ymax * 0.65, 'over-predict\n(r < 0)', ha='center', fontsize=8,
            color=RED, fontweight='bold')

fig.suptitle(
    'Pinball loss per quantile level\n'
    '(x-axis: residual = y - forecast; positive residual = forecast too low = under-predict)',
    fontsize=10
)
fig.savefig(OUT + 'fig-pinball-loss.png')
plt.close(fig)

print("wrote fig-pinball-loss.png to", OUT)
