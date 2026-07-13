import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = "/private/tmp/claude-501/-Users-xingao-Projects-Neurarch/65adcdb4-d882-4642-bf5c-d42bf1be1f61/scratchpad/awesome-ml-system-design/book/content-moderation/assets/"

plt.rcParams.update({
    'figure.dpi': 130, 'font.size': 11,
    'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.autolayout': True
})
BLUE, ORANGE, GREEN, RED, GRAY = '#2563eb', '#ea7317', '#16a34a', '#dc2626', '#64748b'

# ---- 1) Precision-recall curve with per-policy precision floors ----
fig, ax = plt.subplots(figsize=(7, 4.5))
rng = np.random.default_rng(42)

# Simulate a smooth PR curve: as recall rises, precision falls
recall = np.linspace(0.01, 0.99, 300)
# Generic PR curve shape
precision = 1.0 / (1.0 + 4.5 * (recall / (1 - recall + 1e-6)) ** 0.55)
precision = np.clip(precision, 0.01, 0.99)

ax.plot(recall, precision, color=BLUE, lw=2.5, label='PR curve (e.g. hate speech classifier)')

# Per-policy precision floors
policies = [
    ('CSAM / terrorism', 0.98, RED),
    ('Graphic violence', 0.90, ORANGE),
    ('Spam', 0.70, GREEN),
]

for label, p_min, color in policies:
    # Find recall at this precision floor
    idx = np.searchsorted(-precision, -p_min)
    if idx < len(recall):
        r_at_floor = recall[idx]
        ax.axhline(p_min, color=color, ls='--', lw=1.5, alpha=0.8)
        ax.axvline(r_at_floor, color=color, ls=':', lw=1.2, alpha=0.6)
        ax.scatter([r_at_floor], [p_min], color=color, s=80, zorder=5)
        ax.annotate(
            f'{label}\nP_min={p_min:.2f}, R={r_at_floor:.2f}',
            xy=(r_at_floor, p_min),
            xytext=(r_at_floor + 0.04, p_min - 0.07),
            fontsize=8.5, color=color,
            arrowprops=dict(arrowstyle='->', color=color, lw=1)
        )

ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Per-policy operating points: maximize recall\nsubject to a precision floor')
ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9, frameon=False, loc='upper right')
fig.savefig(OUT + 'fig-precision-recall-floor.png')
plt.close(fig)
print('wrote fig-precision-recall-floor.png')

# ---- 2) Cost asymmetry: c_fn vs c_fp per harm type ----
fig, ax = plt.subplots(figsize=(8, 4.2))

harm_types = ['CSAM', 'Terrorism', 'Violence', 'Hate speech', 'Nudity', 'Spam']
c_fn = [1000, 950, 400, 180, 60, 10]   # cost of a miss (false negative)
c_fp = [5,    5,   20,  30,  25, 8]    # cost of a false block

x = np.arange(len(harm_types))
w = 0.38

bars1 = ax.bar(x - w/2, c_fn, w, label='Cost of a miss (false negative)', color=RED, alpha=0.85)
bars2 = ax.bar(x + w/2, c_fp, w, label='Cost of a false block (false positive)', color=BLUE, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(harm_types, fontsize=10)
ax.set_ylabel('Relative cost (illustrative)')
ax.set_title('Asymmetric costs drive per-policy precision floors\n(miss cost >> false-block cost for severe harm)')
ax.legend(fontsize=9, frameon=False)
ax.set_yscale('log')
ax.annotate(
    'CSAM: miss cost is\n200x the false-block cost',
    xy=(0, 1000), xytext=(1.2, 600),
    fontsize=8.5, color=RED,
    arrowprops=dict(arrowstyle='->', color=RED, lw=1)
)
fig.savefig(OUT + 'fig-cost-asymmetry.png')
plt.close(fig)
print('wrote fig-cost-asymmetry.png')

# ---- 3) Harm prevalence and adversarial drift over time ----
fig, ax = plt.subplots(figsize=(8, 4.2))

weeks = np.arange(0, 52)
# spam: initially high, classifier deployed at week 8, then adversarial drift at week 28
spam_base = 0.06
spam = np.where(weeks < 8, spam_base,
       np.where(weeks < 28, spam_base * np.exp(-0.12 * (weeks - 8)),
       spam_base * np.exp(-0.12 * 20) * np.exp(0.10 * (weeks - 28))))
spam = np.clip(spam, 0, 0.1)

# hate speech: slower drift
hate = 0.015 + 0.003 * np.sin(weeks * 0.3) + 0.0001 * weeks

# nudity: relatively stable
nudity = 0.025 + 0.002 * np.random.default_rng(7).normal(0, 1, 52)
nudity = np.clip(nudity, 0.018, 0.035)

ax.plot(weeks, spam * 100, color=RED, lw=2, label='Spam')
ax.plot(weeks, hate * 100, color=ORANGE, lw=2, label='Hate speech')
ax.plot(weeks, nudity * 100, color=BLUE, lw=2, label='Nudity')

ax.axvline(8, color=GRAY, ls='--', lw=1.5)
ax.text(8.5, 5.8, 'Classifier\ndeployed', fontsize=8.5, color=GRAY)

ax.axvline(28, color=RED, ls=':', lw=1.5)
ax.text(28.5, 5.2, 'Adversarial\nevasion', fontsize=8.5, color=RED)

ax.set_xlabel('Week')
ax.set_ylabel('Harm prevalence (%)')
ax.set_title('Harm prevalence over time: spam drifts adversarially,\nnudity stays stable, hate speech varies seasonally')
ax.legend(fontsize=9, frameon=False)
fig.savefig(OUT + 'fig-harm-prevalence.png')
plt.close(fig)
print('wrote fig-harm-prevalence.png')

# ---- 4) Human review funnel ----
fig, ax = plt.subplots(figsize=(7, 4.5))

stages = [
    'All content ingested',
    'After hash match\n(known-bad filtered)',
    'After cheap classifier\n(confident auto-action)',
    'Borderline: routed\nto human review',
    'Reviewed by\nhumans',
]
volumes = [1_000_000, 980_000, 50_000, 15_000, 3_000]
colors = [BLUE, ORANGE, GREEN, RED, GRAY]

y_pos = np.arange(len(stages))
bars = ax.barh(y_pos, volumes, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)

for bar, vol in zip(bars, volumes):
    label = f'{vol:,}' if vol < 100_000 else f'{vol/1_000:.0f}k'
    ax.text(bar.get_width() + 5_000, bar.get_y() + bar.get_height() / 2,
            label, va='center', fontsize=9, color=GRAY)

ax.set_yticks(y_pos)
ax.set_yticklabels(stages, fontsize=9.5)
ax.set_xlabel('Items per hour (illustrative)')
ax.set_title('Human review funnel: cheap gates protect\nscarce reviewer capacity')
ax.set_xlim(0, 1_200_000)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
ax.grid(axis='y', alpha=0)
fig.savefig(OUT + 'fig-human-review-funnel.png')
plt.close(fig)
print('wrote fig-human-review-funnel.png')

print('all 4 figures written to', OUT)
