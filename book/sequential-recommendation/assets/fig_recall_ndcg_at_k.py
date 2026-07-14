import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = "/private/tmp/claude-501/-Users-xingao-Projects-Neurarch/65adcdb4-d882-4642-bf5c-d42bf1be1f61/scratchpad/awesome-ml-system-design/book/sequential-recommendation/assets/"

plt.rcParams.update({
    'figure.dpi': 130, 'font.size': 11, 'axes.grid': True,
    'grid.alpha': 0.3, 'axes.spines.top': False,
    'axes.spines.right': False, 'figure.autolayout': True
})
BLUE, ORANGE, GREEN, RED, GRAY = '#2563eb', '#ea7317', '#16a34a', '#dc2626', '#64748b'

ks = np.array([1, 2, 5, 10, 20, 50, 100])

# Illustrative curves: recall rises and saturates, NDCG rises more slowly
# (NDCG cares about rank position, so it saturates more slowly relative to recall)
recall = 1 - np.exp(-ks / 18.0)
recall = np.clip(recall * 0.87 + 0.02, 0, 0.90)

# NDCG for next-item = 1/log2(rank+1) if item in top-k, so it grows slower
ndcg = 1 - np.exp(-ks / 30.0)
ndcg = np.clip(ndcg * 0.68 + 0.05, 0, 0.75)

# Downstream k
k_downstream = 20
recall_at_kd = np.interp(k_downstream, ks, recall)
ndcg_at_kd   = np.interp(k_downstream, ks, ndcg)

fig, ax = plt.subplots(figsize=(7.5, 4.8))

ax.plot(ks, recall, 'o-', color=BLUE,   lw=2.2, ms=7, label='Recall@k')
ax.plot(ks, ndcg,   's-', color=ORANGE, lw=2.2, ms=7, label='NDCG@k')

# vertical line at downstream k
ax.axvline(k_downstream, color=RED, lw=1.5, ls='--', alpha=0.8)
ax.annotate(f'k = {k_downstream}\n(passed to ranking)',
            xy=(k_downstream, 0.62),
            xytext=(k_downstream + 6, 0.50),
            fontsize=8.5, color=RED,
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.2))

# mark the values at k_downstream
ax.scatter([k_downstream], [recall_at_kd], color=BLUE,   s=80, zorder=5)
ax.scatter([k_downstream], [ndcg_at_kd],   color=ORANGE, s=80, zorder=5)
ax.annotate(f'Recall@{k_downstream} = {recall_at_kd:.2f}',
            xy=(k_downstream, recall_at_kd),
            xytext=(k_downstream + 7, recall_at_kd + 0.04),
            fontsize=8, color=BLUE,
            arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.0))
ax.annotate(f'NDCG@{k_downstream} = {ndcg_at_kd:.2f}',
            xy=(k_downstream, ndcg_at_kd),
            xytext=(k_downstream + 7, ndcg_at_kd - 0.06),
            fontsize=8, color=ORANGE,
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.0))

ax.set_xlabel('k (list length)')
ax.set_ylabel('metric value')
ax.set_xscale('log')
ax.set_xticks(ks)
ax.set_xticklabels(ks)
ax.set_ylim(0, 0.95)
ax.set_title(
    'Recall@k rises and saturates with k; NDCG@k rises more slowly\n'
    '(NDCG penalises the true item appearing near the bottom of the top-k list)'
)
ax.legend(fontsize=10, frameon=False)

ax.text(0.02, 0.97,
        'Evaluate at the k\nyou pass downstream',
        transform=ax.transAxes, fontsize=8, color=RED,
        va='top', ha='left', style='italic')

fig.suptitle(
    'Recall@k and NDCG@k vs k for next-item recommendation (illustrative)',
    fontsize=11, y=1.01
)
fig.savefig(OUT + 'fig-recall-ndcg-at-k.png', bbox_inches='tight')
plt.close(fig)
print("fig-recall-ndcg-at-k.png done")
