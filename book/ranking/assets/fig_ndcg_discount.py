import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = "/private/tmp/claude-501/-Users-xingao-Projects-Neurarch/65adcdb4-d882-4642-bf5c-d42bf1be1f61/scratchpad/awesome-ml-system-design/book/ranking/assets/"

plt.rcParams.update({
    'figure.dpi': 130, 'font.size': 11, 'axes.grid': True,
    'grid.alpha': 0.3, 'axes.spines.top': False,
    'axes.spines.right': False, 'figure.autolayout': True
})
BLUE, ORANGE, GREEN, RED, GRAY = '#2563eb', '#ea7317', '#16a34a', '#dc2626', '#64748b'

# ── NDCG computation: two panels ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

# Panel 1: position discount 1/log2(i+1) for ranks 1..10
ranks = np.arange(1, 11)
discounts = 1.0 / np.log2(ranks + 1)

ax = axes[0]
ax.plot(ranks, discounts, 'o-', color=BLUE, lw=2.2, ms=7)
ax.fill_between(ranks, 0, discounts, alpha=0.12, color=BLUE)
ax.set_xlabel('rank position i')
ax.set_ylabel(r'position discount  $1 / \log_2(i+1)$')
ax.set_title('Position discount decreases with rank\n(rank 1 receives the highest weight)')
ax.set_xticks(ranks)
ax.set_ylim(0, 1.08)

for r, d in zip(ranks, discounts):
    ax.annotate(f'{d:.2f}', (r, d), textcoords='offset points',
                xytext=(0, 7), ha='center', fontsize=7.5, color=GRAY)

# Panel 2: DCG vs IDCG for an example list
# Actual order: item relevances at positions 1..5
actual_rel = np.array([2, 0, 3, 1, 2], dtype=float)   # actual ranking
ideal_rel  = np.array([3, 2, 2, 1, 0], dtype=float)   # ideal (sorted desc)
pos = np.arange(1, 6)
disc5 = 1.0 / np.log2(pos + 1)

dcg_terms  = actual_rel * disc5
idcg_terms = ideal_rel  * disc5
dcg  = dcg_terms.sum()
idcg = idcg_terms.sum()
ndcg = dcg / idcg

ax = axes[1]
w = 0.35
bars_dcg  = ax.bar(pos - w/2, dcg_terms,  w, label=f'DCG terms (sum = {dcg:.2f})',
                   color=BLUE,  alpha=0.82)
bars_idcg = ax.bar(pos + w/2, idcg_terms, w, label=f'IDCG terms (sum = {idcg:.2f})',
                   color=GREEN, alpha=0.82)

ax.set_xlabel('rank position i')
ax.set_ylabel(r'$r_i \;/\; \log_2(i+1)$')
ax.set_title(f'DCG vs IDCG (example)\n'
             r'NDCG = DCG / IDCG = '
             f'{dcg:.2f} / {idcg:.2f} = {ndcg:.2f}')
ax.set_xticks(pos)
ax.legend(fontsize=9, frameon=False)

# label relevance above each bar
for p, ar, ir in zip(pos, actual_rel, ideal_rel):
    ax.text(p - w/2, dcg_terms[p-1] + 0.01,  f'r={int(ar)}',
            ha='center', va='bottom', fontsize=7.5, color=BLUE)
    ax.text(p + w/2, idcg_terms[p-1] + 0.01, f'r={int(ir)}',
            ha='center', va='bottom', fontsize=7.5, color=GREEN)

ax.text(0.97, 0.96,
        f'NDCG = {ndcg:.2f}',
        transform=ax.transAxes, ha='right', va='top',
        fontsize=12, fontweight='bold', color=RED,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor=RED, alpha=0.9))

fig.suptitle(
    r'How NDCG is computed: position discount $1/\log_2(i+1)$ and DCG/IDCG (illustrative)',
    fontsize=11, y=1.01
)
fig.savefig(OUT + 'fig-ndcg-discount.png', bbox_inches='tight')
plt.close(fig)
print("fig-ndcg-discount.png done")
