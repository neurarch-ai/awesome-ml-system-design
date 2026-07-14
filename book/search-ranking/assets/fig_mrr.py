import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = "/private/tmp/claude-501/-Users-xingao-Projects-Neurarch/65adcdb4-d882-4642-bf5c-d42bf1be1f61/scratchpad/awesome-ml-system-design/book/search-ranking/assets/"

plt.rcParams.update({
    'figure.dpi': 130, 'font.size': 11, 'axes.grid': True,
    'grid.alpha': 0.3, 'axes.spines.top': False,
    'axes.spines.right': False, 'figure.autolayout': True
})
BLUE, ORANGE, GREEN, RED, GRAY = '#2563eb', '#ea7317', '#16a34a', '#dc2626', '#64748b'

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

# Panel 1: reciprocal rank curve
ranks = np.arange(1, 11)
rr    = 1.0 / ranks

ax = axes[0]
ax.plot(ranks, rr, 'o-', color=BLUE, lw=2.2, ms=7)
ax.fill_between(ranks, 0, rr, alpha=0.12, color=BLUE)
ax.set_xlabel('rank of first relevant result')
ax.set_ylabel('reciprocal rank  1 / rank')
ax.set_title('Reciprocal rank falls sharply as the\nrelevant result moves down the list')
ax.set_xticks(ranks)
ax.set_ylim(0, 1.15)

for r, v in zip(ranks, rr):
    ax.annotate(f'1/{r}' if r <= 6 else f'{v:.2f}',
                (r, v), textcoords='offset points',
                xytext=(0, 7), ha='center', fontsize=8, color=GRAY)

# Panel 2: three illustrative queries showing how MRR averages
ax = axes[1]
ax.axis('off')
ax.set_xlim(0, 10); ax.set_ylim(-0.3, 5.5)
ax.set_title('MRR = mean of reciprocal ranks across queries', pad=8)

query_data = [
    ('Query 1', [('doc A', False), ('doc B', True),  ('doc C', False)], 2),
    ('Query 2', [('doc X', True),  ('doc Y', False), ('doc Z', False)], 1),
    ('Query 3', [('doc P', False), ('doc Q', False), ('doc R', True)],  3),
]

row_h = 1.45
for qi, (qname, docs, first_rank) in enumerate(query_data):
    y_top = 5.1 - qi * row_h
    ax.text(0.1, y_top, qname, fontsize=10, fontweight='bold', color=GRAY, va='top')
    for di, (dname, is_rel) in enumerate(docs):
        col = GREEN if is_rel else '#94a3b8'
        marker = '✓' if is_rel else '·'
        ax.text(2.0 + di * 2.5, y_top - 0.45,
                f'rank {di+1}  {dname}  {marker}',
                fontsize=9, color=col, va='center')
    rr_val = 1.0 / first_rank
    ax.text(8.5, y_top - 0.45,
            f'1/{first_rank} = {rr_val:.2f}',
            fontsize=9.5, color=BLUE, fontweight='bold', va='center', ha='center')

# separator and MRR summary
ax.axhline(0.45, xmin=0.02, xmax=0.98, color=GRAY, lw=0.8, ls='--')
mrr = (1/2 + 1/1 + 1/3) / 3
ax.text(5.0, 0.1,
        f'MRR = (1/2 + 1/1 + 1/3) / 3 = {mrr:.3f}',
        fontsize=11, color=RED, fontweight='bold', ha='center', va='center')

fig.suptitle(
    'How MRR is computed: reciprocal rank of the first relevant result, averaged over queries (illustrative)',
    fontsize=10.5, y=1.01
)
fig.savefig(OUT + 'fig-mrr-reciprocal.png', bbox_inches='tight')
plt.close(fig)
print("fig-mrr-reciprocal.png done")
