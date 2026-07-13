import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

BASE = "/private/tmp/claude-501/-Users-xingao-Projects-Neurarch/65adcdb4-d882-4642-bf5c-d42bf1be1f61/scratchpad/awesome-ml-system-design/book/ranking/assets/"

plt.rcParams.update({
    'figure.dpi': 130, 'font.size': 11, 'axes.grid': True,
    'grid.alpha': 0.3, 'axes.spines.top': False,
    'axes.spines.right': False, 'figure.autolayout': True
})
BLUE, ORANGE, GREEN, RED, GRAY = '#2563eb', '#ea7317', '#16a34a', '#dc2626', '#64748b'

# ------------------------------------------------------------------
# Figure 1: Calibration reliability diagram
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 5.2))
p = np.linspace(0, 1, 200)

ax.plot(p, p,
        color=GRAY, lw=1.5, ls='--', label='perfect calibration')
ax.plot(p, np.clip(p * 0.58 + 0.04, 0, 1),
        color=RED, lw=2.2, label='over-confident (bows below diagonal)')
ax.plot(p, np.clip(p ** 0.52, 0, 1),
        color=ORANGE, lw=2.2, label='under-confident (bows above diagonal)')
ax.plot(p, np.clip(p * 0.94 + 0.025, 0, 1),
        color=GREEN, lw=2.2, label='well-calibrated (close to diagonal)')

ax.fill_between(p, p, np.clip(p * 0.58 + 0.04, 0, 1),
                alpha=0.07, color=RED)
ax.fill_between(p, p, np.clip(p ** 0.52, 0, 1),
                alpha=0.07, color=ORANGE)

ax.set_xlabel('mean predicted probability (confidence)')
ax.set_ylabel('fraction of positives (actual rate)')
ax.set_title('Calibration reliability diagram\n'
             'ECE = area between curve and diagonal')
ax.legend(fontsize=9, frameon=False, loc='upper left')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.annotate('over-confident model\nbows toward x-axis',
            xy=(0.7, 0.45), xytext=(0.45, 0.25),
            fontsize=8, color=RED,
            arrowprops=dict(arrowstyle='->', color=RED, lw=1))
fig.savefig(BASE + 'fig-calibration.png')
plt.close(fig)
print("fig-calibration.png done")

# ------------------------------------------------------------------
# Figure 2: NDCG@k curves for three ranking approaches
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.2))
ks = np.array([1, 2, 3, 5, 10, 20, 50])

# Illustrative NDCG@k values; LambdaMART optimizes NDCG change per pair
lambda_mart = np.array([0.47, 0.54, 0.58, 0.63, 0.69, 0.72, 0.74])
deep_dlrm   = np.array([0.44, 0.51, 0.55, 0.61, 0.67, 0.70, 0.72])
pointwise   = np.array([0.38, 0.45, 0.49, 0.55, 0.62, 0.66, 0.69])

ax.plot(ks, lambda_mart, marker='o', color=BLUE,   lw=2,
        label='LambdaMART (NDCG-weighted gradient)')
ax.plot(ks, deep_dlrm,   marker='s', color=ORANGE, lw=2,
        label='DLRM / DCN (pairwise cross-entropy)')
ax.plot(ks, pointwise,   marker='^', color=RED,    lw=2,
        label='pointwise log-loss (treats items independently)')

ax.set_xlabel('k'); ax.set_ylabel('NDCG@k')
ax.set_xscale('log'); ax.set_xticks(ks); ax.set_xticklabels(ks)
ax.set_ylim(0.3, 0.80)
ax.set_title('NDCG@k vs k by optimization approach\n'
             '(LambdaMART scales each gradient by how much swapping two items moves NDCG)')
ax.legend(fontsize=9, frameon=False)
fig.savefig(BASE + 'fig-ndcg-k.png')
plt.close(fig)
print("fig-ndcg-k.png done")

# ------------------------------------------------------------------
# Figure 3: Pointwise / pairwise / listwise illustration
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
item_colors = [BLUE, ORANGE, GREEN, RED]
items = ['A', 'B', 'C', 'D']

# Panel 1: Pointwise (score each item independently)
ax = axes[0]
scores_pw = [0.80, 0.30, 0.62, 0.18]
bar_cols = [BLUE, ORANGE, GREEN, RED]
ax.barh(['D', 'C', 'B', 'A'], [scores_pw[3], scores_pw[2], scores_pw[1], scores_pw[0]],
        color=[RED, GREEN, ORANGE, BLUE], alpha=0.8, height=0.6)
ax.set_xlim(0, 1)
ax.set_xlabel('predicted P(click)')
ax.set_title('Pointwise\n(score each item\nindependently)', fontsize=10)
ax.axvline(0.5, color=GRAY, ls='--', lw=1)
ax.text(0.52, 0.08, 'threshold', color=GRAY, fontsize=8,
        transform=ax.get_xaxis_transform())

# Panel 2: Pairwise (which item wins?)
ax = axes[1]
ax.set_xlim(0, 4); ax.set_ylim(-0.3, 4.2)
ax.axis('off')
ax.set_title('Pairwise\n(which of two should\nrank higher?)', fontsize=10)
pairs = [(0, 3, 'A', 'D'),
         (0, 1, 'A', 'B'),
         (2, 1, 'C', 'B')]
for row, (wi, li, wl, ll) in enumerate(pairs):
    y = 3.5 - row * 1.1
    ax.text(0.35, y, f'item {wl}', fontsize=11,
            color=item_colors[wi], fontweight='bold', va='center')
    ax.annotate('',
                xy=(2.2, y), xytext=(1.5, y),
                arrowprops=dict(arrowstyle='->', color=GRAY, lw=2))
    ax.text(2.4, y, f'item {ll}', fontsize=11,
            color=item_colors[li], fontweight='bold', va='center')
    ax.text(1.85, y + 0.28, 'beats', fontsize=8, color=GRAY, ha='center')

# Panel 3: Listwise (optimize whole list, gradient scaled by NDCG change)
ax = axes[2]
ax.set_xlim(0, 3.4); ax.set_ylim(-0.2, 4.8)
ax.axis('off')
ax.set_title('Listwise\n(optimize the ordered\nlist as a whole)', fontsize=10)
list_order = [('A', BLUE, 1.00),
              ('C', GREEN, 0.63),
              ('B', ORANGE, 0.50),
              ('D', RED, 0.43)]
for rank, (lbl, col, wt) in enumerate(list_order):
    y = 4.2 - rank * 1.0
    ax.text(0.05, y, f'rank {rank+1}', fontsize=9, color=GRAY, va='center')
    ax.text(0.85, y, f'item {lbl}', fontsize=11,
            color=col, fontweight='bold', va='center')
    bar_w = wt * 1.1
    rect = FancyBboxPatch((1.6, y - 0.22), bar_w, 0.44,
                           boxstyle='round,pad=0.05',
                           facecolor=col, edgecolor='none', alpha=0.55)
    ax.add_patch(rect)
    ax.text(1.6 + bar_w + 0.05, y, f'wt {wt:.2f}',
            fontsize=8, color=GRAY, va='center')
ax.text(1.0, -0.12, 'gradient scales by NDCG-change per pair',
        fontsize=7.5, color=GRAY, style='italic')

plt.suptitle('Three ranking learning approaches (illustrative)', fontsize=12, y=1.02)
fig.savefig(BASE + 'fig-ranking-approaches.png', bbox_inches='tight')
plt.close(fig)
print("fig-ranking-approaches.png done")

# ------------------------------------------------------------------
# Figure 4: DLRM feature interaction schematic
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 5.5))
ax.axis('off')
ax.set_xlim(0, 10); ax.set_ylim(0, 8)
ax.set_title(
    'DLRM feature interaction layer: sits after embeddings, before top MLP (illustrative)',
    fontsize=10)

def draw_box(cx, cy, w, h, color, text, fontsize=8.5):
    rect = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle='round,pad=0.12',
        facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.88)
    ax.add_patch(rect)
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fontsize, color='white', fontweight='bold',
            multialignment='center')

def draw_arrow(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#94a3b8', lw=1.6))

# Dense features -> bottom MLP
draw_box(1.2, 7.3, 1.7, 0.7, GRAY,   'dense\nfeatures')
draw_box(1.2, 5.9, 1.7, 0.7, ORANGE, 'bottom\nMLP')
draw_arrow(1.2, 6.95, 1.2, 6.25)

# Three sparse feature columns
sparse_xs = [3.4, 5.1, 6.8]
for i, xi in enumerate(sparse_xs):
    label = f'sparse {i + 1}' if i < 2 else 'sparse N'
    draw_box(xi, 7.3, 1.5, 0.7, GRAY, label)
    draw_box(xi, 5.9, 1.5, 0.7, BLUE, 'embedding\ntable')
    draw_arrow(xi, 6.95, xi, 6.25)

# Interaction layer
int_y = 4.5
draw_box(4.5, int_y, 8.4, 0.85, RED,
         'explicit pairwise dot products  (all pairs of embedding vectors concatenated)',
         fontsize=8)
draw_arrow(1.2, 5.55, 2.0, int_y + 0.4)
for xi in sparse_xs:
    tx = 3.8 + (xi - 5.1) * 0.15
    draw_arrow(xi, 5.55, tx + 0.5, int_y + 0.4)

# Top MLP and output
draw_box(4.5, 3.15, 3.2, 0.75, GREEN, 'top MLP')
draw_box(4.5, 2.0, 3.2, 0.75, GREEN, 'sigmoid  -> P(click)')
draw_arrow(4.5, int_y - 0.43, 4.5, 3.53)
draw_arrow(4.5, 2.78, 4.5, 2.38)

# Side annotation
ax.text(9.0, int_y,
        'key placement:\ninteraction is after\nembeddings, before\ntop MLP',
        fontsize=8, color='#b91c1c', va='center', style='italic', ha='center')

fig.savefig(BASE + 'fig-feature-interaction.png', bbox_inches='tight')
plt.close(fig)
print("fig-feature-interaction.png done")
print("all 4 figures done")
