import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUT = ("/private/tmp/claude-501/-Users-xingao-Projects-Neurarch/"
       "65adcdb4-d882-4642-bf5c-d42bf1be1f61/scratchpad/"
       "awesome-ml-system-design/book/nlp/assets/")

plt.rcParams.update({'figure.dpi': 130, 'font.size': 11, 'figure.autolayout': True})

BLUE   = '#2563eb'
GREEN  = '#16a34a'
RED    = '#dc2626'
PURPLE = '#7c3aed'
GRAY   = '#64748b'
L_GREEN = '#dcfce7'
L_RED   = '#fee2e2'
L_GRAY  = '#f1f5f9'

TP, FP, FN, TN = 40, 10, 20, 130
P  = TP / (TP + FP)          # 0.80
R  = TP / (TP + FN)          # 0.6667
F1 = 2 * P * R / (P + R)     # 0.7273

fig = plt.figure(figsize=(10.5, 5.0))
fig.patch.set_facecolor('white')

# ------------------------------------------------------------------ #
# Left panel: 2x2 confusion matrix with precision / recall brackets   #
# ------------------------------------------------------------------ #
ax = fig.add_subplot(121)
ax.set_xlim(-0.8, 7.0)
ax.set_ylim(-0.5, 7.0)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Confusion matrix (illustrative counts)', fontsize=11, pad=8)

# Column headers
ax.text(2.0, 6.5, 'Predicted\nPositive', ha='center', va='center',
        fontsize=9.5, color=BLUE, fontweight='bold')
ax.text(4.5, 6.5, 'Predicted\nNegative', ha='center', va='center',
        fontsize=9.5, color=GRAY)

# Row headers
ax.text(-0.2, 4.4, 'Actual\nPositive', ha='center', va='center',
        fontsize=9.5, color=GREEN, fontweight='bold')
ax.text(-0.2, 2.0, 'Actual\nNegative', ha='center', va='center',
        fontsize=9.5, color=GRAY)

def draw_cell(ax, x, y, w, h, label, count, face, label_color='#1e293b'):
    r = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.12',
        facecolor=face, edgecolor='#94a3b8', lw=1.6
    )
    ax.add_patch(r)
    ax.text(x + w / 2, y + h * 0.67, label,
            ha='center', va='center', fontsize=14,
            fontweight='bold', color=label_color)
    ax.text(x + w / 2, y + h * 0.28, f'= {count}',
            ha='center', va='center', fontsize=10, color='#475569')

CW, CH = 2.2, 1.8
draw_cell(ax, 0.9, 3.2, CW, CH, 'TP', TP, L_GREEN, GREEN)
draw_cell(ax, 3.3, 3.2, CW, CH, 'FN', FN, L_GRAY)
draw_cell(ax, 0.9, 1.2, CW, CH, 'FP', FP, L_RED, RED)
draw_cell(ax, 3.3, 1.2, CW, CH, 'TN', TN, L_GRAY)

# Precision bracket below the "Predicted Positive" column (x: 0.9 to 3.1)
ax.annotate(
    '', xy=(0.9, 0.5), xytext=(3.1, 0.5),
    arrowprops=dict(arrowstyle='<->', color=BLUE, lw=2.0)
)
ax.text(2.0, 0.10,
        f'Precision = TP / (TP + FP) = {TP} / ({TP}+{FP}) = {P:.2f}',
        ha='center', va='center', fontsize=8.5, color=BLUE, fontweight='bold')

# Recall bracket to the right of the "Actual Positive" row (y: 3.2 to 5.0)
ax.annotate(
    '', xy=(6.2, 3.2), xytext=(6.2, 5.0),
    arrowprops=dict(arrowstyle='<->', color=GREEN, lw=2.0)
)
ax.text(6.25, 4.1,
        f'Recall\n= TP / (TP + FN)\n= {TP} / ({TP}+{FN})\n= {R:.2f}',
        ha='left', va='center', fontsize=8.3, color=GREEN, fontweight='bold')

# ------------------------------------------------- #
# Right panel: formula derivation                    #
# ------------------------------------------------- #
ax2 = fig.add_subplot(122)
ax2.axis('off')
ax2.set_title('From confusion-matrix counts to F1', fontsize=11, pad=8)

rows = [
    (0.50, 0.82,
     r'$P = \dfrac{TP}{TP + FP} = \dfrac{40}{50} = 0.80$',
     BLUE, 12),
    (0.50, 0.57,
     r'$R = \dfrac{TP}{TP + FN} = \dfrac{40}{60} \approx 0.67$',
     GREEN, 12),
    (0.50, 0.32,
     r'$F_1 = \dfrac{2PR}{P + R} = \dfrac{2(0.80)(0.67)}{0.80 + 0.67} \approx 0.73$',
     RED, 12),
    (0.50, 0.10,
     r'$F_1^{\text{macro}} = \dfrac{1}{C}\sum_{c=1}^{C} F_1^{(c)}$  (unweighted mean over classes)',
     PURPLE, 10),
]
for x, y, txt, color, fs in rows:
    ax2.text(x, y, txt, transform=ax2.transAxes,
             ha='center', va='center', fontsize=fs, color=color, linespacing=1.5)

fig.tight_layout()
fig.savefig(OUT + 'fig-confusion-matrix-prf1.png', dpi=130, bbox_inches='tight')
plt.close(fig)
print('wrote fig-confusion-matrix-prf1.png')
