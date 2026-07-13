import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = "/private/tmp/claude-501/-Users-xingao-Projects-Neurarch/65adcdb4-d882-4642-bf5c-d42bf1be1f61/scratchpad/awesome-ml-system-design/book/fraud-detection/assets/"

plt.rcParams.update({
    'figure.dpi': 130,
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.autolayout': True,
})

BLUE   = '#2563eb'
ORANGE = '#ea7317'
GREEN  = '#16a34a'
RED    = '#dc2626'
GRAY   = '#64748b'
PURPLE = '#7c3aed'

# -------------------------------------------------------------------
# Figure 1: class imbalance  (two panels)
# Left:  bar chart, fraud vs legit proportions
# Right: grouped bar, accuracy vs recall for two "models"
# -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

# --- Left panel: base rate ---
ax = axes[0]
labels = ['Legitimate\n(99.8%)', 'Fraud\n(0.2%)']
values = [99.8, 0.2]
colors = [BLUE, RED]
bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.5)
ax.set_yscale('log')
ax.set_ylabel('Share of transactions (%, log scale)')
ax.set_title('Base rate: 0.2% fraud')
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.6,
            f'{val}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# --- Right panel: accuracy vs recall ---
ax = axes[1]
categories = ['Accuracy', 'Recall']
never_fraud = [99.8, 0.0]
real_model  = [97.5, 80.0]
x = np.arange(len(categories))
w = 0.35
b1 = ax.bar(x - w/2, never_fraud, w, label='"Never-fraud" baseline', color=GRAY, alpha=0.85)
b2 = ax.bar(x + w/2, real_model,  w, label='Real model',             color=GREEN, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylabel('Value (%)')
ax.set_ylim(0, 115)
ax.set_title('Accuracy fails; recall reveals the truth')
ax.legend(fontsize=9, frameon=False)
for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
            f'{h:.0f}%', ha='center', va='bottom', fontsize=9)

fig.suptitle('Why accuracy fails at 0.2% base rate', fontsize=13, fontweight='bold', y=1.01)
fig.savefig(OUT + 'fig-class-imbalance.png', bbox_inches='tight')
plt.close(fig)
print('fig-class-imbalance.png written')

# -------------------------------------------------------------------
# Figure 2: PR vs ROC  (two panels)
# We synthesize score distributions for two classifiers without sklearn.
# -------------------------------------------------------------------

def make_pr_roc(scores_pos, scores_neg):
    """Return (fpr, tpr, rec, prec) curves by thresholding a score array."""
    n_pos = len(scores_pos)
    n_neg = len(scores_neg)
    all_scores = np.concatenate([scores_pos, scores_neg])
    all_labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    # sort descending by score
    order = np.argsort(-all_scores)
    sorted_labels = all_labels[order]

    tp = np.cumsum(sorted_labels)
    fp = np.cumsum(1 - sorted_labels)
    fn = n_pos - tp

    # avoid divide-by-zero
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (n_pos + 1e-9)
    fpr  = fp / (n_neg + 1e-9)
    tpr  = tp / (n_pos + 1e-9)
    # prepend (0,0) for ROC / (0,1) start for PR
    fpr  = np.concatenate([[0.0], fpr])
    tpr  = np.concatenate([[0.0], tpr])
    prec = np.concatenate([[1.0], prec])
    rec  = np.concatenate([[0.0], rec])
    return fpr, tpr, rec, prec

rng = np.random.default_rng(42)
N_pos = 2000
N_neg = 998000  # ~0.2% positive rate

# Classifier A: decent model
scores_pos_A = rng.beta(5, 2, N_pos)        # positives score high
scores_neg_A = rng.beta(2, 6, N_neg)        # negatives score low

# Classifier B: always-legit baseline (near-random, very slightly above chance)
scores_pos_B = rng.uniform(0.45, 0.55, N_pos)
scores_neg_B = rng.uniform(0.44, 0.54, N_neg)

fpr_A, tpr_A, rec_A, prec_A = make_pr_roc(scores_pos_A, scores_neg_A)
fpr_B, tpr_B, rec_B, prec_B = make_pr_roc(scores_pos_B, scores_neg_B)

# compute AUCs via trapezoid
roc_auc_A = float(np.trapezoid(tpr_A, fpr_A))
roc_auc_B = float(np.trapezoid(tpr_B, fpr_B))
pr_auc_A  = float(np.trapezoid(prec_A, rec_A))
pr_auc_B  = float(np.trapezoid(prec_B, rec_B))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# --- ROC ---
ax = axes[0]
ax.plot(fpr_A, tpr_A, color=BLUE,   lw=2, label=f'Model A (ROC-AUC={roc_auc_A:.2f})')
ax.plot(fpr_B, tpr_B, color=ORANGE, lw=2, label=f'Always-legit B (ROC-AUC={roc_auc_B:.2f})', linestyle='--')
ax.plot([0, 1], [0, 1], color=GRAY, lw=1, linestyle=':')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC curves')
ax.legend(fontsize=9, frameon=False)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)

# --- PR ---
ax = axes[1]
ax.plot(rec_A, prec_A, color=BLUE,   lw=2, label=f'Model A (PR-AUC={pr_auc_A:.3f})')
ax.plot(rec_B, prec_B, color=ORANGE, lw=2, label=f'Always-legit B (PR-AUC={pr_auc_B:.4f})', linestyle='--')
base_rate_line = N_pos / (N_pos + N_neg)
ax.axhline(base_rate_line, color=GRAY, lw=1, linestyle=':', label=f'base rate ({base_rate_line*100:.1f}%)')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall curves\n(PR-AUC reveals what ROC hides)')
ax.legend(fontsize=9, frameon=False)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)

fig.suptitle('ROC vs PR: ROC flatters the always-legit baseline, PR does not', fontsize=12, fontweight='bold', y=1.01)
fig.savefig(OUT + 'fig-pr-vs-roc.png', bbox_inches='tight')
plt.close(fig)
print('fig-pr-vs-roc.png written')

# -------------------------------------------------------------------
# Figure 3: threshold vs cost
# -------------------------------------------------------------------
tau = np.linspace(0.001, 0.999, 500)

# Synthetic: simulate FP(tau) and FN(tau) with logistic-like curves
# FP decreases as tau rises, FN increases as tau rises
# FP(tau) ~ sigmoid-like (fraction of negatives above tau)
# FN(tau) ~ 1 - sigmoid-like (fraction of positives below tau)

# For a "good" model, positives concentrate near 1, negatives near 0
def fp_curve(tau, mu_neg=0.25, sigma_neg=0.15):
    """Fraction of negatives scoring above tau (Gaussian CDF complement)."""
    return 0.5 * (1 - np.tanh((tau - mu_neg) / (sigma_neg * np.sqrt(2))))

def fn_curve(tau, mu_pos=0.75, sigma_pos=0.15):
    """Fraction of positives scoring below tau."""
    return 0.5 * (1 + np.tanh((tau - mu_pos) / (sigma_pos * np.sqrt(2))))

FP = fp_curve(tau)
FN = fn_curve(tau)

# Symmetric costs
c_FP_sym, c_FN_sym = 1.0, 1.0
cost_sym = c_FP_sym * FP + c_FN_sym * FN
cost_sym_norm = cost_sym / cost_sym.max()

# Asymmetric costs: missing fraud is 20x more costly
c_FP_asym, c_FN_asym = 1.0, 20.0
cost_asym = c_FP_asym * FP + c_FN_asym * FN
cost_asym_norm = cost_asym / cost_asym.max()

opt_sym  = tau[np.argmin(cost_sym_norm)]
opt_asym = tau[np.argmin(cost_asym_norm)]

# Theoretical optimum: tau* = c_FP / (c_FP + c_FN)
tau_star_sym  = c_FP_sym  / (c_FP_sym  + c_FN_sym)
tau_star_asym = c_FP_asym / (c_FP_asym + c_FN_asym)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(tau, cost_sym_norm,  color=BLUE,   lw=2, label=f'Symmetric costs (c_FP=c_FN=1)')
ax.plot(tau, cost_asym_norm, color=RED,    lw=2, label=f'Asymmetric costs (c_FP=1, c_FN=20)')
ax.axvline(opt_sym,  color=BLUE, lw=1.5, linestyle='--', alpha=0.8)
ax.axvline(opt_asym, color=RED,  lw=1.5, linestyle='--', alpha=0.8)
ax.text(opt_sym  + 0.02, 0.55, f'opt={opt_sym:.2f}',  color=BLUE, fontsize=9)
ax.text(opt_asym + 0.02, 0.75, f'opt={opt_asym:.2f}', color=RED,  fontsize=9)
ax.annotate('Asymmetric shifts\noptimum left (catch more fraud)',
            xy=(opt_asym, cost_asym_norm[np.argmin(cost_asym_norm)]),
            xytext=(opt_asym + 0.12, 0.35),
            fontsize=9, color=RED,
            arrowprops=dict(arrowstyle='->', color=RED))
ax.set_xlabel('Classification threshold')
ax.set_ylabel('Expected cost (normalized)')
ax.set_title('Asymmetric costs shift the optimal threshold left')
ax.legend(fontsize=9, frameon=False)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
fig.savefig(OUT + 'fig-threshold-vs-cost.png', bbox_inches='tight')
plt.close(fig)
print('fig-threshold-vs-cost.png written')

# -------------------------------------------------------------------
# Figure 4: cost matrix heatmap (2x2 annotated imshow)
# Rows = predicted (Fraud / Legit), Cols = Actual (Fraud / Legit)
# -------------------------------------------------------------------

# Cost values (qualitative, illustrative)
# TP=1 (prevented fraud, positive value shown as negative cost)
# FN=4 (chargeback + fees, large cost)
# FP=2 (lost sale + support + churn, moderate cost)
# TN=0 (correct allow, zero cost)
cost_matrix = np.array([
    [1.0,  4.0],  # Predicted Fraud: TP (actual fraud), FN (actual legit -- actually FP here)
    [2.0,  0.0],  # Predicted Legit: FP (actual fraud), TN (actual legit)
])

# Rows: Predicted [Fraud, Legit], Cols: Actual [Fraud, Legit]
# cell[0,0] = TP = prevented fraud (good, low cost)
# cell[0,1] = FP = blocked legit user (lost sale + support + churn)
# cell[1,0] = FN = missed fraud (chargeback + fees + overhead)
# cell[1,1] = TN = correct allow (zero cost)
cost_matrix = np.array([
    [0.5,  2.5],  # Predicted Fraud: [TP, FP]
    [4.0,  0.0],  # Predicted Legit: [FN, TN]
])

cell_names = [
    ['TP\n(prevented fraud)', 'FP\n(lost sale + support\n+ churn risk)'],
    ['FN\n(chargeback + fees\n+ overhead)',    'TN\n(correct allow)'],
]

row_labels = ['Predicted: Fraud', 'Predicted: Legit']
col_labels = ['Actual: Fraud', 'Actual: Legit']

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(cost_matrix, cmap='RdYlGn_r', vmin=0, vmax=4.5, aspect='auto')
fig.colorbar(im, ax=ax, fraction=0.046, label='Relative cost (0 = free, high = expensive)')

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(col_labels, fontsize=11)
ax.set_yticklabels(row_labels, fontsize=11)
ax.set_title('Cost matrix: the decision framing behind fraud scoring', fontsize=12, fontweight='bold')

for i in range(2):
    for j in range(2):
        ax.text(j, i, cell_names[i][j], ha='center', va='center',
                fontsize=10, fontweight='bold',
                color='white' if cost_matrix[i, j] > 2.5 else 'black')

fig.savefig(OUT + 'fig-cost-matrix.png', bbox_inches='tight')
plt.close(fig)
print('fig-cost-matrix.png written')

print('\nAll 4 figures written to:', OUT)
