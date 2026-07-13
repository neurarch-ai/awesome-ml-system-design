import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = "/private/tmp/claude-501/-Users-xingao-Projects-Neurarch/65adcdb4-d882-4642-bf5c-d42bf1be1f61/scratchpad/awesome-ml-system-design/book/nlp/assets/"

plt.rcParams.update({'figure.dpi': 130, 'font.size': 11, 'axes.grid': True,
                     'grid.alpha': 0.3, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.autolayout': True})

BLUE, ORANGE, GREEN, RED, GRAY = '#2563eb', '#ea7317', '#16a34a', '#dc2626', '#64748b'

# ── 1) Fine-tuned encoder vs LLM: latency and cost ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))

models = ['Distilled\nencoder\n(BERT-tiny)', 'BERT-base\n(fine-tuned)', 'Small LLM\n(7B)', 'Large LLM\n(GPT-4 class)']
latencies = [3, 10, 250, 900]        # ms per call, illustrative
costs     = [0.003, 0.012, 0.8, 6.0] # $/1M inferences, illustrative
colors    = [GREEN, BLUE, ORANGE, RED]

ax = axes[0]
ax.bar(models, latencies, color=colors, alpha=0.85)
ax.set_yscale('log')
ax.set_ylim(0.8, 3000)
ax.set_ylabel('Latency per call (ms, log)')
ax.set_title('Inline latency\n(50 ms budget = dashed)')
ax.axhline(50, color=GRAY, ls='--', lw=1.5)
ax.text(2.55, 65, '50 ms', color=GRAY, fontsize=9)

ax = axes[1]
ax.bar(models, costs, color=colors, alpha=0.85)
ax.set_yscale('log')
ax.set_ylim(0.001, 30)
ax.set_ylabel('Cost per 1M inferences ($, log)')
ax.set_title('Cost at firehose scale\n(illustrative, order-of-magnitude)')

for bar, val in zip(axes[1].patches, costs):
    axes[1].text(bar.get_x() + bar.get_width() / 2, val * 1.4, f'${val}',
                 ha='center', va='bottom', fontsize=8, color=GRAY)

fig.savefig(OUT + 'fig-encoder-vs-llm.png')
plt.close(fig)

# ── 2) Per-class F1 under class imbalance ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))

classes = ['Not spam\n(99.5 %)', 'Spam\n(0.5 %)']
x = np.arange(len(classes))
w = 0.36

acc_scores = [99.7, 99.7]      # accuracy: looks great on both
f1_naive   = [99.85, 14.0]     # F1 before fixing: minority class collapses
f1_fixed   = [99.4,  72.0]     # F1 after resampling + cost-aware threshold

ax = axes[0]
ax.bar(x - w/2, acc_scores, w, label='Accuracy', color=BLUE, alpha=0.85)
ax.bar(x + w/2, f1_naive,   w, label='F1',        color=RED,  alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(classes)
ax.set_ylim(0, 115); ax.set_ylabel('Score (%)')
ax.set_title('Accuracy hides minority-class collapse')
ax.legend(fontsize=9, frameon=False)
ax.text(0.82, 18, '!', fontsize=24, color=RED, ha='center', fontweight='bold')

ax = axes[1]
ax.bar(x - w/2, f1_naive, w, label='F1 before fix', color=RED,   alpha=0.80)
ax.bar(x + w/2, f1_fixed, w, label='F1 after fix',  color=GREEN, alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(classes)
ax.set_ylim(0, 115); ax.set_ylabel('F1 (%)')
ax.set_title('Resampling + per-class cost-aware threshold')
ax.legend(fontsize=9, frameon=False)

fig.savefig(OUT + 'fig-class-imbalance-f1.png')
plt.close(fig)

# ── 3) Tokenizer fertility by language ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(8.4, 4.2))

langs   = ['English', 'German', 'Arabic', 'Turkish', 'Finnish', 'Japanese\n(char-based)']
fert    = [1.25,       1.60,     2.35,     2.80,      2.55,      3.20]
colors_l = [BLUE, BLUE, ORANGE, ORANGE, ORANGE, RED]

bars = ax.bar(langs, fert, color=colors_l, alpha=0.85)
ax.set_ylabel('Avg tokens per word (WordPiece / BPE)')
ax.set_title('Tokenizer fertility by language\n(more tokens = more latency, more cost per inference)')
ax.axhline(1.0, color=GRAY, ls='--', lw=1.0, alpha=0.5)
ax.set_ylim(0, 4.2)

for bar, val in zip(bars, fert):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.06, f'{val:.2f}',
            ha='center', va='bottom', fontsize=9)

ax.annotate('morphologically rich and\nnon-Latin scripts fragment more',
            xy=(3, 2.80), xytext=(3.6, 3.7), fontsize=9, color=GRAY,
            arrowprops=dict(arrowstyle='->', color=GRAY))

fig.savefig(OUT + 'fig-tokenizer-fertility.png')
plt.close(fig)

# ── 4) Calibration curve (reliability diagram) ──────────────────────────────
fig, ax = plt.subplots(figsize=(5.6, 5.2))

p_pred = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
frac_overconf  = np.array([0.10, 0.22, 0.32, 0.41, 0.50, 0.58, 0.67, 0.76, 0.85, 0.91])
frac_calibrated = np.array([0.05, 0.14, 0.24, 0.34, 0.45, 0.56, 0.66, 0.76, 0.86, 0.95])

ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfect calibration', zorder=1)
ax.plot(p_pred, frac_overconf,   'o-', color=RED,   lw=2, ms=7, label='Before calibration',          zorder=3)
ax.plot(p_pred, frac_calibrated, 's-', color=GREEN, lw=2, ms=7, label='After temperature scaling',    zorder=3)

ax.fill_between(p_pred, p_pred, frac_overconf, alpha=0.10, color=RED)

ax.set_xlabel('Mean predicted probability (confidence)')
ax.set_ylabel('Fraction of positives (actual rate)')
ax.set_title('Calibration curve (reliability diagram)\nset thresholds only on calibrated scores')
ax.legend(fontsize=9, frameon=False)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

fig.savefig(OUT + 'fig-calibration-curve.png')
plt.close(fig)

print("wrote 4 figures")
