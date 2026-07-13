import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = "/private/tmp/claude-501/-Users-xingao-Projects-Neurarch/65adcdb4-d882-4642-bf5c-d42bf1be1f61/scratchpad/awesome-ml-system-design/book/computer-vision/assets/"

plt.rcParams.update({
    'figure.dpi': 130, 'font.size': 11, 'axes.grid': True,
    'grid.alpha': 0.3, 'axes.spines.top': False,
    'axes.spines.right': False, 'figure.autolayout': True
})
BLUE, ORANGE, GREEN, RED, GRAY = '#2563eb', '#ea7317', '#16a34a', '#dc2626', '#64748b'


# 1) Task taxonomy vs label cost and serving cost
fig, ax = plt.subplots(figsize=(9, 4))
tasks = [
    'Classification\n(whole-image)',
    'Detection\n(bounding box)',
    'Segmentation\n(per-pixel)',
    'Embedding\n(no fixed classes)',
    'OCR\n(text + position)',
]
label_cost  = [1.0, 6.0, 25.0, 2.0, 8.0]
serving_cost = [1.0, 2.5,  4.0, 1.5, 3.0]
x = np.arange(len(tasks))
w = 0.38
ax.bar(x - w / 2, label_cost,   w, label='Labeling cost (relative)',      color=ORANGE, alpha=0.85)
ax.bar(x + w / 2, serving_cost, w, label='GPU serving cost (relative)',    color=BLUE,   alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=9)
ax.set_ylabel('Relative cost  (classification = 1)')
ax.set_title('Task taxonomy: labeling cost and GPU serving cost by task type')
ax.legend(frameon=False)
fig.savefig(OUT + 'fig-task-cost.png')
plt.close(fig)


# 2) Transfer learning data-efficiency curves
fig, ax = plt.subplots(figsize=(7.5, 4.2))
n = np.logspace(2, 5, 300)
acc_scratch  = 0.95 * (1 - np.exp(-n / 80000))
acc_finetune = 0.85 + 0.10 * (1 - np.exp(-n / 5000))
acc_linear   = 0.70 + 0.18 * (1 - np.exp(-n / 15000))
ax.semilogx(n, acc_scratch,  color=RED,    lw=2, label='Train from scratch')
ax.semilogx(n, acc_finetune, color=GREEN,  lw=2, label='Fine-tune backbone')
ax.semilogx(n, acc_linear,   color=ORANGE, lw=2, label='Linear probe (frozen backbone)')
ax.axvline(1000, color=GRAY, ls='--', lw=1)
ax.text(1100, 0.35, 'typical labeling\nbudget at project start', color=GRAY, fontsize=9)
ax.set_xlabel('Labeled examples (log scale)')
ax.set_ylabel('Validation accuracy')
ax.set_ylim(0.28, 1.02)
ax.set_title('Transfer learning: fine-tuning beats training from scratch at low label counts')
ax.legend(frameon=False)
fig.savefig(OUT + 'fig-transfer-learning.png')
plt.close(fig)


# 3) Precision-recall curves with mAP illustration
fig, ax = plt.subplots(figsize=(7.5, 4.2))

def pr_curve(a, b, n=300):
    r = np.linspace(0, 1, n)
    p = a * np.exp(-b * r) + (1 - a) * (1 - r)
    return r, np.clip(p, 0, 1)

r1, p1 = pr_curve(0.95, 2.5)
r2, p2 = pr_curve(0.80, 3.5)
r3, p3 = pr_curve(0.65, 5.0)
ap1 = np.trapezoid(p1, r1)
ap2 = np.trapezoid(p2, r2)
ap3 = np.trapezoid(p3, r3)
mean_ap = (ap1 + ap2 + ap3) / 3

ax.fill_between(r1, p1, alpha=0.12, color=GREEN)
ax.fill_between(r2, p2, alpha=0.12, color=BLUE)
ax.fill_between(r3, p3, alpha=0.12, color=ORANGE)
ax.plot(r1, p1, color=GREEN,  lw=2, label=f'Class A   AP = {ap1:.2f}')
ax.plot(r2, p2, color=BLUE,   lw=2, label=f'Class B   AP = {ap2:.2f}')
ax.plot(r3, p3, color=ORANGE, lw=2, label=f'Class C   AP = {ap3:.2f}')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title(f'Per-class PR curves; shaded area = AP; mAP = {mean_ap:.2f}')
ax.legend(frameon=False)
fig.savefig(OUT + 'fig-pr-map.png')
plt.close(fig)


# 4) IoU illustration
fig, ax = plt.subplots(figsize=(5.5, 4.8))
# ground-truth box
gx1, gy1, gw, gh = 0.10, 0.18, 0.48, 0.55
# predicted box (offset)
px1, py1, pw, ph = 0.30, 0.30, 0.48, 0.55
gt   = plt.Rectangle((gx1, gy1), gw, gh, lw=2.5, edgecolor=GREEN, facecolor=GREEN, alpha=0.20, label='Ground truth')
pred = plt.Rectangle((px1, py1), pw, ph, lw=2.5, edgecolor=BLUE,  facecolor=BLUE,  alpha=0.20, label='Predicted')
ax.add_patch(gt)
ax.add_patch(pred)
# intersection
ix1 = max(gx1, px1); iy1 = max(gy1, py1)
ix2 = min(gx1+gw, px1+pw); iy2 = min(gy1+gh, py1+ph)
inter = plt.Rectangle((ix1, iy1), ix2-ix1, iy2-iy1, lw=0, facecolor=ORANGE, alpha=0.55, label='Intersection')
ax.add_patch(inter)
ia = (ix2-ix1) * (iy2-iy1)
ua = gw*gh + pw*ph - ia
iou_val = ia / ua
ax.text((ix1+ix2)/2, (iy1+iy2)/2, 'Intersection', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
ax.text(gx1+gw/2 - 0.06, gy1+gh/2 + 0.10, 'GT',   ha='center', fontsize=11, color=GREEN, fontweight='bold')
ax.text(px1+pw/2 + 0.06, py1+ph/2 - 0.10, 'Pred', ha='center', fontsize=11, color=BLUE,  fontweight='bold')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title(f'IoU = Intersection / Union = {iou_val:.2f}\nCOCO mAP uses thresholds 0.5 to 0.95', pad=8)
ax.legend(loc='upper right', frameon=True, fontsize=9)
fig.savefig(OUT + 'fig-iou.png')
plt.close(fig)

print("wrote 4 figures: fig-task-cost.png, fig-transfer-learning.png, fig-pr-map.png, fig-iou.png")
