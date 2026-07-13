import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = "/private/tmp/claude-501/-Users-xingao-Projects-Neurarch/65adcdb4-d882-4642-bf5c-d42bf1be1f61/scratchpad/awesome-ml-system-design/book/realtime-serving/assets/"

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

# ------------------------------------------------------------------ #
# 1. Latency vs batch size: CPU (linear) vs GPU (sub-linear)
# ------------------------------------------------------------------ #
fig, ax = plt.subplots(figsize=(7, 4))
batch = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])

# CPU: roughly linear in batch size (c0 + c1 * B)
cpu_lat = 2.0 + 0.35 * batch
# GPU: sub-linear (g0 + g1 * B^alpha, alpha < 1)
gpu_lat = 3.5 + 0.05 * (batch ** 0.60)

ax.plot(batch, cpu_lat, color=ORANGE, lw=2, marker='o', markersize=5, label='CPU serving')
ax.plot(batch, gpu_lat, color=BLUE,   lw=2, marker='s', markersize=5, label='GPU serving')

# Annotate the sub-linear win
ax.annotate(
    'GPU latency grows\nsub-linearly with batch',
    xy=(64, gpu_lat[6]),
    xytext=(80, gpu_lat[6] + 10),
    fontsize=9, color=BLUE,
    arrowprops=dict(arrowstyle='->', color=BLUE),
)
ax.annotate(
    'CPU latency grows\nlinearly with batch',
    xy=(64, cpu_lat[6]),
    xytext=(70, cpu_lat[6] + 8),
    fontsize=9, color=ORANGE,
    arrowprops=dict(arrowstyle='->', color=ORANGE),
)

ax.set_xscale('log', base=2)
ax.set_xlabel('batch size (log2 scale)')
ax.set_ylabel('inference latency (ms)')
ax.set_title('CPU vs GPU latency scaling with batch size\n(GPU sub-linear: bigger batches fill the accelerator)')
ax.legend(frameon=False)
fig.savefig(OUT + 'fig-latency-vs-batch.png')
plt.close(fig)

# ------------------------------------------------------------------ #
# 2. p50 / p95 / p99 latency: why the average hides the tail
# ------------------------------------------------------------------ #
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Simulate a latency distribution: most requests are fast, a fat tail
rng = np.random.default_rng(42)
n = 50_000
fast = rng.exponential(scale=8, size=int(n * 0.93))
slow = rng.exponential(scale=80, size=int(n * 0.07))
latencies = np.concatenate([fast, slow])

p50  = np.percentile(latencies, 50)
p95  = np.percentile(latencies, 95)
p99  = np.percentile(latencies, 99)
mean = latencies.mean()

ax = axes[0]
ax.hist(latencies, bins=120, range=(0, 250), color=BLUE, alpha=0.65, density=True)
ax.axvline(mean, color=ORANGE, lw=2, linestyle='--', label=f'mean  {mean:.0f} ms')
ax.axvline(p50,  color=GREEN,  lw=2, linestyle='-',  label=f'p50   {p50:.0f} ms')
ax.axvline(p95,  color=RED,    lw=2, linestyle='-.',  label=f'p95   {p95:.0f} ms')
ax.axvline(p99,  color='black',lw=2, linestyle=':',  label=f'p99   {p99:.0f} ms')
ax.set_xlabel('request latency (ms)')
ax.set_ylabel('density')
ax.set_title('Latency distribution\n(fat tail hidden by the mean)')
ax.legend(fontsize=9, frameon=False)

# Bar chart: how many requests breach a 50 ms SLA at each percentile
ax2 = axes[1]
thresholds = [10, 20, 50, 100]
pcts = [np.mean(latencies > t) * 100 for t in thresholds]
colors = [GREEN if p < 1 else ORANGE if p < 5 else RED for p in pcts]
bars = ax2.bar([str(t) + ' ms' for t in thresholds], pcts, color=colors, alpha=0.8)
ax2.set_xlabel('SLA threshold')
ax2.set_ylabel('requests breaching SLA (%)')
ax2.set_title('Fraction of requests breaching each threshold\n(alert on p99, not the mean)')
for bar, pct in zip(bars, pcts):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
             f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

fig.savefig(OUT + 'fig-latency-percentiles.png')
plt.close(fig)

# ------------------------------------------------------------------ #
# 3. Canary rollout: traffic split over time
# ------------------------------------------------------------------ #
fig, ax = plt.subplots(figsize=(8, 4))

# Timeline in hours
t = np.array([0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16])
# New version traffic fraction
new_pct = np.array([0, 0, 5, 5, 25, 25, 50, 50, 100, 100, 100])
old_pct = 100 - new_pct

shadow_end = 2  # shadow phase ends at hour 2
canary_end = 4  # canary (5%) ends at hour 4

ax.fill_between(t, 0, old_pct, alpha=0.35, color=BLUE,   label='stable (v1)')
ax.fill_between(t, old_pct, 100, alpha=0.55, color=GREEN, label='candidate (v2)')
ax.plot(t, old_pct, color=BLUE,  lw=1.5)
ax.plot(t, new_pct, color=GREEN, lw=1.5)

# Annotations for phases
ax.axvspan(0, shadow_end, alpha=0.07, color=GRAY)
ax.text(1, 50, 'shadow\n(mirror only)', ha='center', va='center', fontsize=9, color=GRAY)

ax.axvline(shadow_end, color=GRAY, lw=1, linestyle='--')
ax.text(3, 85, 'canary 5%', ha='center', fontsize=9, color=GREEN)
ax.text(7, 75, 'ramp 50%',  ha='center', fontsize=9, color=GREEN)
ax.text(13, 55, 'full rollout', ha='center', fontsize=9, color=GREEN)

ax.set_xlabel('time (hours)')
ax.set_ylabel('traffic share (%)')
ax.set_xlim(0, 16)
ax.set_ylim(0, 105)
ax.set_title('Canary rollout: shadow then step-ramp to 100%\n(each step gated on health metrics)')
ax.legend(loc='center right', frameon=False)
fig.savefig(OUT + 'fig-canary-rollout.png')
plt.close(fig)

# ------------------------------------------------------------------ #
# 4. Cost curve: throughput per dollar on CPU vs GPU
# ------------------------------------------------------------------ #
fig, ax = plt.subplots(figsize=(7, 4))

qps = np.linspace(100, 20000, 300)

# CPU: cost scales linearly with QPS (one replica per N rps, cost per replica fixed)
cpu_cost = qps / 800  # ~$1 per 800 QPS (illustrative)

# GPU: fixed cost for the GPU instance, amortized over high throughput
# At low QPS GPU is expensive per-request; at high QPS it wins
gpu_cost = 4.0 + qps / 8000  # high base, very shallow slope

ax.plot(qps, cpu_cost, color=ORANGE, lw=2, label='CPU replicas (linear cost)')
ax.plot(qps, gpu_cost, color=BLUE,   lw=2, label='GPU serving (high base, flat marginal cost)')

# Crossover
crossover_qps = 800 * (4.0 / (1 - 800/8000))  # approximate
crossover_qps = 3700  # rounded illustrative value
ax.axvline(crossover_qps, color=GRAY, lw=1, linestyle='--')
ax.text(crossover_qps + 200, 5, f'crossover\n~{crossover_qps:,.0f} QPS',
        fontsize=9, color=GRAY, va='top')

ax.set_xlabel('throughput (QPS)')
ax.set_ylabel('cost per hour (USD, illustrative)')
ax.set_title('CPU vs GPU cost curve\n(GPU wins at high throughput; CPU wins for bursty low-QPS workloads)')
ax.legend(frameon=False)
fig.savefig(OUT + 'fig-cpu-vs-gpu-cost.png')
plt.close(fig)

print("wrote 4 figures:")
print("  fig-latency-vs-batch.png")
print("  fig-latency-percentiles.png")
print("  fig-canary-rollout.png")
print("  fig-cpu-vs-gpu-cost.png")
