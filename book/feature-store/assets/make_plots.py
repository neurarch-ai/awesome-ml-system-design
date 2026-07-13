import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = "/private/tmp/claude-501/-Users-xingao-Projects-Neurarch/65adcdb4-d882-4642-bf5c-d42bf1be1f61/scratchpad/awesome-ml-system-design/book/feature-store/assets/"

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


# ---------------------------------------------------------------------------
# Figure 1: training vs serving distribution divergence
# Two overlapping density curves, with a PSI annotation.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))

x = np.linspace(-3, 8, 600)
def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

train_pdf = gaussian(x, mu=1.5, sigma=1.1)
serve_pdf = gaussian(x, mu=3.0, sigma=1.6)

ax.fill_between(x, train_pdf, alpha=0.30, color=BLUE,   label='Training distribution')
ax.fill_between(x, serve_pdf, alpha=0.30, color=ORANGE, label='Serving distribution')
ax.plot(x, train_pdf, color=BLUE,   linewidth=2)
ax.plot(x, serve_pdf, color=ORANGE, linewidth=2)

ax.set_xlabel('Feature value')
ax.set_ylabel('Density')
ax.set_title('Training vs. serving feature distribution (skew)')
ax.legend(loc='upper right')

ax.annotate(
    'PSI = 0.24\n(skew detected)',
    xy=(5.5, 0.06), fontsize=10, color=RED,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=RED, alpha=0.85),
)
ax.set_xlim(-3, 8)
plt.savefig(OUT + 'fig-skew-distributions.png')
plt.close()
print("fig-skew-distributions.png saved")


# ---------------------------------------------------------------------------
# Figure 2: point-in-time join timeline
# Feature write events (blue dots) and label events (red crosses) on a
# shared time axis; green arrows show which feature version each label picks.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3.5)
ax.axis('off')
ax.set_title('Point-in-time join: each label picks the feature value valid just before it', pad=10)

# Main time axis
ax.annotate('', xy=(9.6, 1.6), xytext=(0.2, 1.6),
            arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.8))
ax.text(9.8, 1.6, 'time', ha='left', va='center', color=GRAY, fontsize=10)

# Feature write events
feat_times = [1.0, 3.2, 6.0, 8.6]
feat_labels = ['v1\n(count=4)', 'v2\n(count=11)', 'v3\n(count=19)', 'v4\n(count=25)']
for t, v in zip(feat_times, feat_labels):
    ax.plot(t, 1.6, 'o', color=BLUE, markersize=12, zorder=4)
    ax.text(t, 2.15, v, ha='center', va='bottom', fontsize=9, color=BLUE)

# Label / event times
evt_times  = [2.1, 4.5, 7.2]
evt_labels = ['label A\n(T=2.1)', 'label B\n(T=4.5)', 'label C\n(T=7.2)']
for t, lab in zip(evt_times, evt_labels):
    ax.plot(t, 1.6, marker='x', color=RED, markersize=13, markeredgewidth=2.5, zorder=5)
    ax.text(t, 0.95, lab, ha='center', va='top', fontsize=9, color=RED)

# Green arrows: each label picks the most-recent feature version before it
correct_feat = [feat_times[0], feat_times[1], feat_times[2]]  # v1, v2, v3
correct_name = ['v1', 'v2', 'v3']
for et, ft, cn in zip(evt_times, correct_feat, correct_name):
    ax.annotate(
        '', xy=(ft, 1.6), xytext=(et, 1.6),
        arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.6,
                        connectionstyle='arc3,rad=-0.35'),
    )
    mid = (et + ft) / 2
    ax.text(mid, 0.4, cn, ha='center', va='center', fontsize=9, color=GREEN,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=GREEN, alpha=0.75))

plt.savefig(OUT + 'fig-point-in-time-join.png')
plt.close()
print("fig-point-in-time-join.png saved")


# ---------------------------------------------------------------------------
# Figure 3: freshness tiers -- staleness tolerance vs relative infra cost
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))

tiers = [
    ('Daily batch\naggregates',   24*60,  1,  BLUE),
    ('Hourly batch\nrefresh',      60,    3,  BLUE),
    ('Streaming\n(seconds lag)',    0.5,  7,  ORANGE),
    ('Request-time\ncomputation',   0.01, 10, RED),
]
for label, staleness, cost, col in tiers:
    ax.scatter(staleness, cost, s=200, color=col, zorder=4)
    ax.text(staleness * 1.3, cost + 0.25, label, fontsize=9, color=col, va='bottom')

ax.set_xscale('log')
ax.set_xlabel('Max staleness (minutes, log scale)')
ax.set_ylabel('Relative infrastructure cost')
ax.set_title('Feature freshness tiers: staleness vs. cost')
ax.set_ylim(0, 12)
ax.set_xlim(0.005, 5000)

# Annotate budget zone
ax.axhspan(0, 4.5, alpha=0.06, color=BLUE,   label='Batch-viable zone')
ax.axhspan(4.5, 12, alpha=0.06, color=ORANGE, label='Streaming zone')
ax.legend(loc='upper right', fontsize=9)

plt.savefig(OUT + 'fig-freshness-tiers.png')
plt.close()
print("fig-freshness-tiers.png saved")


# ---------------------------------------------------------------------------
# Figure 4: online store read latency (p50 and p99) -- horizontal bar chart
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.5))

stores = [
    'Redis\n(in-memory)',
    'PostgreSQL\n(with pgcache)',
    'Cassandra\n(Uber, P95 cited)',
    'DynamoDB\n(single-region SSD)',
    'Bigtable\n(GCP)',
]
p50_ms = [0.5,  2.0,  5.0,  3.5,  5.5]
p99_ms = [1.5,  8.0, 12.0, 16.0, 22.0]
y = np.arange(len(stores))

ax.barh(y + 0.18, p99_ms, 0.32, label='p99', color=ORANGE, alpha=0.85)
ax.barh(y - 0.18, p50_ms, 0.32, label='p50', color=BLUE,   alpha=0.85)

ax.axvline(10, color=RED, linestyle='--', linewidth=1.5, label='10 ms serving budget')

ax.set_yticks(y)
ax.set_yticklabels(stores, fontsize=10)
ax.set_xlabel('Latency (ms)')
ax.set_title('Online store read latency (illustrative)')
ax.legend(fontsize=9)

plt.savefig(OUT + 'fig-online-store-latency.png')
plt.close()
print("fig-online-store-latency.png saved")

print("All figures saved.")
