import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = "/private/tmp/claude-501/-Users-xingao-Projects-Neurarch/65adcdb4-d882-4642-bf5c-d42bf1be1f61/scratchpad/awesome-ml-system-design/book/speech/assets/"

plt.rcParams.update({
    'figure.dpi': 130,
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.autolayout': True,
})
BLUE, ORANGE, GREEN, RED, GRAY = '#2563eb', '#ea7317', '#16a34a', '#dc2626', '#64748b'

# ---------------------------------------------------------------
# 1) Streaming vs batch latency comparison
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))

utterance_len = np.array([1, 2, 3, 5, 8, 12, 20, 30])  # seconds

# Streaming: first partial very fast, roughly constant lag regardless of length
streaming_first_partial = np.full_like(utterance_len, 0.28, dtype=float)
# Streaming total = utterance + small fixed overhead
streaming_total = utterance_len + 0.5

# Batch: must wait for whole utterance then process
batch_latency = utterance_len * 1.05 + 1.2  # processing overhead

ax.plot(utterance_len, streaming_first_partial, 'o-', color=GREEN, lw=2,
        label='Streaming: first partial')
ax.plot(utterance_len, streaming_total, 's--', color=BLUE, lw=2,
        label='Streaming: final transcript')
ax.plot(utterance_len, batch_latency, '^-', color=RED, lw=2,
        label='Batch: transcript ready')

ax.axhline(0.3, color=GRAY, ls=':', lw=1)
ax.text(1.3, 0.38, '300 ms first-partial target', color=GRAY, fontsize=9)

ax.set_xlabel('Utterance length (s)')
ax.set_ylabel('Latency to result (s)')
ax.set_title('Streaming vs batch ASR latency\n(streaming gives fast partials; batch waits for full audio)')
ax.legend(fontsize=9, frameon=False)
fig.savefig(OUT + 'fig-streaming-vs-batch-latency.png')
plt.close(fig)

# ---------------------------------------------------------------
# 2) WER vs model size (illustrative, inspired by Whisper scaling)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4))

# Model sizes in millions of parameters
sizes = np.array([39, 74, 244, 769, 1550])
labels = ['tiny\n39M', 'base\n74M', 'small\n244M', 'medium\n769M', 'large\n1550M']

# WER on clean vs noisy (illustrative)
wer_clean = np.array([14.8, 11.2, 6.8, 4.5, 3.0])
wer_noisy = np.array([28.0, 23.5, 17.2, 12.0, 8.5])

ax.plot(sizes, wer_clean, 'o-', color=BLUE, lw=2, label='WER clean test set')
ax.plot(sizes, wer_noisy, 's--', color=RED, lw=2, label='WER noisy / far-field test set')
ax.fill_between(sizes, wer_clean, wer_noisy, alpha=0.10, color=ORANGE,
                label='robustness gap')

ax.set_xscale('log')
ax.set_xlabel('Model parameters (log scale)')
ax.set_ylabel('WER (%)')
ax.set_title('WER vs model size: clean vs noisy\n(robustness gap narrows at scale, never disappears)')

for i, (s, lbl) in enumerate(zip(sizes, labels)):
    ax.annotate(lbl, (s, wer_clean[i]), xytext=(0, 10),
                textcoords='offset points', ha='center', fontsize=8, color=BLUE)

ax.legend(fontsize=9, frameon=False)
fig.savefig(OUT + 'fig-wer-vs-model-size.png')
plt.close(fig)

# ---------------------------------------------------------------
# 3) Schematic log-mel spectrogram + waveform
# ---------------------------------------------------------------
fig, (ax_wave, ax_spec) = plt.subplots(2, 1, figsize=(8, 4.5), sharex=False)

rng = np.random.default_rng(42)
sr = 16000
t_total = 1.0
t = np.linspace(0, t_total, int(sr * t_total))

# Synthetic waveform: voiced segment + silence + voiced
wave = np.zeros_like(t)
voiced1 = (t < 0.35)
voiced2 = (t > 0.55)
wave[voiced1] = (0.5 * np.sin(2 * np.pi * 120 * t[voiced1]) +
                 0.3 * np.sin(2 * np.pi * 360 * t[voiced1]) +
                 0.15 * rng.standard_normal(voiced1.sum()))
wave[voiced2] = (0.4 * np.sin(2 * np.pi * 150 * t[voiced2]) +
                 0.25 * np.sin(2 * np.pi * 450 * t[voiced2]) +
                 0.1 * rng.standard_normal(voiced2.sum()))

ax_wave.plot(t, wave, color=BLUE, lw=0.6)
ax_wave.set_ylabel('Amplitude')
ax_wave.set_title('Raw waveform (16 kHz, ~1 s utterance)')
ax_wave.set_xlim(0, t_total)
ax_wave.axvspan(0.35, 0.55, color=GRAY, alpha=0.2, label='silence / pause')
ax_wave.legend(fontsize=8, frameon=False)
ax_wave.grid(False)

# Spectrogram
hop = 160  # 10 ms
win = 400  # 25 ms
n_fft = 512
n_mels = 80
frames = []
for i in range(0, len(wave) - win, hop):
    frame = wave[i:i + win] * np.hanning(win)
    spec = np.abs(np.fft.rfft(frame, n=n_fft)) ** 2
    frames.append(spec)
S = np.array(frames).T  # (freq_bins, time_frames)

# Simple mel filterbank (triangular, illustrative)
mel_freqs = np.linspace(0, 2595 * np.log10(1 + (sr / 2) / 700), n_mels + 2)
hz_freqs = 700 * (10 ** (mel_freqs / 2595) - 1)
fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
mel_fb = np.zeros((n_mels, n_fft // 2 + 1))
for m in range(n_mels):
    lo, ctr, hi = hz_freqs[m], hz_freqs[m + 1], hz_freqs[m + 2]
    for f, freq in enumerate(fft_freqs):
        if lo <= freq <= ctr:
            mel_fb[m, f] = (freq - lo) / (ctr - lo + 1e-9)
        elif ctr < freq <= hi:
            mel_fb[m, f] = (hi - freq) / (hi - ctr + 1e-9)

mel_S = np.log(mel_fb @ S + 1e-9)

img = ax_spec.imshow(mel_S, aspect='auto', origin='lower', cmap='viridis',
                     extent=[0, t_total, 0, n_mels])
ax_spec.set_ylabel('Mel bin')
ax_spec.set_xlabel('Time (s)')
ax_spec.set_title('Log-mel spectrogram (80 mel bins)')
fig.colorbar(img, ax=ax_spec, fraction=0.03, label='log energy')

fig.tight_layout()
fig.savefig(OUT + 'fig-waveform-spectrogram.png')
plt.close(fig)

# ---------------------------------------------------------------
# 4) On-device vs cloud tradeoff: accuracy vs footprint
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.5))

systems = [
    ("Wake word\non-device\n(Amazon/Apple)", 0.05, 0.45, ORANGE),
    ("RNN-T 80MB\nGboard", 0.18, 0.72, BLUE),
    ("VoiceFilter-Lite\n2.2MB", 0.10, 0.62, GREEN),
    ("Conformer\n(AssemblyAI cloud)", 0.72, 0.90, RED),
    ("Whisper large\n(cloud)", 0.85, 0.88, GRAY),
    ("Batch diarization\n(Spotify cloud)", 0.60, 0.68, ORANGE),
]

for name, x, y, c in systems:
    ax.scatter(x, y, s=220, color=c, alpha=0.80, edgecolor='white', lw=1.5, zorder=3)
    ax.annotate(name, (x, y), xytext=(8, 4), textcoords='offset points',
                fontsize=8, color=c)

ax.axvline(0.40, color=GRAY, ls='--', lw=1)
ax.text(0.42, 0.46, 'on-device | cloud', color=GRAY, fontsize=9, rotation=90, va='center')

ax.set_xlabel('Relative compute / footprint (0=tiny always-on, 1=heavy cloud)')
ax.set_ylabel('Task accuracy proxy')
ax.set_title('On-device vs cloud: compute footprint vs accuracy\n(Illustrative. Each point is a production system.)')
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(0.35, 1.0)
fig.savefig(OUT + 'fig-ondevice-vs-cloud.png')
plt.close(fig)

print("wrote 4 figures to", OUT)
