# 10. Putting it together: the complete build

Sections 1 through 6 taught each stage with its options and tradeoffs; section 7
showed where real teams diverge. What none of them show is a single system with
every decision made. This capstone does three things: it gives you an opinionated
default stack so option paralysis never blocks a first build, it walks the
chapter's scenario end to end with every choice committed and costed, and it
shows how the same decisions flip when the constraints change. It closes with
the smallest runnable speech-metrics core, one file, no installs.

## The default stack: start here, deviate with reason

Every stage in this chapter has several credible options, and a first-time
builder can burn a week comparing toolkits before transcribing a single
utterance. Skip that. The stack below is a sane default for a first production
build; each row names when to deviate and which section explains why. Toolkits
change yearly, but the interface of each stage (featurize, augment, label,
model, decode, evaluate, serve) does not, so pick per stage by interface and
treat any specific library as replaceable.

| Stage | Default | Deviate when | Why (section) |
|---|---|---|---|
| Audio frontend | 16 kHz, 25 ms window, 10 ms hop, 80-bin log-mel, frozen identically between train and serve | Labeled data is scarce: a self-supervised encoder (wav2vec 2.0 / HuBERT) replaces hand-built features | [3](03-data-preparation.md) |
| Augmentation | SpecAugment always, plus noise and reverb mixing | Product is strictly near-field and clean (rare): drop the reverb, keep SpecAugment | [3](03-data-preparation.md) |
| Labels | Weak supervision at scale for training; human transcribers only for the golden eval set | New language with a few labeled hours: self-supervised pretrain, fine-tune a small CTC head | [3](03-data-preparation.md) |
| Streaming acoustic model | Causal RNN-T with an emission-delay regularizer (FastEmit class) | Forced alignment is the job, or an external LM is acceptable: CTC | [4](04-model-development.md) |
| Batch acoustic model | Conformer encoder-decoder, chunk long audio with 1-2 s overlap and stitch | Zero-shot multilingual breadth beats per-domain tuning: Whisper-style weak supervision | [4](04-model-development.md) |
| Decoding and LM fusion | RNN-T internal LM, beam width 4-8 for streaming | CTC path: shallow-fuse an external LM at decode time; on-device memory forbids it | [4](04-model-development.md) |
| Wake word | Two-stage: tiny loose on-device detector, heavier cloud verifier | No always-on trigger in the product: delete the whole subsystem | [4](04-model-development.md), [6](06-serving-and-scaling.md) |
| Evaluation | WER sliced by accent, noise, and entity; endpoint latency and RTF for streaming; DET curve for wake word | Never. Slice before every release decision | [5](05-evaluation.md) |
| Serving | Two separate paths: stateful per-session streaming, throughput-batched offline | Never merge them; a "streaming flag" on a batch model is the classic trap | [6](06-serving-and-scaling.md) |

The last two rows are the ones beginners skip and regret: an aggregate WER hides
the accent cohort that will file the bug reports, and a single model serving both
workloads satisfies neither causality requirement. State both refusals up front.

## The complete build

Return to the scenario from [section 1](01-clarifying-requirements.md): live
dictation with partials while the user speaks and final text within roughly
300 ms of the endpoint; uploaded meeting recordings transcribed with speaker
labels and punctuation; an always-on "Hey Product" wake word that runs
on-device with a cloud check to suppress false accepts; English first with
varied accents and background noise; proper nouns and numbers measured
separately; always-on audio never leaves the device. Here is the whole system
with every choice committed and the reason it won.

| Decision | Choice | Why it won |
|---|---|---|
| Frontend | 16 kHz, 80-bin log-mel at 10 ms hop, shared across every branch, frozen at serve | One frontend keeps wake word, streaming, and batch coherent; a train/serve stats mismatch is the classic silent killer |
| Dictation model | Causal RNN-T, int8-quantized to run on-device, cloud fallback for weak hardware | Sub-300 ms partials force causality; the internal LM removes an external LM the phone cannot fit |
| Emission control | FastEmit-style delay penalty in the transducer loss | Untreated RNN-T learns to hoard frames; two models with equal WER can feel completely different live |
| Endpointing | Cheap per-frame detector fusing silence with hypothesis completeness | Runs on every frame, so it must cost almost nothing; VAD alone cuts off mid-utterance pauses |
| Wake word stage 1 | Tiny on-device detector, a few MB, deliberately loose threshold | False rejects are fatal to the product; the low-power core bounds the size |
| Wake word stage 2 | Cloud verifier on the triggered snippet | Fires rarely, so it can afford the heavier model that kills stage 1's false accepts |
| Batch model | Conformer encoder-decoder, 1-2 s overlap chunking, then diarize and punctuate | Uploaded recordings have no latency bound, so full context and self-correction win |
| Diarization | Embedding plus clustering, scored on DER with purity and coverage | The requirement is speaker-labeled transcripts; WER says nothing about turn accuracy |
| Data | Weak supervision for training volume; human transcripts only for the golden set; TTS-synthesized wake word data plus hard negatives | Human hours are too expensive to train on and too valuable not to evaluate on |
| Augmentation | SpecAugment plus noise and reverb mixing, speed perturbation 0.9x-1.1x | The stated conditions are varied accents and background noise; clean-only training collapses there |
| Eval gates | WER sliced by accent and noise, entity and numeric WER separate, endpoint latency, RTF, false accepts per hour | The interviewer said names and addresses are where users notice failures; aggregate WER underweights exactly those |
| Retraining signal | On-device correction events and consented opt-in audio; no logs from the wake-word path | Always-on audio stays on the device, so the design must plan for its absence from day one |

**Model footprint.** The chapter's on-device reference points bound the budget:
int8 quantization shrinks a float32 streaming model roughly 4x (Google's Gboard
RNN-T went from 450 MB to 80 MB at a 4x runtime speedup), and the always-on
wake word stage must stay within a few megabytes on the low-power core, with
tens-of-kilobyte detectors possible. The WER cost of quantization is validated
per-model, never assumed: the 4x size reduction is arithmetic, the accuracy
cost is a property of the specific trained weights.

**Audio-hours budget.** Human transcription runs roughly \$10 to \$30 per audio
hour, and a decent supervised baseline wants hundreds of hours per language:
call it 500 hours at a \$20 midpoint, \$10,000 per language before a single
accent slice (Illustrative). That price is why training volume comes from weak
supervision (Whisper's 680K hours would cost around \$13.6M at the same rate)
and why the human dollars go to the golden eval set instead: 50 carefully
transcribed hours, \$1,000-1,500, sliced by accent, noise, and entity density
(Illustrative), is the asset every release decision leans on.

**Real-time factor and latency.** Streaming needs RTF below 1.0 to keep up and
below 0.1 on mobile for headroom: at a 10 ms frame hop, an RTF of 0.1 leaves
about 1 ms of compute per frame, which is why the joint network stays shallow
and the beam stays at 4-8. The user-facing budget is under 300 ms to first
partial and roughly 300 ms from endpoint to final text. The batch path inverts
the math: a 30-second clip returns nothing for 30-35 seconds, and the metric
that matters is audio hours per GPU-hour; at a batch RTF of 0.05 one GPU clears
20 hours of audio per hour (Illustrative), which is a throughput-scheduling
problem, not a latency one.

**WER targets by cohort.** The chapter's own warning sets the gates: a 5%
aggregate WER can coexist with 12% for an accent group, 15% for children, and
20% for far-field audio. So the release gate is not one number but a family:
aggregate at or below 5%, no sliced cohort worse than twice the aggregate,
entity and numeric WER reported separately and trending down (Illustrative
gates). The wake word gates on its own axes entirely: under one false accept
per day of ambient audio, with false-reject rate read off the DET curve at the
chosen operating point.

**What breaks in month one.** Three failure modes dominate early operations, so
wire their signals before launch: endpoint false cutoffs (users pause
mid-dictation to think, the endpointer finalizes on the silence, and the
complaint is "it cuts me off" while WER stays flat, so track false cutoff rate
as its own metric), wake-word operating-point drift (false accepts per hour
climbs in noisy homes and on new device types while the lab DET curve still
looks fine, so monitor the rate per environment and watch stage 2 verifier
load), and hallucination on silence in the batch path (the attention decoder
emits fluent text over non-speech spans of meeting audio, so alarm on
transcribed words inside VAD-flagged silence).

## The same techniques under different constraints

The review question that matters in practice is not "which architecture is
best" but "which architecture is best under my constraints." Here is the same
pipeline built three times. Only the middle column is the build above; the
other two keep the identical stage interfaces and swap nearly every
implementation choice.

| | On-device command recognition | Voice product (this chapter) | Batch transcription farm |
|---|---|---|---|
| Workload | A few dozen fixed commands on an earbud or speaker; no network guaranteed | Open-vocabulary dictation, uploads, and a wake word | Podcast and meeting archives; thousands of audio hours nightly, no interactivity |
| Latency budget | Trigger-to-action under a few hundred ms, always | First partial under 300 ms live; minutes for uploads | None; audio hours per GPU-hour and cost are the only axes |
| Acoustic model | One tiny keyword-spotting or phrase-level model; no full ASR at all | RNN-T streaming plus Conformer batch, two models | Largest accurate Conformer or Whisper-class model the farm affords |
| Decoding and LM | Thresholded scores over a closed grammar; no LM fusion | Internal LM in the transducer; beam 4-8 streaming | Wide beam, external LM rescoring; decode time is amortized across the batch |
| Data | TTS-synthesized phrases plus hard negatives per command | Weak supervision at scale plus a human golden set | Weak supervision; per-domain fine-tunes where volume justifies them |
| Eval gate | False accepts per hour and FRR per command, on the DET curve | Sliced WER plus endpoint latency plus RTF plus DET | WER per domain and cost per transcribed hour |
| Serving | Everything on the low-power core; cloud is optional or absent | Split paths: stateful streaming sessions, throughput batch | Dynamic batching by audio length; spot GPU capacity; chunk, stitch, diarize |
| What would be over-engineering | Any open-vocabulary ASR, any cloud dependency, any LM | A single unified model for all tasks | Streaming anything: endpointers, partials, per-session state |

Two lessons fall out. First, the command-recognition column is mostly
deletions: when the vocabulary is closed, transcription itself disappears and
the whole problem collapses into the wake-word framing of
[section 2](02-frame-as-ml-task.md), gated by false accepts per hour rather
than WER. Second, the farm column shows latency and cost trading places as the
binding constraint: with no user waiting, causality stops mattering, the model
grows, the beam widens, and every optimization chases audio hours per GPU-hour
instead of milliseconds per frame.

## What each constraint decides

The compressed decision guide. Read the left column off your requirements; the
right columns say which lever it moves before you compare any tools.

| Your constraint | Lever it moves | Rule of thumb |
|---|---|---|
| First-partial latency | Causality, hence architecture family | Under ~300 ms: causal RNN-T or CTC, never an attention seq2seq that waits for the utterance |
| Memory and battery envelope | Model family, quantization, beam width | On-device: int8 RNN-T, shallow joint, beam 4-8; a full bidirectional Conformer does not fit at competitive WER |
| Always-on requirement | Detector design, not ASR | Two stages: loose tiny on-device model, heavier verifier behind it; gate on false accepts per hour |
| Privacy (audio cannot leave the device) | Retraining signal | Plan on-device correction metrics, federated signals, or consented opt-in from day one; there will be no audio logs |
| Labeled-hours budget | Training strategy | Below ~10 labeled hours: self-supervised pretrain plus a small CTC head; at scale: weak supervision, humans only for the golden set |
| Accent and noise diversity | Augmentation and eval slicing | SpecAugment plus noise and reverb mixing; gate every release on per-cohort WER, not the aggregate |
| Entity-heavy content | Metric design | Names, addresses, numbers: report entity and numeric WER separately; aggregate WER underweights exactly these |
| Multi-speaker audio | Pipeline stages, metric | Uploads: diarize and score DER with purity and coverage; live overlap: target-speaker separation conditioned on the enrolled voice |
| Language coverage | Pretraining scope | Multilingual joint pretraining or a Whisper-class model; per-language supervised training does not scale to the tail |
| No latency bound at all | Everything | Full context, biggest accurate model, wide beam, dynamic batching; optimize audio hours per GPU-hour |

## The smallest runnable speech metrics

The review of every toolkit tutorial is the same: the reader assembles five
dependencies and still cannot see the computation. So here are the two
mechanisms this chapter leans on hardest, in one file with zero installs. The
first is WER as an edit-distance alignment with the S, I, D counts separated,
including the classic gotcha that WER can exceed 100%. The second is dynamic
time warping, the alignment-tolerant distance that lets two renditions of the
same word match despite different speaking rates, which is the same
marginalize-over-alignments idea that CTC and RNN-T build their losses on. The
toy 1-D feature track stands in for the 80-bin log-mel frames of
[section 3](03-data-preparation.md).

```python
"""WER and DTW from scratch, runnable with no installs."""
import random

# --- WER: word-level edit distance with a backtrace ------------------------

def wer(ref, hyp):
    """Return (WER, S, I, D); production: jiwer plus a shared text normalizer."""
    r, h = ref.split(), hyp.split()
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1): d[i][0] = i          # delete everything
    for j in range(len(h) + 1): d[0][j] = j          # insert everything
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1  # 0 match, 1 substitution
            d[i][j] = min(d[i - 1][j] + 1,           # deletion
                          d[i][j - 1] + 1,           # insertion
                          d[i - 1][j - 1] + cost)    # substitution or match
    i, j, S, I, D = len(r), len(h), 0, 0, 0          # backtrace to split the count
    while i > 0 or j > 0:
        sub = 0 if (i and j and r[i - 1] == h[j - 1]) else 1
        if i and j and d[i][j] == d[i - 1][j - 1] + sub:
            S += sub; i -= 1; j -= 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            D += 1; i -= 1
        else:
            I += 1; j -= 1
    return (S + I + D) / len(r), S, I, D

PAIRS = [
    ("the cat sat on the mat", "the cat sat on the mat"),
    ("the cat sat on the mat", "the bat sat on mat"),
    ("call doctor reyes at noon", "call doctor race at new noon"),
    ("yes", "yes yes okay yes"),                     # WER above 100% is legal
]
for ref, hyp in PAIRS:
    w, S, I, D = wer(ref, hyp)
    print(f"WER {w:7.1%}  S={S} I={I} D={D}  ref='{ref}' | hyp='{hyp}'")

# --- DTW: alignment-tolerant distance between two utterances ----------------

def utterance(word, speed, seed):
    """Toy 1-D feature track; production: 80-bin log-mel frames at a 10 ms hop.
    The word fixes the shape (its anchors); speed stretches or squeezes time."""
    shape = random.Random(word)                       # same word -> same anchors
    anchors = [shape.uniform(-1, 1) for _ in range(8)]
    n = int(40 / speed)                               # frame count varies with speed
    noise = random.Random(seed)
    track = []
    for i in range(n):
        pos = i * (len(anchors) - 1) / (n - 1)
        k, f = int(pos), pos - int(pos)
        v = anchors[k] if k == len(anchors) - 1 else anchors[k] * (1 - f) + anchors[k + 1] * f
        track.append(v + noise.gauss(0, 0.03))
    return track

def dtw(a, b):
    """Classic DP: warp time so each frame pairs with its best counterpart."""
    INF = float("inf")
    d = [[INF] * (len(b) + 1) for _ in range(len(a) + 1)]
    d[0][0] = 0.0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            step = abs(a[i - 1] - b[j - 1])
            d[i][j] = step + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return d[len(a)][len(b)] / (len(a) + len(b))      # normalize by path length

def naive(a, b):
    """Frame i compared to frame i, no warping; truncate to the shorter track."""
    n = min(len(a), len(b))
    return sum(abs(a[i] - b[i]) for i in range(n)) / n

slow = utterance("hello", speed=0.8, seed=1)          # 50 frames
fast = utterance("hello", speed=1.4, seed=2)          # 28 frames
other = utterance("play", speed=1.0, seed=3)          # 40 frames
print(f"\nframes: hello-slow={len(slow)} hello-fast={len(fast)} play={len(other)}")
print(f"hello-slow vs hello-fast   naive={naive(slow, fast):.3f}  dtw={dtw(slow, fast):.3f}")
print(f"hello-slow vs play         naive={naive(slow, other):.3f}  dtw={dtw(slow, other):.3f}")
```

Run it and both halves land their point. The WER block prints the S, I, D
decomposition per pair: a perfect hypothesis scores 0.0%, a substitution plus a
deletion on a six-word reference scores 33.3%, a mangled proper noun ("reyes"
to "race" plus an inserted word) scores 40.0%, and the one-word reference
answered with four words scores 300.0%, three insertions against N = 1, the
proof that WER is not a probability. The DTW block is sharper: the naive
frame-by-frame distance actually ranks the wrong word as closer (0.416 for
"hello" vs "play" against 0.552 for the two "hello" renditions), because
comparing frame i to frame i punishes the speed difference more than the
identity difference. DTW reverses the verdict decisively, 0.032 for the same
word against 0.210 for the different one, by letting one slow frame align with
several fast ones. That tolerance to time-warped alignments is exactly what
CTC and RNN-T internalize when they marginalize over every frame-level path
that collapses to the target transcript, and the toy track you just warped is
the stand-in for the log-mel frames every model in this chapter consumes.
