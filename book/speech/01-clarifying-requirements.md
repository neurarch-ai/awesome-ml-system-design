# 1. Clarifying the requirements

Before drawing a single box, pin down exactly which speech problems you are
solving. "Voice input" can mean at least six different model families, and the
right design for each differs sharply. Here is how to work through it with an
interviewer.

---

**Candidate:** The prompt says voice input plus meeting transcription plus a wake
word. Are all three in scope, or is this primarily one of them?

**Interviewer:** All three. Users tap a mic and dictate; they also upload meeting
recordings; and we want a "Hey Product" trigger that works offline.

**Candidate:** For dictation: does it need to stream text back while the user is
still speaking, or is a short wait acceptable after they stop?

**Interviewer:** It must feel instant. Partials while they speak, final text within
a couple hundred milliseconds of the last word.

**Candidate:** For the meeting recordings: what does the output look like? Bare
transcript, or punctuated with "Speaker 1 / Speaker 2" labels?

**Interviewer:** Speaker-labeled, punctuated transcript. Latency is not critical;
accuracy is.

**Candidate:** The wake word: always-on, battery-powered phone. Can I run any
inference in the cloud for it?

**Interviewer:** The trigger detector itself must be on-device. A brief cloud
check to suppress false accepts is fine once it fires.

**Candidate:** Language and accent coverage. One locale, a fixed set, or
open-ended multilingual?

**Interviewer:** English first, multilingual later. Assume varied accents and
some background noise.

**Candidate:** Can audio leave the device? Retention and consent model?

**Interviewer:** Uploaded recordings are consented. Always-on wake-word audio
stays on the device until the trigger fires. Dictation audio can go to the cloud
under consent.

**Candidate:** Accuracy bar: WER (word error rate, the share of words the
transcript gets wrong) on what test set? Does the product care more
about proper nouns and numbers than raw WER suggests?

**Interviewer:** Proper nouns matter a lot. Names and addresses are where users
notice failures. Measure those separately.

---

## What we just locked down

Let me collect the consequences, because stating them explicitly is most of the
signal in the first few minutes.

**Functional scope:**

- Streaming ASR (automatic speech recognition, turning audio into text): emits
  partial and final hypotheses while the user speaks and
  finalizes within roughly 300 ms of the endpoint.
- Batch ASR with diarization (labeling who spoke when): full-context transcription of uploaded recordings,
  punctuated, with speaker-turn labels. Latency is minutes, not milliseconds.
- Always-on wake word: tiny on-device detector, loose threshold to avoid false
  rejects, cloud verification to kill false accepts.

**Non-functional bounds that will drive architecture:**

- Streaming ASR must be **causal**: at time t it can only use audio up to t. It
  cannot wait for the end of the utterance, so it commits to hypotheses it
  cannot revise. This rules out any attention encoder-decoder for the live path.
- Wake word must run **continuously** on a low-power core. Anything over a few
  megabytes drains the battery. This rules out a full ASR model.
- Batch transcription can use full context, so accuracy wins over latency. A
  bidirectional Conformer encoder is the right family here.
- Privacy: always-on audio stays on-device; the design must handle the fact that
  no audio logs flow back from the wake-word path, so on-device metrics or
  federated signals are needed for retraining.

Two consequences to state before any design:

- **Streaming and batch are two separate models and two separate serving paths,
  not one model with a flag.** They have different causality requirements,
  different architectures, and different evaluation metrics.
- **The wake word is not a small ASR model.** It is a detection problem, not a
  transcription problem. Conflating them is the most common early mistake.
