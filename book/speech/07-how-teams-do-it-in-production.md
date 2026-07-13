# 7. How teams do it in production

Every speech system in this chapter starts from the same front end (16 kHz, log-mel
features) and then diverges by latency requirement, compute budget, and what the
product actually needs from audio. The table below lines up the main decisions;
the writeups underneath explain why each team landed where they did.

## Where the real designs diverge

| System | Task | Architecture | On-device or cloud | Key design choice | Primary metric |
|---|---|---|---|---|---|
| Google Gboard (RNN-T) | Streaming ASR | RNN-T: 8 LSTM encoder + joint | On-device (80 MB int8) | Quantize to 80 MB; full offline, no server round-trip | WER + RTF below 1.0 |
| AssemblyAI Conformer-1 | Batch ASR | Conformer: grouped attn + sparse attn | Cloud (650K hr training) | Noise robustness via modified sparse attention and 650K hr data | WER per-condition, sliced by noise |
| OpenAI Whisper | Batch ASR + translation | Transformer encoder-decoder, weak sup. | Cloud | Zero-shot multilingual via 680K hr weak supervision | WER / CER across languages |
| Amazon Alexa wake word | Wake word | Two-stage: CNN on-device + CRA cloud | On-device + cloud verify | Metadata-conditioned on-device stage; CRA cloud verifier | False accepts per hour, FRR |
| Apple Hey Siri | Wake word + speaker ID | Two-stage: DNN trigger + DNN speaker embed | On-device (4x256 DNN, 8-bit) | Speaker verification via cosine on 442-dim supervector; EER 4.3% | EER per enrolled user |
| Spotify diarization | Diarization | VAD + VGGVox embeddings + sparse factorization | Cloud | Unsupervised, language-agnostic, overlap-aware via sparse matrix factorization | DER + purity + coverage |
| Google Tacotron 2 | TTS | Seq2seq acoustic model + WaveNet vocoder | Cloud | Split into mel-spectrogram stage and vocoder stage | MOS (human ratings) |
| Google VoiceFilter-Lite | Speaker separation | Mask network conditioned on d-vector, 2.2 MB | On-device | Target-speaker conditioning; asymmetric loss; degrades gracefully without enrollment | WER on overlapped speech |
| Meta SeamlessM4T | ASR + speech translation | Unified encoder-decoder, multimodal | Cloud | 100+ languages, unified speech and text in one model | WER / translation BLEU |
| NVIDIA Parakeet | Batch ASR | GPU-optimized Conformer family | Cloud (transcription farms) | Throughput-per-GPU optimization for production transcription at scale | WER + GPU hours per audio hour |

The dividing line is causality and power budget. Streaming on-device systems
(Gboard, Hey Siri, VoiceFilter-Lite) must fit a memory and power envelope and
commit left-to-right. Cloud batch systems (Conformer-1, Whisper, Tacotron 2,
Parakeet) can attend over the full audio and spend more compute.

## The systems (first-party write-ups)

- **Google** [An All-Neural On-Device Speech Recognizer](https://research.google/blog/an-all-neural-on-device-speech-recognizer/): RNN-T streaming ASR quantized from 450 MB to 80 MB for offline Gboard voice typing on Pixel phones. Shows how quantization and hybrid kernels achieve 4x compression with near-server WER. *(deployment)*

- **AssemblyAI** [Conformer-1: robust speech recognition trained on 650K hours](https://www.assemblyai.com/blog/conformer-1): Conformer batch ASR scaled to 650K hours with grouped attention, progressive downsampling, and moving-median sparse attention. 43% fewer errors on noisy real-world audio vs competitors. *(product design + eval bar)*

- **OpenAI** [Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://github.com/openai/whisper): Transformer encoder-decoder trained on 680K hours of weakly supervised, multilingual audio. Zero-shot across languages, tasks (ASR, translation, VAD), and domains. *(eval bar + product design)*

- **Amazon** [Alexa's new wake word research at Interspeech](https://www.amazon.science/blog/amazon-alexas-new-wake-word-research-at-interspeech): Metadata-aware on-device CNN stage plus CRA cloud verifier. Device-metadata conditioning cuts false-reject rate 14.6%; CRA cuts false accepts 60% on noisily aligned audio. *(product design)*

- **Apple** [Personalized Hey Siri](https://machinelearning.apple.com/research/personalized-hey-siri): Enrolls a speaker with five phrases, builds a 442-dim supervector, runs a 4x256 DNN to 100-dim embedding, verifies with cosine. Implicit profile update over 40 vectors. EER 4.3%, one false accept per month end-to-end. *(product design)*

- **Spotify** [Unsupervised Speaker Diarization using Sparse Optimization](https://research.atspotify.com/2022/09/unsupervised-speaker-diarization-using-sparse-optimization): VAD + VGGVox embeddings + sparse matrix factorization. Unsupervised, language-agnostic, overlap-aware. Beats Google Cloud diarization on hour-long podcasts. *(product design)*

- **Google** [Tacotron 2: Generating Human-like Speech from Text](https://research.google/blog/tacotron-2-generating-human-like-speech-from-text/): Seq2seq acoustic model to 80-dim mel-spectrogram plus WaveNet vocoder. MOS comparable to professional recordings. The canonical reference for the two-stage TTS pipeline. *(eval bar)*

- **Google** [Improving On-Device Speech Recognition with VoiceFilter-Lite](https://research.google/blog/improving-on-device-speech-recognition-with-voicefilter-lite/): 2.2 MB streaming target-speaker separation model. Runs on-device, conditions on enrolled d-vector, improves overlapped-speech WER 25.1% with asymmetric loss guarding against over-suppression. *(deployment)*

- **Meta** [SeamlessM4T: a foundational multimodal model for speech translation](https://ai.meta.com/blog/seamless-m4t/): Unified model for ASR, speech translation, and text translation across approximately 100 languages. *(who it serves)*

- **NVIDIA** [NeMo Parakeet ASR Models](https://developer.nvidia.com/blog/pushing-the-boundaries-of-speech-recognition-with-nemo-parakeet-asr-models/): GPU-optimized Conformer ASR family for high-throughput, low-WER production transcription. Optimized for transcription-farm economics, not just WER. *(deployment)*

- **PyTorch** [Forced Alignment with Wav2Vec2](https://docs.pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html): CTC trellis backtracking to align a known transcript to audio timestamps. Used for caption timing, TTS data preparation, and karaoke-style highlighting. *(deployment)*
