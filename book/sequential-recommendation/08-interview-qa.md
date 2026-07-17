# 8. Interview Q&A

The questions an interviewer actually asks about sequential recommendation,
grouped by how they are used. The commonly-missed ones are where interviews are
won or lost.

## Commonly asked

**Q: Why use a sequence model instead of just aggregating the user's history into
a feature vector?**
A: Aggregates lose two things: **order** and **recency**. A user who just
switched from cooking to travel looks identical to a steady cooking fan under
lifetime category counts. The sequence model sees the recent shift and can weight
it appropriately. Instacart's experiment is the concrete evidence: randomly
shuffling the sequence order (so the model cannot use order) dropped recall by
10-45% across surfaces. Order is not a detail; it is the signal.

**Q: Where in the funnel does this model sit?**
A: It depends on the latency budget and what you already have. A sequence encoder
can power a retrieval user tower (producing the user embedding for ANN search,
like PinnerFormer) or a ranking feature (producing a feature vector the ranker
uses to score candidates, like BST or TransAct). The latency budget differs: a
retrieval user tower can be computed and cached at session start, while a ranking
feature may need to reflect the last few actions before each scoring call. Pin
down which one; it changes whether you need a streaming pipeline or a batch
update is enough.

**Q: How do you keep the user state fresh within a session?**
A: A streaming pipeline (Kafka or similar) ingests each user action and appends
it to a per-user recent-event store (Redis, DynamoDB). The next request reads
the updated sequence from the store, encodes it, and the encoder sees the latest
action. The non-trivial part: the online sequence must be built by exactly the
same logic as the offline training sequences, or the encoder serves on a
distribution it never trained on.

**Q: How do you handle a new user with no interaction history?**
A: Degrade gracefully down the same model rather than switching to a separate
one. Empty sequence: fall back to popularity and context (location, device, time
of day). Short sequence (2-5 events): the session-based sequence model already
works; even two actions give intent signal. Content features (category, price
tier) carry cold items and cold users until the ID embeddings accumulate training
signal. The key point for an interviewer: cold start is a degradation path, not
a second model.

**Q: How do you evaluate a sequential recommendation model offline?**
A: Held-out next-item prediction on a **time-based split**: for each user, hold
out their last interaction as the test target, use everything earlier as context.
Measure Recall@k (is the true next item in the top k?) and NDCG@k (how highly
is it ranked within the top k?). Do not use a random split; it leaks future
events into the training context.

## Tricky follow-ups

**Q: You injected a positional index (1st, 2nd, 3rd action) as the time signal.
What does that miss?**
A: Position encodes order but not elapsed time. Two actions one second apart
differ from two actions one month apart, but a plain 1..N index assigns them
identical offsets. Strong systems encode the actual **time gap** between
consecutive events, not just their rank in the sequence, so the model learns that
a recent action carries more recency weight than an old one at the same position.
This is the single most-skipped detail in interview answers about sequence models.
**Why:** attention can only condition on what is in the input representation.
With a bare 1..N index, an action from one second ago and an action from one
month ago produce identical position features, so the model has no way to learn
different weights for them even if the signal is strong; adding the time gap
gives attention an actual feature that separates "same session" from "stale
history."

**Q: You said training-serving skew is the headline risk. What specifically
causes it here?**
A: The batch pipeline builds training sequences with explicit rules: dedup
adjacent identical items, filter rare items, cap at N, break ties on simultaneous
events by item ID. The streaming pipeline builds sequences from a live event
stream. If either set of rules differs even slightly (a different dedup window, a
missing filter, a different tie-break), the encoder sees sequence distributions
at serving that it never saw during training. The fix is not a better monitoring
dashboard; it is shared code for sequence construction that both pipelines call.
**Why a slight rule difference is enough to hurt:** the encoder's attention
patterns are fit to the exact statistics of the training sequences (typical
lengths, adjacency structure, duplicate patterns), so a changed dedup window
shifts those statistics and puts every serving request slightly
out-of-distribution, where the model degrades silently rather than erroring.

**Q: Pinterest uses both TransAct and PinnerFormer. Why two models?**
A: They cover two timescales. TransAct captures the last 100 real-time actions,
which reflects what the user is doing right now in the current session.
PinnerFormer captures longer-term taste from a daily batch embedding. Fusing both
(in the Homefeed ranker via DCN v2 feature crossing) covers what neither handles
alone: a casual user's current-session intent on top of their durable preferences.
The all-action loss in PinnerFormer is what makes a daily batch embedding
competitive with a real-time one, by training it to predict a window of future
actions rather than just the next one.

**Q: Why does self-attention outperform a recurrent net on long sequences?**
A: A GRU or LSTM compresses the entire past into one fixed-size hidden state and
passes it forward one step at a time. For a sequence of 100 items, the gradient
from item 100 back to item 1 has to pass through 99 recurrent steps; long-range
dependencies are hard to learn. Self-attention connects every pair of positions
directly (one attention step), so the gradient from the last position to any
earlier position is a single multiply. It also parallelizes at training time,
which makes it faster to train on long sequences. Spotify CoSeRNN accepts the
recurrent structure deliberately because it only needs to react at session
granularity, not per-action; name that tradeoff explicitly.

**Q: BERT4Rec trains bidirectionally but you serve next-item prediction
causally. Isn't that a train-serve mismatch?**
A: It is a real subtlety, and handling it is exactly why BERT4Rec (Alibaba, 2019)
appends a special `[mask]` token at the end of the sequence for the next-item
task. During training it masks random interior positions and predicts them from
both sides (the Cloze objective), which is where the bidirectional context comes
from. At serving you cannot see the future, so you append a `[mask]` at the end
and read the prediction off that position only; the model is deliberately trained
with end-position masks so this readout is in-distribution. SASRec (2018) sidesteps
the whole issue: its causal mask means every position is already trained to predict
the next item from left context alone, so the training computation and the serving
computation are identical. That train-serve consistency is precisely the tradeoff
you give up when you choose BERT4Rec for bidirectional context.

**Q: In-batch negatives and sampled-softmax negatives look similar; both replace
the full-catalog softmax with a small negative set. When does the difference
actually matter?**
A: The difference is the distribution the negatives are drawn from. In-batch
negatives reuse the other users' positive items in the same batch, so negatives
arrive in proportion to item popularity: popular items appear in many rows and
therefore get used as negatives far more often. Sampled softmax draws negatives
from an explicit distribution you choose, typically uniform over the catalog.
The difference matters exactly when catalog popularity is skewed, which is
always: uncorrected in-batch training systematically punishes head items (the
logQ correction exists to cancel that), while pure uniform sampling mostly
serves up trivially easy negatives (random items the user was never close to),
so gradients carry little signal and the embedding space stays coarse. In-batch
popularity-weighted negatives are what make training hard enough to be useful,
which is why production recipes start from in-batch negatives plus the
correction rather than either pure option.

## Commonly answered wrong

**Q: Can I model the sequence as a bag of items, or does order really matter?**
A: Order matters, and the evidence is direct: Instacart showed that shuffling the
sequence order (removing order without removing any items) dropped recall by
10-45% across surfaces. The bag-of-items answer is the most common wrong answer
in this topic; it is also the easiest trap to lay because it looks harmless. If
order did not matter, you would not need a sequence model at all.
**Why order carries the signal:** two users with the identical item set but
opposite orderings are moving in opposite directions (one drifting from cooking
toward travel, the other the reverse), and their next items differ accordingly.
A bag representation is identical for both by construction, so an order-blind
model is forced to give them the same prediction; the information that
distinguishes them exists only in the ordering.

**Q: DIN uses attention over the user's history. Does it model sequence order?**
A: No. DIN's local activation unit computes attention weights between each
historical behavior and the current candidate ad, and **pools** the results with
a weighted sum. It has no positional signal, no causal masking, and no softmax
normalization over behaviors (the normalization is deliberately absent to preserve
interest intensity). DIN adapts the user representation per candidate but ignores
the order of behaviors entirely. BST is what adds sequential order on top of that
idea. Claiming DIN models sequence order is one of the most common and most
revealing mistakes in this interview topic.
**Why pooling erases order by construction:** a weighted sum is
permutation-invariant, meaning you can shuffle the history into any order and,
with no positional signal feeding the weights, every term in the sum is
unchanged, so the output is bit-for-bit identical. Order sensitivity has to be
injected explicitly (positional encodings, causal masking), which is exactly
what BST adds.

**Q: Is training-serving skew just a matter of preprocessing? Can I fix it by
standardizing the feature pipeline?**
A: It is broader than preprocessing. In sequential recommendation, the sequence
itself is constructed differently in batch (from logs) and in streaming (from a
live event feed), and the differences in dedup logic, filtering, or
tie-breaking propagate directly into what the encoder sees. Standardizing
features (scaling, bucketing) does not fix a mismatch in the sequence that the
features came from. The fix is shared code that constructs the sequence, not just
shared feature transforms.

**Q: A new user has no history, so I should train a separate cold-start model.**
A: You should degrade the same model, not switch to a separate one. A cold-start
user is just a user with a short or empty sequence. The session-based path
already handles short sequences; content features and popularity cover the empty
case. A separate model means twice the infrastructure, twice the maintenance, and
a hard boundary at the cold-warm transition that usually creates a discontinuous
experience. The degradation-ladder framing (session model, then content, then
popularity) handles the full spectrum with one system.
**Why the boundary is discontinuous:** two models trained on different data with
different objectives produce scores in different spaces, so the day a user
crosses the history threshold their recommendations jump to a different model's
opinion all at once. A single model with shared parameters shifts smoothly
instead, because each new event only incrementally updates the same
representation that was already serving them.
