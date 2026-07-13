# 1. Clarifying the requirements

Before designing anything, pin down what the system must do. Here is a typical
exchange. Notice that every question either removes work or changes the design,
and that the answers to the first two questions alone determine the dominant
engineering constraint.

**Candidate:** What does the model predict, exactly: click, conversion, or both?
**Interviewer:** Probability of click given the user, the ad, and the page context.
We may want pCVR later, but for now focus on pCTR.

**Candidate:** Who consumes the score, and how?
**Interviewer:** An auction. We compute eCPM as bid times pCTR and rank ads by
eCPM. The top ad wins and is charged second-price.

**Candidate:** That means calibration is load-bearing, not cosmetic. A score that
is off by 20 percent does not reorder a few ads; it mis-prices every auction it
touches. Is that a fair statement?
**Interviewer:** Exactly right. Calibration is a non-negotiable output.

**Candidate:** How many candidate ads per request, and what is the latency budget?
**Interviewer:** Tens to a few hundred eligible ads per slot. Scoring must fit
inside a page-load budget: low tens of milliseconds, often less.

**Candidate:** What is the feature space?
**Interviewer:** Massive and sparse. User ids, ad ids, advertiser ids, creative
ids, placement, device, geo, plus feature crosses. Millions to billions of
distinct categorical values.

**Candidate:** How do labels arrive? Are clicks and conversions immediate?
**Interviewer:** Clicks come back in seconds. Conversions can arrive days later
inside an attribution window, or never.

**Candidate:** What volume, and does the model need to track drift?
**Interviewer:** Very high QPS across many ad slots. Campaigns launch and pause
constantly, so yes, the model must retrain continuously or at high cadence.

---

Let us summarize the problem statement. **We are asked to design a pCTR
prediction system that feeds a real-time second-price auction.** The input is a
(user, ad, context) tuple; the output is a calibrated probability of click. The
system must score tens to a few hundred candidate ads per request in low tens of
milliseconds, operate over a sparse feature space of billions of categorical ids,
and retrain continuously to track campaign and demand drift.

Three consequences fall out of this immediately, and stating them early is the
signal the interviewer is looking for:

- **Calibration is the primary output constraint, not ranking quality.** The
  auction multiplies pCTR into a bid and derives a second-price charge from the
  result. A model with great AUC but drifted calibration silently mis-prices every
  slot, at every QPS. Say this before you name a single model family.
- **The feature space is dominated by sparse embedding tables, not by MLP
  parameters.** Hundreds of millions of user and ad ids, each needing a learned
  vector. The tables are measured in gigabytes; the dense network is small. This
  sizes the infrastructure, shapes the training pipeline, and forces feature
  hashing for open-ended id spaces.
- **Delayed conversions mean some labels are unresolved, not negative.** A click
  that has not yet converted is not a confirmed non-conversion; treating it as
  negative at training time biases pCVR downward and under-bids real value.
  The label pipeline must know the difference.
