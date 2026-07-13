# 1. Clarifying the requirements

Before designing anything, pin down what the system must do. Here is a typical
exchange between a candidate and an interviewer. Notice that every question
either removes work or changes the design.

**Candidate:** What kind of model are we monitoring? A ranker, a classifier,
a retrieval model?
**Interviewer:** A recommendation ranker. It scores a few hundred candidates
and returns a ranked feed for each user.

**Candidate:** How fast do labels arrive? Do we learn the outcome seconds
after serving, or days later?
**Interviewer:** The primary signal is click-through. Clicks arrive within
seconds of the impression. Downstream engagement signals like long-term
retention take weeks.

**Candidate:** What does failure look like? Silent decay over weeks, or a
sudden crash?
**Interviewer:** Silent decay is the scenario to prevent. The model degraded
slowly for six months before anyone noticed.

**Candidate:** What can we log on the serving path? Can we capture the exact
features that produced each prediction?
**Interviewer:** Assume we control the serving layer and can log whatever we
need. Storage cost matters but is not a blocker.

**Candidate:** What is the cost of a false alarm versus a missed degradation?
A recommendation model that sends too many pages is a nuisance; a model that
silently fails loses engagement.
**Interviewer:** Both matter. The team has alert fatigue from noisy monitors
right now, so precision matters, but so does catching real regressions.

**Candidate:** How often can the team retrain? Daily? Weekly?
**Interviewer:** Retraining takes a day, including validation. They can do it
on demand but it is not free.

---

Let us summarize the problem statement. **We are asked to design a monitoring
system for a recommendation ranker.** Labels (clicks) arrive in seconds, so
we can measure accuracy fairly live, but the primary failure mode is slow
silent decay, not an acute crash. We can log anything at serving time. We
have an alert-fatigued team, so false alarms must be controlled. Retraining
is on-demand but costly, so triggers should be deliberate.

Three consequences fall out immediately:

- **We must monitor the model's inputs and outputs, not just its uptime.**
  Latency and error-rate monitoring would not have caught six months of
  silent decay. The serving process was healthy; the predictions were not.
  This is the central distinction between ML monitoring and service monitoring.

- **The monitoring loop has two speeds.** Drift checks on the feature log run
  immediately, no labels needed. Performance metrics (AUC, calibration) run
  once outcomes join back, which is fast for clicks but slow for retention.
  A good design uses the fast signals as an early warning and the slow ones as
  confirmation.

- **An alert must be diagnosable.** An alert that says "AUC dropped 2%" gets
  investigated; an alert that says "something changed" gets muted. The system
  must point at the specific feature, segment, or pipeline stage that moved.
