# 5. Evaluation

Link prediction is judged as ranking on a time-based split, and the online metric
is acceptance, not clicks.

## Offline metrics

- **AUC / average precision (AP)** over positive edges versus sampled negatives:
  can the model rank a true future edge above a non-edge. AP is more informative
  than AUC under the extreme non-edge imbalance.
- **Hits@k and MRR** per member: of the actual future connections, how many land in
  the top k suggestions, and how high. This matches the product (a short ranked
  list), so it is the metric to headline.
- **Always a time-based split.** Train on the graph as of T, evaluate on edges that
  formed after T. A random edge split leaks the answer.

$$\text{Hits@k} = \frac{1}{|U|}\sum_{u \in U} \mathbf{1}\!\left[\text{a true future edge of } u \in \text{top-}k(u)\right]$$

## Online metrics

- **Invitation acceptance rate**, not invitations sent. The product wins on
  accepted connections; a suggestion that gets invited and rejected is a cost (it
  annoys the recipient), so acceptance is the true objective.
- **Downstream network growth and engagement**, since a good connection drives
  long-term activity, and a bad one adds spammy invites.
- **Coverage and fairness.** Does the system only suggest hub nodes and dense
  communities, starving new or peripheral members? Track suggestion coverage across
  the degree distribution.

## When to use which metric

| Reach for | When | Instead of |
|---|---|---|
| Hits@k / MRR | judging the ranked suggestion list as the product sees it | AUC alone, which does not reflect a short top-k list |
| AP over AUC | the non-edge class massively outnumbers edges | AUC, which looks high even for a weak model under imbalance |
| Time-based split | any offline link-prediction eval | a random edge split, which leaks the future edge |
| Online acceptance rate | the launch decision | invitations sent, which rewards spammy over-suggesting |

The guardrail to state: an offline Hits@k gain must survive an online A/B on
**acceptance rate** and coverage, because a model that only re-suggests popular hubs
can win offline ranking while flooding people with unwanted invites.
