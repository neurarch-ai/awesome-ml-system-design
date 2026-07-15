# 3. Batching and throughput

## Why a single request underuses hardware

A GPU (or even a well-optimized CPU inference path) is a massively parallel
device. Sending one request at a time leaves most compute lanes idle: the
arithmetic happens in nanoseconds and the rest of the time is kernel launch
overhead, memory transactions, and waiting. **Dynamic batching** collects
requests that arrive in a short time window and sends them through the model as
one combined batch, amortizing the fixed overhead and filling the hardware.

The throughput gain is real. Pinterest moved from CPU scatter-gather to GPU
batching and saw copy latency drop from 10 ms to under 1 ms by coalescing
tensors into one pre-allocated buffer before a single device transfer. That
single change was more impactful than any model optimization.

## The latency-throughput tradeoff

Dynamic batching introduces a direct tradeoff: a longer wait window and a
larger maximum batch raise throughput and raise tail latency. The model does
more useful work per second, but each request waits longer before its batch is
dispatched.

Formally, if $W$ is the wait window and $B$ is the effective batch size, the
latency a request sees inside the server is:

$$L_{\text{batch}} \;=\; W + \frac{B}{\text{tput}(B)}$$

The total QPS the server delivers is approximately:

$$\text{QPS} \;\approx\; \frac{B}{W + L_{\text{model}}(B)}$$

And the p99 budget constraint is:

$$T_{p99} \;\geq\; L_{\text{net}} + L_{\text{feat}} + W + L_{\text{model}}(B)$$

**Size $W$ and $B$ backwards from $T_{p99}$, not for peak throughput on idle
hardware.** Benching at low load gives misleadingly large $W$ budgets; the
constraint bites at the tail under real traffic.

## Little's law: sizing the replica fleet

Little's law connects queue length, arrival rate, and wait time:

$$\bar{Q} \;=\; \lambda \cdot \bar{W}_{\text{queue}}$$

where $\lambda$ is the arrival rate (QPS) and $\bar{W}_{\text{queue}}$ is the
mean time in the queue. The utilization per replica must stay below 1:

$$\rho \;=\; \frac{\lambda}{N_{\text{replicas}} \cdot \mu} \;\lt\; 1$$

where $\mu$ is the per-replica throughput in requests per second. To size the
fleet, invert: you need at least $\lceil \lambda / \mu \rceil$ replicas to stay
below saturation, then add headroom for cold-start and traffic spikes (covered
in section 5).

## CPU vs GPU: different cost curves

On CPU, inference latency scales roughly linearly with batch size because the
compute is serial:

$$L_{\text{CPU}}(B) \;\approx\; c_0 + c_1 B$$

On GPU, the cost curve is sub-linear: a larger batch fills the SIMD lanes that
would otherwise idle, so latency grows more slowly than batch size:

$$L_{\text{GPU}}(B) \;\approx\; g_0 + g_1 B^{\alpha}, \quad \alpha \;\lt\; 1$$

![CPU vs GPU latency scaling with batch size](assets/fig-latency-vs-batch.png)

*On CPU, latency grows linearly: doubling the batch roughly doubles the time.
On GPU, latency is sub-linear: bigger batches fill the accelerator without
proportional cost. The crossover point is where GPU starts winning per-request.
Illustrative, not a benchmark.*

This difference changes the batching strategy. On CPU you size the window
conservatively to hold the SLO; on GPU you batch larger to amortize the high
fixed cost of GPU invocation.

## Batch fill efficiency

The fraction of the maximum batch that is actually used on average:

$$\eta \;=\; \frac{\mathbb{E}[B]}{B_{\max}} \;=\; \frac{\min(\lambda W,\; B_{\max})}{B_{\max}}$$

At low QPS, $\lambda W \ll B_{\max}$ and the batch is mostly empty. At high QPS
the batch fills. This means batching mostly helps at sustained high load, where
it matters most; at low load the wait window dominates and adds latency for
little gain.

## When to use which batching approach

| Reach for | When | Instead of |
|---|---|---|
| Dynamic batching with short window | Online serving with a p99 SLA; tune window against the budget | Static batch size, which wastes throughput when traffic is below peak |
| Large GPU batches (Pinterest-style) | GPU hardware with sub-linear cost curve; many candidates per request already provide a natural batch | CPU-era scatter-gather, which wastes GPU lanes |
| No batching (request-at-a-time) | Tiny model, CPU, or p50 is the SLA not p99; latency budget leaves no room for a wait | Any GPU-heavy model where the accelerator is underused |
| Micro-batching within one request | A ranker scoring hundreds of candidates for one user: the candidates are already a natural batch | Dynamic cross-request batching, which adds cross-user complexity |
| Prediction caching | Inputs repeat (e.g., popular item or static context); high cardinality queries make it dead weight | Low-cardinality or real-time personalized inputs where cache hit rate is near zero |

**Provenance.** The production form of cross-request dynamic batching, a wait-window plus max-batch knob sized against the tail budget, is a first-class feature of Triton Inference Server (NVIDIA) and TorchServe (Meta); the queue-then-coalesce mechanism described here is the one those servers implement.

**Tools.** Dynamic cross-request batching ships in Triton Inference Server (NVIDIA), TensorFlow Serving, TorchServe (Meta), and Ray Serve, all of which expose a wait-window and max-batch knob to size against the p99 budget. For GPU-heavy autoregressive models, vLLM adds continuous (in-flight) batching so slots refill without waiting for the whole batch to finish. Micro-batching within one request is usually hand-rolled by stacking the candidate tensors before a single forward pass in PyTorch (Meta) or ONNX Runtime. Prediction caching leans on Redis or Memcached keyed by the repeating input.

**Worked example.** A streaming service scores hundreds of candidate titles per home-screen request on GPU. Because those candidates already form a natural batch, it micro-batches them into one forward pass rather than adding cross-user dynamic batching complexity, filling the sub-linear GPU cost curve for free. For a separate low-QPS CPU model behind a p50 SLA, it serves request-at-a-time, since the wait window would add latency for little fill gain at that load. It caches predictions for a small set of globally popular titles in Redis, but skips caching on the personalized per-user ranker where hit rate would be near zero. A Triton dynamic-batching window is reserved for the high-QPS shared embedding model, sized backward from the tail budget rather than for peak idle throughput.
