# Implementing *The Case for Learned Index Structures* — an 8-Week Plan

Kraska, Beutel, Chi, Dean, Polyzotis — SIGMOD'18.
Working directory: `/home/nischal/Desktop/cs631/learned index/` (currently: the PDF only).
**Everything is built from scratch.** No existing code is reused.
**Two people**, ~12–18 hrs/week each, run as two parallel lanes.

---

## Context

### What the paper actually claims

One idea, applied three times. §2.2: **a range index is a model of the CDF** —
`pos = F(key) · N`. A B-Tree "learns" that CDF as a regression tree; anything else
that approximates the same function is also an index. From that:

| § | Index family | Classical | Learned replacement | Headline result |
|---|---|---|---|---|
| 3 | **Range** | B-Tree | Recursive Model Index (RMI) | 1.5–3× faster, up to 2 orders of magnitude smaller |
| 4 | **Point** | Hash map | `h(K) = F(K)·M` | up to 77% fewer conflicts |
| 5 | **Existence** | Bloom filter | classifier + overflow filter | 36% memory saving at 1% FPR |

§3 carries the weight; §4 and §5 reuse its machinery — which is why the RMI must be
built as a *reusable CDF model*, not as a monolithic index. Get that interface wrong
in Week 3 and Weeks 6–7 cost double.

The paper's own honest limits — no inserts/updates (Appendix D.2), no
multi-threading, no end-to-end training, GPU/TPU only speculative — are also the
boundary of this project. Read-only, single-threaded, CPU.

### The three ideas that make the RMI work, and where each can break

1. **Staging (§3.2).** Getting from 100M to ±100 with one model is hard; getting
   from 100M to ±10k, then ±10k to ±100, is easy. Stage 1 picks an expert, stage 2
   predicts. There is no search *between* stages — stage 1's output directly indexes
   stage 2. That is the whole trick.
2. **Per-model min/max error (§3.3, §3.4).** Each last-stage model stores the worst
   over- and under-prediction *over its own keys*, so lookup searches a bounded
   window. This is what turns "probably close" into **guaranteed zero false
   negatives** — and it is where sign errors and floating-point drift silently
   destroy correctness.
3. **Hybrid fallback (§3.3, Algorithm 1).** Any last-stage model whose max error
   exceeds a threshold is replaced by a B-Tree over its keys. This bounds worst-case
   performance to a B-Tree's: pathological data degenerates, never loses.

### Feasibility — the honest verdict

**Two people at 12–18 hrs/week each ≈ 160–230 effective hours** (coordination loss is
real; a pair is ~1.6×, not 2×). Estimated cost of the full scope after the Tier-1 cuts
below: **~110–160 hours**. So it fits, with genuine slack — but only by exploiting a
structural fact:

> **The B-Tree, the RMI, and the Bloom filter are mutually independent.** Only §4
> (learned hash) truly depends on §3. Everything else can run in parallel lanes.

Run this as two lanes, not eight sequential weeks. Sequential, this project does not
fit; parallel, it fits with a buffer week.

**Already cut** (no argument lost): the TensorFlow-slowness demo · FAST baseline
(third-party SIMD integration, and the paper concedes FAST is not significantly
faster) · string keys §3.5 (a whole separate tokenization + model pipeline, and the
paper's own speedups there are "not as prominent") · N=200M runs (100M exhibits every
effect; 200M only costs RAM and wall clock) · model-hash Bloom filters §5.1.2.

**Demoted to stretch:** neural-net stage-1 models. Multivariate linear regression with
automatic feature engineering is the *primary* path — it is what the paper's own
Figure-5 "learned index without overhead" configuration actually uses, it still fixes
the lognormal sigmoid problem, and it removes PyTorch, weight export, C++ NN inference,
and an expensive grid search from the critical path.

---

## The collaboration protocol

Six rules we set for working with AI agents on this project, made operational below.
Each week names one **dominant** rule matched to the character of that week's work;
Rule 5 (constrain format) applies to every prompt, and Rule 6 (post-mortem) closes
every week. "I"/"me" in the sample prompts refers to the agent.

| # | Rule | What it means in practice here |
|---|---|---|
| 1 | **Solve first, verify second** | You write the derivation/code, *then* hand it over. Never ask "how do I…" for something the paper already answers — ask "here's mine, break it." |
| 2 | **Make agents argue, not agree** | Prompt for attack, not review. "Find the flaw" beats "does this look right." Assign me a hostile role. |
| 3 | **Options, not answers** | For design forks, demand N alternatives with trade-offs and *no* recommendation. You choose. |
| 4 | **Reverse the roles** | Sometimes you're the reviewer and I'm the author. Sometimes you explain and I interrogate. |
| 5 | **Constrain the output format** | Tables, fixed schemas, word caps, "no code", "one sentence per row". Vague prompt → vague answer. |
| 6 | **Post-mortem** | End of week: what did I get wrong, what did you accept too fast, what would you prompt differently. |

**Standing weekly post-mortem** (Friday, 20 min), constrained to this format:

```
Prompt me with, verbatim:
"Week N post-mortem. Output exactly four sections, max 5 bullets each, no code:
 (1) Claims you made this week that turned out wrong.
 (2) Things I accepted from you without verifying — flag the risky ones.
 (3) Where I used you as an answer machine instead of a verifier.
 (4) One prompt I should have written differently, with the rewrite."
```

---

## The two lanes

| | **Lane A — systems / C++** | **Lane B — models / ML** |
|---|---|---|
| W1 | *joint: foundations* | *joint: foundations* |
| W2 | B-Tree baseline | RMI core |
| W3 | *cross-review*, then search strategies | *cross-review*, then stage-1 models |
| W4 | Hybrid indexes §3.3 + §3.4 | Stage-1 model quality, lognormal fix |
| W5 | Learned hash §4 | Bloom filter track begins |
| W6 | Figure-5 alt baselines | GRU + τ + overflow filter |
| W7 | *joint: integration + full evaluation (buffer)* | |
| W8 | *joint: report* | |

Assign lanes by taste, but **whoever does not build the RMI must still be able to
derive its error bounds** — that is the viva question.

---

## Week 1 — Foundations *(joint — do not split this)*

**Paper:** §1, §2 (all), §3.1.
**Ship:** repo + build, dataset generators, timing harness, correctness oracle, and the
Figure-4 table skeleton with plain binary search as the only row filled in.

Nothing is learned this week. You are building the instrument every later claim is
measured with — if it is wrong, every number in Weeks 2–8 is wrong too. Both of you
build it, because it is the contract the two lanes meet at.

**Problems to solve**

* **Interface design — the decision that makes or breaks the parallel structure.** §4
  needs the RMI as a *hash function*, §5 needs a model as a *classifier*, and Lane A
  needs to benchmark structures it did not write. Fix the interfaces in Week 1 and the
  lanes never block each other; get them wrong and every merge is a rewrite.
* **Timing methodology.** Independent lookups in a timing loop let the CPU overlap cache
  misses, so you measure *throughput under memory-level parallelism*, not single-lookup
  latency. Both defensible; different numbers. Pick one, state it, apply it identically
  to every structure.
* **The real datasets don't exist publicly.** Weblogs (200M university web-log
  timestamps) and Maps (200M OSM longitudes) are Google-internal. Find substitutes that
  preserve the *claim*, not just the size — SOSD's `wiki_ts` for Weblogs (timestamps,
  nasty temporal structure from class schedules, holidays, semester breaks: the paper's
  stated worst case), `osm_cellids` for Maps. Say so in the report.
* **Generating 190M unique lognormal(μ=0, σ=2) keys scaled to integers up to 1e9**
  without a 1.8 GB draw-sort-dedupe pass. (A presence bitset over the value range is
  ~125 MB.) Also build `dense` (`0..N-1`) as the sanity dataset — a perfectly linear CDF
  that any correct model must fit exactly.
* **Harness discipline, fixed now and never relaxed:** fixed seeds; 1M lookups sampled
  with replacement then shuffled (kills sequential locality); warm-up discarded; median
  of 5 reps; `volatile` sink; build/train time never inside a timed loop.
* **Correctness oracle before any timing:** every key at its exact position; sampled
  absent keys report not-found; the run aborts on failure.
* N ∈ {10M, 100M}. Check the RAM budget for 100M now, not in Week 2.

**Agent protocol — Rule 3 (options, not answers), then Rule 2**

The interface is a real fork. Don't ask me to design it:

> "Give me exactly 3 ways to structure a trained CDF model in C++ so the same object
> serves as (a) a range index, (b) a hash function `F(K)·M`, and (c) a classifier
> scorer. Format: one table, columns = design | how §4 reuses it | how §5 reuses it |
> what it costs at inference | the thing that breaks later. ≤ 15 words per cell. No
> code, no recommendation."

Pick one between the two of you, then switch me to attack mode:

> "We chose X. Give me the 3 strongest objections in order of severity, one sentence
> each, then the single condition under which we should reverse the decision."

---

## Week 2 — Split: B-Tree baseline ∥ RMI core

The two hardest single pieces, built simultaneously against the Week-1 harness.

### Lane A — the B-Tree baseline

**Paper:** §3.7.1, §2.1.
**Ship:** B-Tree rows of Figure 4 — size MB, `lookup_ns`, `traverse_ns`, model %, page
sizes 32/64/128/256/512, all datasets × N.

Every result in this paper is a ratio against **B-Tree page=128**. A weak baseline
invalidates the whole project and is the easiest way to accidentally cheat. Build it as
if you were trying to beat the RMI.

* **Should nodes store child pointers?** With fixed fanout and 100% fill,
  `child(i,j) = i·P + j` — the layout is implicit and no pointers are needed. That makes
  the baseline *smaller and faster*, the conservative choice on both axes. Storing
  pointers would flatter your RMI. Do the hard thing.
* Read-only, 100% fill, built bottom-up. The leaf level should *be* the sorted array —
  no copy — and the array is excluded from both structures' reported size.
* 64-byte-aligned dense key arrays per level; the same branch-free binary search
  primitive Lane B uses, so no measured difference comes from the search code.
* **Verify page=128 is actually optimal on your machine.** The paper picks it because it
  wins for them. If 64 wins on your CPU, say so and report both reference points.
* **Redo §2.1's budget calculation for your cache hierarchy.** A page traversal ≈ 50
  cycles; a CPU does 8–16 SIMD ops/cycle; so a model gets ~400 arithmetic operations to
  beat a 1/100 precision gain per node. This tells Lane B how complex a stage-1 model
  they can afford — hand them the number.

### Lane B — the RMI core

**Paper:** §3.2, §3.4 (bounds), §3.6.
**Ship:** 2-stage RMI, `K ∈ {10k, 50k, 100k, 200k}` (the paper's exact second-stage
sizes), 100% recall at every config.

* **Derive the search-bound sign convention yourself, from scratch, before writing
  code.** If `error = predicted − actual`, then `actual = predicted − error`, so
  `actual ∈ [pred − max_err, pred − min_err]` — it *subtracts*, and the bounds swap
  roles. Define the convention once, in a comment, and make every call site obey it.
  Getting it backwards produces false negatives *only on inexact models*, so it passes
  on `dense` and fails silently everywhere else. Highest-value hour of the project.
* **Floating-point determinism is load-bearing.** GCC defaults to `-ffp-contract=fast`
  and may fuse `slope*key + intercept` into an FMA at one call site but not another.
  Training then computes a different prediction than lookup, a stored bound is off by
  one, and a key vanishes. Compile with `-ffp-contract=off`, in the build file, with a
  comment saying why.
* **Precision.** Keys reach 2^63 and N reaches 10^8; the uncentred normal equations lose
  too much precision. Centred form, `long double` accumulators.
* **Memory feasibility.** Stage 1 is monotone in the key and the input is sorted, so each
  stage-2 model owns a *contiguous run* — partition in one pass storing run boundaries
  rather than materialising a key vector per model.
* **Degenerate models.** An empty bucket is reachable only by absent keys — decide what it
  returns. A single-key bucket has `var_x = 0`, so the closed form degenerates to
  `slope=0, intercept=pos`, which is exactly right. Convince yourself rather than
  branching.
* Closed-form least squares, single pass (§3.6) — no gradient descent in stage 2.

**Agent protocol — Rule 1 (solve first, verify second), both lanes**

Write the derivation and the code before you talk to me. Then:

> "Here is my min/max-error derivation and my `search_bounds()` [paste]. Do not explain
> the correct answer and do not rewrite my code. Instead: construct the smallest concrete
> dataset and lookup key on which my version returns a false negative — or state 'no
> counterexample found' and give the argument why. ≤ 150 words."

Asking me to *produce a counterexample* rather than *check your work* is the whole
difference between verification and outsourcing.

---

## Week 3 — Cross-review, integrate, then diverge again

**Ship:** the first real RMI-vs-B-Tree speedup and size table.

**Rule 4 comes free this week — you have a human to reverse roles with.** Spend the
first two sessions with each of you attacking the *other's* structure, using the hostile
prompt below on code you did not write. This catches more than I will, because you each
have to actually understand the other half.

> "You are a SIGMOD reviewer who believes learned indexes only beat B-Trees because the
> baselines are strawmen. Here is our B-Tree [paste]. List every way it is weaker than a
> production B-Tree, ranked by how much it inflates our speedup. Table: weakness | est.
> effect on our numbers | cost to fix. ≤ 20 words per cell. Do not compliment anything."

Then integrate and split again: **Lane A** → model-biased binary search, exponential
search, biased quaternary search (three probes at `pos−σ, pos, pos+σ` so the hardware
prefetches all three). **Lane B** → stage-1 model quality.

---

## Week 4 — Hybrid indexes ∥ stage-1 model quality

### Lane A — hybrid indexes and correct range semantics

**Paper:** §3.3 (Algorithm 1), §3.4.

* **Algorithm 1, faithfully.** Train stage-wise; then replace any last-stage model with
  `max_abs_error > threshold` by a B-Tree over its keys (thresholds 128 and 64). Confirm
  the guarantee: on adversarial data everything degenerates to a B-Tree and you are never
  *worse*. This is one of the paper's best ideas and it is cheap once both structures exist.
* **The monotonicity trap (§3.4) — the subtlest correctness problem in the paper.** Error
  bounds guarantee you find every *existing* key. For a **non-existent** key, a
  non-monotone model can return the wrong upper/lower bound. Range queries need
  `lower_bound`, not just point lookup. Fix: detect that the found bound sits on the
  boundary of the search area and incrementally widen.
* **Expect a negative result and report it.** The paper states plainly that on integer
  datasets, hybrid models and non-binary search "did not provide significant benefit" —
  the gains show up on strings. Reproducing the null result *is* success. Do not tune
  until it wins.

**Agent protocol — Rule 4:**

> "I'm going to explain how our RMI handles `lower_bound` for a key that isn't in the
> dataset. Ask me one question at a time, hardest first, until you find a case I can't
> answer. Do not answer for me, do not suggest fixes, do not move on until I've responded."

### Lane B — stage-1 model quality

**Paper:** §3.1, §3.7.1.

* **Work out why a linear stage 1 fails on lognormal.** The log of a lognormal is
  *normal*, so the CDF in log-space is a sigmoid. A line compresses the sigmoid's dense
  middle and the stage-2 models landing there inherit enormous key counts. Raising K
  subdivides the ordinary buckets but barely touches the worst one. Structural, not a bug.
* **Primary path: multivariate linear regression** with automatic feature engineering over
  `key, log(key), key², √key` — the paper's Figure-5 "learned index without overhead"
  configuration, which fits nonlinear patterns in very few operations.
* *Stretch only if Lane B is ahead:* a small ReLU net for stage 1, trained in PyTorch,
  weights exported to JSON, forward pass hand-written in C++ — **never call the framework
  at inference** (§3.1). Cite §2.3's ≈80,000 ns TensorFlow figure as motivation; do not
  re-measure it.
* Grid search over (stage-1 feature set, K). Expect to confirm: richer stage 1, strictly
  **linear** stage 2 — "for the last mile it is often not worthwhile to execute complex
  models."

**Agent protocol — Rule 3 + Rule 5:**

> "Propose 6 candidate stage-1 model families for an RMI over lognormal keys. One table:
> family | #params | est. ns/prediction | can it represent a sigmoid (Y/N) | training cost
> | one-line failure mode. No code. No recommendation. Do not tell me which to pick."

---

## Week 5 — Learned hash ∥ Bloom filter begins · **cut checkpoint**

### Lane A — the hash-model index (§4)

The cheapest deliverable in the project; it reuses Weeks 2–4 wholesale.

* `h(K) = F(K)·M`, F the learned CDF, M = N slots. Paper's config: 2-stage RMI, 100k
  second-stage models, no hidden layers. Baseline: MurmurHash3-like.
* Measure conflict rate and slot-occupancy distribution on all three datasets. Target: up
  to 77% conflict reduction.
* **Understand when a learned hash is worthless.** On uniform keys a linear model learns
  the CDF perfectly — and is then no better than any decent randomising hash. The win
  requires *skew*. Make sure your datasets can show both outcomes; a result that only ever
  wins is a result you don't understand.
* **Do fewer conflicts mean faster lookups?** Wire it into a real separate-chaining map and
  measure end-to-end, model execution included. The paper is explicit that the benefit
  depends on payload size and conflict policy — for small keys and values, Cuckoo hashing
  probably wins anyway.

### Lane B — Bloom filter track begins

Classical Bloom filter (m bits, k hashes, tuned for 1% and 0.1% FPR) **and** the data
work, which is the part that actually eats time: the paper uses Google's Transparency
Report (1.7M phishing URLs); use PhishTank / OpenPhish. Build the negative set as *both*
random valid URLs and whitelisted lookalikes — the paper gets 60% vs 21% savings on those
two, and that spread *is* the covariate-shift finding.

### ⚠ Cut checkpoint — hold it this week, not in Week 8

If either lane is more than one week behind, cut in this order: Figure-5 alternative
baselines → the NN stretch → §5.1.2 → hybrid indexes. **Never cut:** the B-Tree
baseline's quality, per-model error bounds, or zero-false-negative verification.

---

## Week 6 — Figure-5 baselines ∥ the learned Bloom filter

### Lane A — alternative baselines (§3.7.1, Figure 5)

What separates a reproduction from a demo.

* 3-stage lookup table: every 64th key, padded to a multiple of 64, repeated once more;
  binary search the top table, then branch-free AVX scan.
* Fixed-height B-Tree sized to ~1.5 MB (matching your model) + **interpolation search**.
* Histogram: the paper dismisses it in prose — an accurate CDF needs so many buckets that
  searching the histogram becomes the problem, and fixing that yields a B-Tree. Reproduce
  the *argument*, not the code.

### Lane B — the learned Bloom filter (§5.1)

* **Derive the threshold arithmetic yourself.** `FPR_O = FPR_τ + (1 − FPR_τ)·FPR_B`. The
  paper sets `FPR_τ = FPR_B = p*/2` so `FPR_O ≤ p*`. Work out why that suffices, and what
  an uneven split would buy.
* **The zero-false-negative contract.** Build the overflow filter over
  `K⁻_τ = {x ∈ K : f(x) < τ}` — *every* key the model scores below threshold. Miss one and
  the structure is unsound. Test over the entire key set, not a sample.
* **Count the model in the memory budget.** The paper's 16-dim GRU with 32-dim character
  embeddings is 0.0259 MB and *is* included in its 1.31 MB total against a 2.04 MB Bloom
  filter. Excluding it would be cheating.
* **The shape of the trade-off:** tightening τ lowers FPR but raises FNR, and the overflow
  filter scales with FNR. Paper: 1% FPR → 55% FNR → 36% saving; 0.1% FPR → 76% FNR → 15%.
  The saving shrinks as you demand accuracy.

**Agent protocol — Rule 1, then Rule 2:**

> "Here is my derivation of τ and my overflow-filter construction [paste]. Attack it in two
> passes. Pass 1: find a scenario where the structure returns a false negative. Pass 2:
> find an accounting error that makes my memory saving look better than it is. ≤ 100 words
> per pass. If you find nothing, say 'nothing found' — do not invent weak objections to
> seem useful."

That last clause matters: an agent told to attack will manufacture noise unless you
explicitly permit it to find nothing.

---

## Week 7 — Integration and full evaluation *(joint — this is the buffer)*

**Ship:** every structure, every dataset, N ∈ {10M, 100M}, one reproducible run.

This week exists to absorb slippage. If both lanes are on time, you get a full evaluation
sweep and a week of margin on the report. If they are not, this is where you catch up —
which is why the cut checkpoint is in Week 5 and not here.

* One binary, one set of flags, one lookup sample per (dataset, N) group shared by every
  index in it.
* Regenerate every CSV from fixed seeds. Any number in the report that cannot be
  regenerated by a command does not go in the report.
* Reconcile the two lanes' size accounting — it is very easy for Lane A and Lane B to have
  been counting different things all along.

---

## Week 8 — Report *(joint)*

* Structure it as **reproduced / diverged / not attempted**. Where your numbers disagree
  with the paper, the explanation is the contribution — write divergences as findings, not
  apologies.
* State the scope honestly: read-only, single-threaded, CPU, no inserts (Appendix D.2), N
  capped at 100M, public dataset substitutes for Weblogs and Maps.
* Name what you cut and why. A defended cut reads better than a silent gap.

**Agent protocol — Rule 2, then Rule 6:**

> "You are Reviewer 2 on this as a SIGMOD submission. Output exactly: 3 reasons to reject,
> ranked; 3 claims we have not supported with evidence; 1 experiment whose absence is most
> damaging. One sentence each. No praise, no summary of our work."

> "Full 8-week post-mortem. Four sections, ≤ 6 bullets each, no code: (1) where we used you
> as an answer machine and it cost us understanding; (2) claims you made that we accepted
> and shouldn't have; (3) the weeks where solve-first actually changed our result; (4) the
> 3 prompts that produced the most value, and why."

---

## Cut order — decide at the Week 5 checkpoint

1. Figure-5 alternative baselines (Lane A, Week 6)
2. The neural-net stretch in Week 4
3. §5.1.2 model-hash Bloom filters
4. Hybrid indexes §3.3 — cut last; the worst-case guarantee is one of the paper's best ideas

**Never cut:** the quality of the B-Tree baseline, per-model error bounds, or the
zero-false-negative verification.

**Already cut, for the record:** TensorFlow slowness demo · FAST · string keys §3.5 ·
N=200M runs.

## Standing invariants

1. **Zero false negatives is a hard failure**, every week, for every structure — never
   a warning.
2. Every learned-vs-classical number comes from the same binary, the same flags, and
   the same lookup sample.
3. Training/build time is never inside a timed loop; report it separately.
4. Fixed seeds everywhere.
5. Compile with `-ffp-contract=off`.

## Verification

* **Per structure:** all N keys found at their exact position; ≥1000 sampled absent keys
  correctly report not-found; both run *before* any timing, aborting the run on failure.
* **RMI-specific:** mean error width shrinks monotonically as K grows; on `dense` the fit
  is exact (max error width 0); no single stage-2 model holds >50% of the keys.
* **Bloom filter:** FNR measured over the *entire* key set is exactly 0.
* **Hash index:** conflict count reconciles with the slot-occupancy histogram.
* **End-to-end:** `make && ./benchmark` reproduces every CSV in the repo from fixed
  seeds; the report's tables are generated from those CSVs, never transcribed by hand.
