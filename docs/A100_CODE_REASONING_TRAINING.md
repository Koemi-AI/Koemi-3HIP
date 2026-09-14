# A100 code and reasoning training design

This is the execution contract for the A100 Colab run. It replaces the
five-hour T4-oriented corpus with a pinned English programming and verified
mathematics mixture. It is a training experiment, not a claim of general
intelligence or a promise that every Colab session will remain available.

## Decision

Use the current HERM architecture with width 512, 128 deterministic experts,
six active experts per valid byte, `scan_chunk=128`, and `ablation=no_refine`.
That is 204,816,147 parameters. It is large enough to make the A100 useful
without presenting a 29-hour, from-scratch run as evidence for a useful 1B
parameter model.

The notebook requires an A100-class GPU with at least 70 GiB visible memory,
compute capability 8.0 or later, CUDA BF16 support, and an 80 GiB device in
the run report. BF16 autocast and TF32 matrix math are enabled. The existing
parallel HERM execution is used. Hugging Face Accelerate adds no speed to this
one-GPU workload, and `torch.compile` is not enabled automatically: the
dynamic expert loop has not passed a CUDA equivalence and throughput gate.

## Corpus

The notebook pins each dataset to an immutable Hub revision and rejects every
row that violates its source-specific contract. It never runs code or unit
tests embedded in a training row.

| Portion | Dataset and revision | Selection rule | Target records |
| --- | --- | --- | ---: |
| Code, debugging, terminal and VCS vocabulary | `nvidia/OpenCodeInstruct` at `8f3ba5bafe4d6e8db46082cf7ae6741bc370604d` | English input/output; parsed test score exactly 1.0; every recorded test status is `pass`; priority terms include Python, TypeScript, Bash, PowerShell, Git, shell, CLI, error and debugging | 180,000 |
| Code revision and feedback | `m-a-p/CodeFeedback-Filtered-Instruction` at `a08c213a9748c66c15d0225814be80a2e77adf4a` | English `query` and `answer`, bounded UTF-8 spans | 80,000 |
| Code instructions | `ise-uiuc/Magicoder-Evol-Instruct-110K` at `b0079beaa0361d82412520b873715bee59cc7dd4` | English `instruction` and `response`, bounded UTF-8 spans | 70,000 |
| Explicit mathematics | `open-r1/OpenR1-Math-220k`, `default` config, at `e4e141ec9dea9f8326f4d347be56105859b2bd68` | English problem and answer; one generation that is both Math-Verify-correct and marked reasoning-complete | 35,000 |

The target is 365,000 records. Each source has a scan ceiling. Falling short
raises an error with the source name, target, scanned rows and rejection
counts; it does not silently change the mixture. The corpus is serialized to
Drive atomically with a manifest containing revisions, counts and SHA-256. A
later session reloads precisely that corpus rather than streaming a changed
dataset.

OpenCodeInstruct is CC-BY-4.0. The other three sources are Apache-2.0 at the
pinned revisions above. Their cards are the license and schema authority:

- https://huggingface.co/datasets/nvidia/OpenCodeInstruct
- https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction
- https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K
- https://huggingface.co/datasets/open-r1/OpenR1-Math-220k

Terminal-Bench and BigCodeBench remain evaluation-only. Terminal-Bench's
published dataset explicitly says benchmark data must never enter training
corpora, and its archives contain executable environments. This raw byte model
also has no tool loop or sandboxed agent evaluator, so the A100 notebook does
not claim a Terminal-Bench score.

## Reasoning supervision

Each accepted OpenR1 row has this exact shape:

```text
<|input|>
problem
<|thinking|>
verified explicit derivation
<|output|>
reference final answer
```

The thinking span comes only from a complete, Math-Verify-correct generation.
The final answer remains a separate target, so validation can report answer
and reasoning losses independently. The thinking loss weight is 0.5: long
proofs influence training without outweighing code fixes and final answers.

This is visible process supervision, not a hidden chain of thought and not a
mechanism that establishes cognition. Process supervision can improve a math
solver's reliability when the steps themselves are checked, but it does not
turn a small from-scratch byte model into a general reasoning system.
https://openai.com/index/improving-mathematical-reasoning-with-process-supervision/

## Resumption and measurements

Every optimizer boundary carries a deterministic epoch permutation, next batch
index, model, optimizer, scheduler, scaler, Python and Torch RNG states. Two
rotating Drive slots are written through a temporary file and atomically
replaced. The loader accepts only a checkpoint with the exact corpus and run
signature; a corrupt newest slot falls back to the prior valid slot and records
the reason.

The notebook benchmarks the real data batch sizes before training, selects the
highest measured supervised-bytes/s candidate below the VRAM safety limit, and
stores that choice in the signed manifest. It then saves Drive checkpoints at
most 15 minutes apart, ends each Colab session with a checkpoint, and reports:

- loss, answer loss and thinking loss with separate denominators;
- supervised bytes/s, batch size, accumulation, BF16/TF32 state and peak VRAM;
- non-finite gradients, expert occupancy, entropy and Gini;
- data selection/rejection counts and the exact source revisions;
- validation by the same frozen record-level split on every resumed session.

The first cell runs mocked schema, lazy-dataset equivalence and rotating
checkpoint recovery tests before it builds the full model or opens a remote
dataset. The A100 preflight then validates the actual GPU and parallel/sequential
HERM agreement on a small real-device batch. A failed check stops the run before
long training begins.

## Boundaries

- The notebook can reduce the chance of losing work; it cannot guarantee an
  error-free provider session or recover a VM termination that happens between
  checkpoints.
- A 29-hour credit balance is not a continuous runtime entitlement. The
  notebook is designed for repeated bounded Colab sessions.
- It does not execute training data, evaluate external benchmark archives,
  add a CUDA kernel, use multi-GPU training, or represent an unmeasured
  throughput result as a quality result.
