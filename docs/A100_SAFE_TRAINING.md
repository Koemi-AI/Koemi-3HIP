# Conservative A100 training run

This is the small, code-focused execution path for the remaining A100 budget.
It is a Python module, not a notebook containing a giant embedded source or
JSON payload. The old notebook remains available as a legacy path; this runner
is the safer path for a paid session.

## Budget and target

At US$ 6.33 per hour, 188 hours represent US$ 1,190.04. The default plan uses
25 sessions of 7.5 hours, totaling 187.5 hours and leaving 0.5 hour as a
reserve. The ledger reserves a session before remote work and records actual
elapsed hours after completion or failure. A killed Colab session leaves its
reservation in place deliberately; check the ledger before reusing it.

The model is the existing approximately 0.205B-parameter configuration:
embedding 512, 128 fixed experts, 6 active experts, and BF16 autocast on a
compatible A100. This is a supervised byte-level code assistant experiment,
not a claim of agentic benchmark quality. The bounded default corpus target is
45,000 records, mostly code, with a small verified-math slice.

## Three-stage run

Install the repository in a fresh Colab runtime and use a private Drive
directory for this run:

```bash
python -m pip install -e .
python -m koemi.training.a100_safe_run --mode plan \
  --results-dir /content/drive/MyDrive/koemi-a100-safe-v1
```

`plan` is local-only: it does not initialize CUDA, contact the Hub, or reserve
budget. It must print 25 sessions, 45,000 records, and US$ 1,190.04 before
continuing.

Run the real device probe next:

```bash
python -m koemi.training.a100_safe_run --mode preflight \
  --results-dir /content/drive/MyDrive/koemi-a100-safe-v1
```

This performs internal contracts plus one `2 x 32` BF16 forward/backward and
optimizer step, then writes `preflight_report.json`. It does not download the
training corpus. Stop if it does not report an A100, finite loss, finite
gradient, and a completed optimizer step.

Only after that report is healthy, start the paid session:

```bash
python -m koemi.training.a100_safe_run --mode train \
  --results-dir /content/drive/MyDrive/koemi-a100-safe-v1 \
  --confirm-budget-hours 188
```

The exact confirmation is intentional. Omitting it fails before dataset access
or budget reservation. Each session writes the plan, environment, contract
results, corpus manifest, dataset report, checkpoints, final report, and
`safe_budget_state.json`.

The data sources remain pinned and validated by `a100_run.py`. The reduced
quotas are 10,000 OpenCode priority, 15,000 OpenCode general, 10,000
CodeFeedback, 8,000 Magicoder, and 2,000 OpenR1 Math records. The runner keeps
prefix caches out of training and does not treat semantically similar prompts
as identical state.

## What this does not claim

The plan does not claim that 188 hours guarantees a useful coding agent. The
actual result depends on data quality, optimizer stability, A100 availability,
and measured throughput. Colab paid compute availability and runtime limits are
variable; consult the [official Colab FAQ](https://research.google.com/colaboratory/faq.html)
before committing the full budget. The [A100 specification](https://www.nvidia.com/en-us/data-center/a100/)
supports BF16 and 80 GB memory, but that is a hardware capability, not a
measured Koemi throughput result.
