<!--
Follow this structure (it mirrors what gets merged upstream). Keep PRs focused
and single-concern. Be honest in the Test plan about what you could NOT run.
-->

## Summary

<!-- Bullets: what changed and WHY. Link the issue/discussion that prompted it
     ("Following @maintainer's suggestion in #NNNN", "requested in #NNNN"). -->

-

## Context

<!-- For fixes: the precise root-cause mechanism. Paste the literal error/output
     if relevant. Delete this section for pure feature PRs. -->

## Changes

<!-- File-level list. -->
- **New:**
- **Modified:**

## Test plan

<!-- Checkboxes. State what validates correctness/perf. If you could not run
     something (e.g. no SM121a hardware), SAY SO explicitly and invite review. -->
- [ ] Numerical correctness vs reference (PyTorch golden / HF), tolerance stated
- [ ] `ptxas --gpu-name sm_121a` assembles cleanly, no spills (`scripts/ptx_syntax_check.sh`)
- [ ] `compute-sanitizer --tool memcheck` clean on touched kernels
- [ ] `cargo test --release -- --test-threads=1` passes (or: which subset, and why)
- [ ] No breaking API changes / fallback preserved / behavior-change is opt-in

<!-- Fixes #NNNN / Closes #NNNN / Ref: #NNNN -->
