# Current limitations and blockers

- The exact pinned Hugging Face remote-code files and checkpoint cache were not inspected on the server. G0 cannot pass on configuration claims alone.
- The server checkout contains uncommitted edits in four baseline modules. They were preserved; subsequent SSH inspection timed out before their content or server-side data/cache could be reviewed.
- The latest read-only SSH retry on 2026-10-10 at 07:19 UTC timed out connecting to `172.20.27.4:22`. No server files were changed and no server-side command was run after that timeout.
- The root filesystem has only 1.4 GiB free. The separate `/home/new_drive` volume has 439 GiB free; any isolated worktree and outputs should use that volume.
- No GPU forward, checkpoint parity, no-op replay, 3-prompt smoke, or 50-prompt pilot result is available in this report.
- The v4 command namespace and fail-closed gate handling exist, but the runtime audit does not yet invoke the implemented native/patched fidelity, legal-K, legacy reproduction, replay, and synchronized profile routines. Its corresponding manifest gates therefore remain `not_run`. Smoke/pilot and later-stage execution adapters are also not enabled; they return blocked manifests instead of starting branches. This package is not yet ready to resume a GPU pilot merely by reconnecting the server.
- Attention Q/K extraction remains disabled until post-QK-normalization/post-RoPE tensors and short-sequence proxy fidelity are verified. Raw projection hooks are rejected.
- Community, predictor, allocation, and profiling utilities are infrastructure only. They establish no H1-H4 result and do not justify opening later-stage GPU workloads.
- Full-trajectory executed-pair budgets can diverge when the treatment changes commits/NFE. A local one-forward equal-cost exchange is not a trajectory-level matched-budget comparison.

Local verification on 2026-10-10: 17 CPU tests passed; one pinned-checkpoint GPU integration test was skipped because it requires explicit opt-in and cached model weights. Python compilation, `jrc-moe jrc-v4 --help`, and `git diff --check` passed. No GPU or integration result is implied by these checks.
