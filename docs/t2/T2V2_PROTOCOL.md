# T2-v2 common-control pairwise protocol

Versioned 2026-09-16 before T2-v2 outcomes. This supersedes neither nor
modifies the historical zero-action-prefix T2 experiment.

- Target and planner semantics are unchanged: +8 kg, friction .8, Kp/Kd
  25/.5, 50 Hz, physics dt .005 s, H=25, 472 candidates, 8 MPPI updates,
  64 elites, gamma=.99, weighted-elite-mean first-action execution.
- Two environments are created together and compared in local coordinates.
- The prefix begins immediately after reset. At every step one canonical MPPI
  action is computed from env-0 only and broadcast bitwise-identically to both
  environments. No zero-action stabilization action is used.
- Checkpoints are normal-control steps 10, 50, and 100 after reset.
- A branch is permitted only after a recorded `SYNC_PASS`; branches preserve
  the historical frozen rank-pair manifest and J25 definition.
- The historical T2 result remains `NOT_IDENTIFIABLE_WITH_N2_PARALLEL` under
  its zero-action-prefix protocol.
