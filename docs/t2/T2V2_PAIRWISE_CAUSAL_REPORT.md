# T2-v2 pairwise causal report

## Result

T2-v2 intentionally replaced the historical zero-action prefix with a normal
common-control prefix. The old T2 result is unchanged.

All four frozen T2-v2 smoke cells (CUMULATIVE/RECENT × seeds 13302/13303)
used a bitwise-identical env-0 canonical-MPPI action history for both N=2
members. Nevertheless, reset state equality failed in every cell: largest
mandatory-field max-absolute errors were 2.12445 (seed 13302) and 3.25801
(seed 13303), above the predeclared 1e-5 tolerance. One member subsequently
terminated before t=10 in every cell.

No checkpoint was `SYNC_PASS`; all t=10/50/100 records are
`PRECHECKPOINT_TERMINATION`. No candidate pair branch was eligible or run.

## Conclusions

- Reset synchronization: failed.
- Normal identical common control preserving synchronization: not established.
- Pairwise ranking fidelity, CUM-vs-REC differences, realized J25, and
  dynamics-error attribution: unidentified.
- Evidence about frozen R/V: **no**; no isolating intervention occurred.
