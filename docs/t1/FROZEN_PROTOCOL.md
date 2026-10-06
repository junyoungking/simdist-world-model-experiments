# T1 fixed data-amount / buffer-composition control — frozen protocol

This continuation tests only +4 kg → +8 kg. A valid sample is one post-processing eligible H=T=25 window, not a control step or optimizer occurrence.

For each of five repeats, `B_core` is 500 +8 kg windows, `B_extra` is a disjoint 500 +8 kg windows, and `A_hist` is 500 +4 kg windows. `CURRENT_ONLY_1000 = B_core ∪ B_extra`; `MIXED_500_500 = B_core ∪ A_hist`. The two arms start independently from the same M_pre (`go2_2026-02-12@367004`) and use K=1,000 relative f-only optimizer updates. E/C/R/V/π are frozen.

The released trainer's internal split would otherwise remove windows from optimizer eligibility. This protocol sets `training_data_ratio=1.0` for an explicit 1,000-window training manifest and uses a completely separate pre-existing +8 kg `role=eval` B_eval artifact. With batch size 32, each epoch has 31 full batches and drops 8 rows; this is held identical across arms and logged as expected sampling exposure.

No subset is selected using adaptation/evaluation outcomes. Existing clean E1 +4 and B_eval artifacts are reused read-only; an added +8 kg supplement is collected only where the existing B collection has fewer than 1,000 eligible windows.
