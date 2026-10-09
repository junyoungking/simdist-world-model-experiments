# Raw-derived portfolio evidence

Five measurement tables are regenerated from saved seed/episode/state measurements. `chain_equality_test.csv` is a NOT_REPRODUCED status record with an empty p-value. No replacement statistical test was performed.

The six historical CSV headers and row counts are preserved. `source_report_section` now points to raw-derived provenance, rather than report transcription.

Section4 loss: independent seeds49301–49310, arithmetic seed mean. Section7 loss: target common held-out seeds186001–186020, arithmetic seed mean; mass reduction32.299027–46.356004%. Control evaluations use 23501–23600 plus24001–24100.

MC continuation is175 steps for source/mass and200 for friction/gain. Head and critic share identical within-target pairs; cross-target MC returns are not directly compared.

See [PROVENANCE.json](PROVENANCE.json) for exact values, per-model counts, checkpoint IDs, source selections, SHA256 and per-file publication status; [method documentation](../../analysis/PORTFOLIO_RAW_PROVENANCE.md) for schemas and controls. The full per-file input inventory is local-only and excluded from Git.
