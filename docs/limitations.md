# Limitations (read before quoting any number)

- Each Phase 9 arm was run once (seed 42). Each fact group has 6 facts, so one fact is 16.7 percentage points. Small differences between arms may be noise; the large retention gap (62.5% vs 8% of earlier facts) is large enough to be informative.
- The facts are fictional and injected as trusted `experiment_fixture` facts. The experiments did not exercise live chat -> classifier -> approval end to end.
- The 100 general replay questions are hand-written and answered by the original model; they are not read from the SQLite store.
- The dev set under-estimated forgetting (1.7% on dev vs 7.6% on the exam for the chosen setting), so dev-based tuning was optimistic.
- The old-knowledge and reasoning sets are near ceiling for the original model (99.2% and 92.9%), so they are coarse.
- The leak guard blocks near-copies of exam questions (Jaccard >= 0.8) but allows similar wordings.
- Training GPU memory is not measured properly; the CPU-side RSS number undercounts and is labelled as such.
- Rollback is unit-tested but was never exercised on a promoted model, because no adapter passed the gate.
- Classifier and conflict detector are simple rules.
- All 15 study cycles would have failed the promotion gate; in every arm general knowledge eroded as cycles accumulated. Cycle 4 was bad in all arms and was not investigated.
- Per-cycle wall time varied a lot (about 10 to 34 minutes per cycle including the 220-question exam); the cause was not investigated.
