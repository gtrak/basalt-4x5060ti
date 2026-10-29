# Decode next steps (parked 2026-10-29, prefill has the floor tonight)

State: 25.3-25.9 ms/window solo synthetic, 26.8-31.7 solo real, 41-43 ms
concurrent (2 sessions), 71.9-81.3 t/s.  Production split: 35.4-39.8 ms.

1. **Concurrent pair-sync (the big one).** GPU-reach wait 9.3-9.6 solo ->
   12.6-13.8 with two sessions.  The o_proj combine is a per-layer barrier:
   every layer, both ranks push partials and spin (`tp2_all_reduce`), and at
   2x rows the spin holds SMs longer.  Candidates, cheapest first:
   a. Batch the two slots' rows into ONE combine exchange per layer (the
      exchanges are per-window today; two windows = two exchanges where one
      would do - halves the exchange count under concurrency).
   b. Overlap the combine with the down_proj tail: the partial for expert
      group g is ready before group g+1 finishes; a split-phase reduce
      (reduce-scatter style, per row-block) hides most of the wait.
   c. Replace the SM spin with the tier's mapped-host doorbell pattern
      (wait on the host word, not on SMs) so a waiting rank does not burn
      the SMs the peer's push kernel needs on a PHB-shared box.
2. **L2 warm-up predictor coverage.** "the peer's L2 warm-up is off (a router
   the predictor does not cover)" - the tp2 router's rows never reach the
   warm-up path (verify.cpp:685).  Covering it warms the partner's L2 for the
   routed experts; on a PHB box L2 hit rate is worth real bandwidth.
3. **Adaptive continuation boundary.** The lead/partner slice is static
   (lead: hottest until room).  The adaptation machinery (servable/swaps)
   could rebalance the boundary from observed lookup histograms - the profile
   is a prior, the workload is the truth.  Careful: partner experts are not
   servable today by design (M1 note); this would need servable-partner rows.
4. **Two-slot decode interleaving on the pair.** Slot A's expert layers and
   slot B's attention halves could interleave on the partner card (its
   attention half for A idles while A's experts compute).  Finer-grained
   stream scheduling; only worth it if (1) does not close the concurrent gap.
5. **Chunk sweep + short-read.** --prefill 12288 vs 24576 under tp2 (the
   per-chunk peer handshake is new fixed cost); --short-read 512 is set and
   needs real-traffic validation (the aligned small reads).
6. **The prompt-cache snap.** Appends re-read ~12k tokens (110592 of 122534
   reused) - the cache granularity x append interaction costs 11-15 s TTFT at
   the routed prefill rate.  Not tp2-specific but tp2 amplifies it 3.6x.
