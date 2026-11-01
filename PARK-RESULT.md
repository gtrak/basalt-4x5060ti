# Parking: measured, and the RAM budget that makes it free

Parking works. The cost was never the park write - it was the RAM it stole from the
expert copy. Size the park budget to the room the box has and the spill goes to zero.

## The park write is cheap

Proportional to the tokens a conversation used, NOT to the context capacity. The
`6.0 GiB per session` figure in the brief was worst-case-at-262144 and never happens:

    parked   1381 tokens in  31.9 ms
    parked   1891 tokens in  36.5 ms
    parked 125406 tokens in 129.1 ms

A full 125k conversation costs 129 ms.

## The spill was a RAM-budget collision, not a parking cost

`--conversation-cache-mib` is ADDED to serve growth (src/program/generate.cpp:745), which
raises the headroom the resident copy must leave free. With `--conversation-cache-mib 24000`:

    --resident-budget-gib 37.00 is more than the RAM has room for
    (54.40 GiB available minus 30 GiB headroom and the page overhead): 24.48 GiB

The expert plan wanted 37 GiB and got 24.48, so ~12.5 GiB of experts spilled (19.5 GiB of
SSD traffic observed). Note the engine REFUSES a budget RAM cannot hold rather than
shrinking it ("a budget the RAM has no room for is refused, not shrunk"), so it fell back
to the what-fits plan.

To hold 37 GiB of experts AND 23.4 GiB of parks you need >= 65.7 GiB available; this box
has 54.40. The budget has to come from somewhere.

## The fix: 8000 MiB

| | park 24000 | **park 8000** |
|---|---|---|
| SSD spill | 19.5 GiB | **0.0 GiB** |
| decode (tg=1800, 3 samples) | 96.3 / 105.2 / 96.8 t/s | **106.2 / 107.0 / 105.1 t/s** |
| experts in RAM | 24.48 (capped) | **30.85 GiB** |
| RAM refusal warning | yes | none |

8 GiB of park budget leaves 8.03 GiB of headroom and 46.38 GiB for the experts, well
above the 37 GiB plan. Decode came out ~10% FASTER, because the experts stopped spilling -
that gain outweighs what the smaller park budget gives up. The measured parks are 1.4k
tokens at 32 ms, so 8 GiB still holds several conversations.

Boot:

    --resident-budget-gib 37 --conversation-cache-mib 8000 --conversation-cache-slots 4

## evictions = 0

Across every run, `evictions=0`. Nothing was evicted, so a park-on-eviction policy would
have written nothing at all in this workload. That is the answer to the question the
park/evict brief was gated on: the writes today are already few and small, and moving the
write to eviction has nothing to act on here. The queue half of that brief (admit-queue
rather than decline) is a separate change and still worth having.
