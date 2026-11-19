# GTA VI trophy rarity forecast — week one

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jeffchen-lang/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jeffchen-lang/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Built for the [make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01): forecast each of GTA VI's 55 PS5 trophies' Exophase earn rate at 2026-11-26T00:00:00Z.

`main.py` (Python 3.13, stdlib only, no network) forecasts five quantiles
(0.05/0.25/0.5/0.75/0.95) per trophy, scored by mean pinball loss on the
logit scale.

Referral code: `gxj2dtr1` — owner [jeffchen-lang](https://github.com/jeffchen-lang), repo `gta6-trophy-rarity-forecast-01`.

## Model

An earn rate is earners / players tracked, and both sides move in week
one: new players dilute every rate while story progress grows each
trophy's earners. So each side is fitted separately, on the log scale
against log-time since launch, `x = ln(1 + hours since 2026-11-19T00:00:00Z)`:

- **Players tracked:** one OLS line on `ln P` over the case's distinct
  reads, slope clipped to [0, 1.5].
- **Earners per trophy:** one OLS line on `ln E` (rate x players tracked),
  its slope shrunk toward a group prior — story chapters rise fastest
  (1.3), missions and quick activities next (1.0), grind/collectibles
  slower (0.7), platinum / final mission / 100% slowest (0.6).

The median at the target is `100 * earners(target) / players(target)`.
The history is a step function — Exophase refreshes the page in jumps
and the house re-reads the same frozen page for hours — so the fits use
each *distinct* rate vector once, at its first-seen time, instead of
letting long frozen stretches dominate the slope.

Two special cases:

- **Flat history** (one distinct vector, as on launch night) carries no
  trend, but the page has then been frozen for hours and a refresh is
  pending: the next refresh brings a wave of new players (the house saw
  players tracked jump 638 → 3,868 at the first refresh) while earners
  grow only by hours of play, so every rate is expected to fall. The
  median is shifted down 1.3 logit units instead of repeating the last
  rate. The 100% trophy ("Just Like Paradise", complete the first
  mission) is exempt: it stayed at 100% through that 6x refresh.
- **Story order is enforced:** chapters must be completed in order, the
  final mission precedes 100% completion, and the platinum needs
  everything, so every quantile obeys Gone Fishing ≥ Ch.1 ≥ Ch.2 ≥ Ch.3 ≥
  Ch.4 ≥ final ≥ 100% ≥ platinum.

**Spread:** quantiles are median ± z_tau · sigma on the logit scale.
Sigma combines a random-walk term (0.5 per √day, as the house baseline),
a refresh floor (0.55 logit; 1.0 when the history is flat and the next
refresh is the whole story), and the fits' prediction uncertainty
(residuals + slope error over the extrapolation gap).

## Practice results (pinball loss, lower is better)

The kit holds no local scorer and no practice answers — the answers are
the refreshed page at the targets, which the kit does not include. The
losses below are against **proxy truths**: the 04:30 refresh, players
638 → 3,868, with earners grown by a factor g since launch (every rate
but the 100% one falls, as the rules state). The house baseline scores
~1.51 on the real practice per the rules; it reproduces 1.47 at g=1.25,
so the realistic proxy sits near g≈1.2 (earners grew ~20% while players
grew 6x).

| proxy (earner growth g) | this model | house baseline |
| --- | --- | --- |
| g = 1.0 (pure dilution) | 0.298 (miss 0.10, cov 1.00) | 1.687 (cov 0.00) |
| g = 1.25 (calibrated) | 0.254 (miss 0.06, cov 1.00) | 1.468 (cov 0.00) |
| g = 1.5 | 0.218 (miss 0.02, cov 1.00) | 1.288 (cov 0.00) |
| g = 2.0 | 0.234 (miss 0.05, cov 1.00) | 1.005 (cov 0.00) |
| g = 2.46 | 0.275 (miss 0.11, cov 1.00) | 0.801 (cov 0.00) |

The edge on practice comes from anticipating the pending refresh
(median down ~1.3 logit) and a spread wide enough to cover a
refresh-sized move — the house baseline's narrow 90-minute band misses
every refreshed rate.

A synthetic final-like case (308 reads over 2 days, players
638 → ~56k, log-time power-law growth per group) was also used to check
the non-flat code path: medians track the synthetic truth closely
(Ch.1 23.5% vs 23.3% true, platinum 0.10% vs 0.10% true), loss 0.239 vs
house 0.250. That test validates the path, not the real world — the
synthetic world was generated from the same curve family the model fits.

Deterministic (two runs byte-identical); ~0.05 s per case, far under the
8 s limit; outputs ~2.8 KB, far under the 65,536-byte cap. Edge cases
verified: flat history, the 100% trophy, single-case grader-style run
(`python3 -P -s main.py`).
