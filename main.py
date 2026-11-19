"""GTA VI trophy rarity forecast — logit growth curves with a dilution split.

Forecasts, for each of GTA VI's PS5 trophies, its Exophase earn rate at the
case's target. Python 3.13, standard library only, deterministic, no network.

Model
-----
An earn rate is earners / players_tracked, and both sides move in week one:
new players dilute every rate while story progress grows the earners of each
trophy. So each side is fitted separately, on the log scale against
log-time since launch, x = ln(1 + hours since 2026-11-19T00:00:00Z):

  * ln players_tracked: one OLS line over the case's distinct reads.
  * ln earners of trophy i (rate x players_tracked): one OLS line per trophy,
    its slope shrunk toward a group prior (story chapters rise fastest,
    grind/collectible trophies slowest, platinum/final slowest of all).

The median at the target is 100 * earners(target) / players(target). The
history is a step function — Exophase refreshes the page in jumps and the
house re-reads the same frozen page for hours — so the fits use each
*distinct* rate vector once, at its first-seen time: fitting every read
would let long frozen stretches dominate the slope.

Two special cases:

  * A completely flat history (one distinct vector, as on launch night)
    carries no trend at all, but the page has then been frozen for hours and
    a refresh is pending: the next refresh brings a wave of new players
    (the house saw players tracked jump 638 -> 3,868 at the first refresh)
    while earners grow only by hours of play, so every rate is expected to
    fall. The median is shifted down 1.3 logit units (about a x0.27 rate
    factor: earners ~x1.5 against players ~x6) instead of repeating the
    last rate. The 100% trophy ("Just Like Paradise", complete the first
    mission) is exempt: it stayed at 100% through that 6x refresh.
  * Story order is enforced: chapters must be completed in order, the final
    mission precedes 100% completion, and the platinum needs everything, so
    the medians (and every quantile) obey
    Gone Fishing >= Ch.1 >= Ch.2 >= Ch.3 >= Ch.4 >= final >= 100% >= platinum.

Spread
------
Quantiles are median +/- z_tau * sigma on the logit scale. sigma combines
a random-walk term (0.5 per sqrt day, as the house baseline), a refresh
floor (one refresh can move a displayed rate a lot: 0.55 logit, and 1.0
when the history is flat and the next refresh is the whole story), and the
fits' prediction uncertainty (residuals + slope error over the extrapolation
gap). The trophy at 100% gets a near-degenerate band: it has printed
100.00% at two very different player counts.

What the program must do (per the competition rules):
  - read every <id>.in in the input directory ($MAKEFUN_INPUT_DIR, else
    /input), one case per file;
  - write <id>.out to the output directory ($MAKEFUN_OUTPUT_DIR, else
    /output): {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]},
    one row per trophy, five finite numbers in [0, 100], non-decreasing,
    at most 65,536 bytes. Two runs write identical bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves the repo root off sys.path; put it back first, so a
# module shipped beside this file would import. (This entry ships one file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Launch: GTA VI released 2026-11-19 (game.release_date in the case).
RELEASE = datetime(2026, 11, 19, tzinfo=timezone.utc)

# Logit shift applied to the median when the history is completely flat:
# the pending refresh dilutes every rate (see the module docstring).
FLAT_SHIFT = 1.3
# Spread of the flat-history forecast, in logit units: one refresh moves a
# displayed rate by roughly a logit or more, so the band must span that.
FLAT_SIGMA = 1.0
# Random-walk spread per square-root day, as in the house baseline.
RW_SIGMA = 0.5
# Refresh floor: the displayed rate lags the true rate by up to one refresh.
REFRESH_SIGMA = 0.55

# Earner-growth priors (slope of ln earners against ln(1 + hours)), by group.
# Story chapters rise fastest; grind/collectibles and the endgame slowest.
PRIOR_EARLY_STORY = 1.3
PRIOR_MISSION = 1.0
PRIOR_MISC = 1.0
PRIOR_GRIND = 0.7
PRIOR_ENDGAME = 0.6
PRIOR_PLAYERS = 0.8

# Trophy ids, in Exophase order (see the case's trophy list).
EARLY_STORY = {3, 4, 5, 6}            # Complete Chapter 1..4
ENDGAME = {1, 7, 14}                   # platinum, final mission, 100%
MISSIONS = {8, 9, 10, 11, 12, 13, 22, 23, 24, 26, 27}
GRIND = {15, 17, 18, 19, 20, 25, 29, 30, 31, 32, 35, 40, 42, 52, 54}
# Story chain, most-earned first: each trophy's rate cannot exceed the one
# before it (chapters in order; final before 100%; platinum needs all).
CHAIN = [2, 3, 4, 5, 6, 7, 14, 1]


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    return math.log(clamp(pct) / (100 - clamp(pct)))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100 / (1 + math.exp(-lg))
    e = math.exp(lg)
    return 100 * e / (1 + e)


def round4(x: float) -> float | int:
    """Round to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def norm_ppf(p: float) -> float:
    """Standard normal quantile (Acklam's rational approximation)."""
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285984469534e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    p_low = 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= 1 - p_low:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(when: datetime, release: datetime) -> float:
    """Hours from launch; floored just above -1 so ln(1 + t) stays defined."""
    return max((when - release).total_seconds() / 3600.0, -0.99)


def ols(xs: list[float], ys: list[float]) -> tuple[float, float, float, float]:
    """Least squares y ~ a + b*x. Returns (a, b, sxx, resid_var)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    resid = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    var = resid / max(n - 2, 1) if n > 2 else 0.0
    return a, b, sxx, var


def group_prior(trophy_id: int, tier: str) -> float:
    if trophy_id in EARLY_STORY:
        return PRIOR_EARLY_STORY
    if trophy_id in ENDGAME:
        return PRIOR_ENDGAME
    if trophy_id in MISSIONS:
        return PRIOR_MISSION
    if trophy_id in GRIND:
        return PRIOR_GRIND
    if tier in ("platinum", "gold"):
        return PRIOR_ENDGAME
    if tier == "silver":
        return PRIOR_GRIND
    return PRIOR_MISC


def distinct_reads(case: dict) -> list[dict]:
    """History with each distinct (players, rates) vector kept once, at its
    first-seen time. Long frozen stretches count once, not once per read."""
    seen: set = set()
    out = []
    for read in case.get("history", []):
        key = (read.get("players_tracked"), tuple(read.get("rates") or []))
        if key in seen:
            continue
        seen.add(key)
        out.append(read)
    return out


def forecast_case(case: dict, name: str) -> dict:
    trophies = case["trophies"]
    levels = list(case.get("levels") or LEVELS)
    z = [norm_ppf(p) for p in levels]

    release = RELEASE
    try:
        release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    except Exception:
        pass
    target = parse_time(case["target"])
    x_target = math.log1p(hours_since_release(target, release))
    horizon_days = (case.get("horizon_minutes") or 0) / 1440.0

    reads = distinct_reads(case)
    if not reads:
        # No history at all (should not happen): flat 1% medians, max spread.
        return {"trophies": [
            {"id": t["id"], "q": [round4(inv_logit(logit(1.0) + zz * 2.0)) for zz in z]}
            for t in trophies]}

    last = reads[-1]
    last_rates = last["rates"]
    last_players = last.get("players_tracked") or 0

    xs = [math.log1p(hours_since_release(parse_time(r["at"]), release)) for r in reads]
    x_last = xs[-1]

    # --- players tracked: one OLS line on ln P against log-time. ---
    ln_p = [math.log(max(r.get("players_tracked") or 1, 1)) for r in reads]
    if len(reads) >= 2:
        a_p, b_p, sxx_p, var_p = ols(xs, ln_p)
        w_p = sxx_p / (sxx_p + 1.0)
        b_p = w_p * b_p + (1 - w_p) * PRIOR_PLAYERS
        b_p = min(max(b_p, 0.0), 1.5)
        a_p = (sum(ln_p) / len(ln_p)) - b_p * (sum(xs) / len(xs))
        se_p = math.sqrt(var_p * (1.0 / len(xs) + (x_target - sum(xs) / len(xs)) ** 2 / sxx_p)) \
            if sxx_p > 0 else 0.0
    else:
        a_p, b_p, sxx_p, var_p, se_p = ln_p[0], 0.0, 0.0, 0.0, 0.0
    p_target = math.exp(a_p + b_p * x_target)
    p_target = min(max(p_target, float(last_players or 1)), float(last_players or 1) * 30.0 + 1.0)

    flat = len(reads) == 1
    rows: dict[int, list[float]] = {}

    for i, trophy in enumerate(trophies):
        tid = trophy["id"]
        r_last = last_rates[i] if i < len(last_rates) else None
        if r_last is None:
            r_last = 0.0
        # The 100% trophy: complete the first mission; everyone tracked has
        # it, at every player count seen. Keep it at 100% with a thin band.
        if r_last >= 99.99:
            rows[tid] = [99.9 if lv <= 0.05 else 99.99 if lv <= 0.25 else 100.0
                         for lv in levels]
            continue

        if flat:
            med_logit = logit(r_last) - FLAT_SHIFT
            sigma = FLAT_SIGMA
        else:
            # Earners of this trophy: OLS on ln E against log-time, slope
            # shrunk toward the group prior; earners never fall.
            ex, ey = [], []
            for r, x in zip(reads, xs):
                rate = r["rates"][i] if i < len(r.get("rates") or []) else None
                players = r.get("players_tracked") or 0
                if rate is None or players <= 0:
                    continue
                ex.append(x)
                ey.append(math.log(max(rate / 100.0 * players, 0.5)))
            if len(ex) >= 2:
                a_e, b_e, sxx_e, var_e = ols(ex, ey)
                w_e = sxx_e / (sxx_e + 0.5)
                b_e = w_e * b_e + (1 - w_e) * group_prior(tid, trophy.get("tier", ""))
                b_e = min(max(b_e, -0.5), 2.5)
                a_e = (sum(ey) / len(ey)) - b_e * (sum(ex) / len(ex))
                se_e = math.sqrt(var_e * (1.0 / len(ex) + (x_target - sum(ex) / len(ex)) ** 2 / sxx_e)) \
                    if sxx_e > 0 else 0.0
            elif len(ex) == 1:
                a_e, b_e, se_e = ey[0], group_prior(tid, trophy.get("tier", "")), 0.0
            else:
                a_e, b_e, se_e = math.log(0.5), group_prior(tid, trophy.get("tier", "")), 0.0
            e_last = max(r_last / 100.0 * (last_players or 1), 0.5)
            e_target = max(math.exp(a_e + b_e * x_target), e_last)
            e_target = min(e_target, 0.999 * p_target)
            med_logit = logit(100.0 * e_target / p_target)
            fit_var = min(se_e * se_e + se_p * se_p, 1.0)
            sigma = math.sqrt(RW_SIGMA ** 2 * horizon_days + REFRESH_SIGMA ** 2 + fit_var)
            sigma = min(max(sigma, 0.8), 2.0)

        qs = [round4(inv_logit(med_logit + zz * sigma)) for zz in z]
        # Non-decreasing, and inside [0, 100] (rounding can touch the edges).
        qs = [min(max(q, 0), 100) for q in qs]
        for k in range(1, len(qs)):
            qs[k] = max(qs[k], qs[k - 1])
        rows[tid] = qs

    # Story order: a later chapter cannot be earned more often than an
    # earlier one. Enforce on every quantile, walking the chain.
    for a, b in zip(CHAIN, CHAIN[1:]):
        if a in rows and b in rows:
            for k in range(len(levels)):
                rows[b][k] = min(rows[b][k], rows[a][k])

    return {"trophies": [{"id": t["id"], "q": rows[t["id"]]} for t in trophies]}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
