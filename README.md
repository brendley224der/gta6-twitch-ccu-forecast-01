# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/brendley224der/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/brendley224der/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts, hour by hour, how many people watch Grand Theft Auto VI live on Twitch over its release day (19 November 2026, UTC), as five quantiles an hour, from the first hour's count and the release-day curves of past big launches. Pure Python standard library, deterministic, offline.

## The model

A launch-day curve is mostly a timezone story. GTA VI unlocks at local midnight, rolling around the world, so hour 1 (00:00–01:00 UTC) already holds New Zealand, Australia, Europe and the UK; viewership then dips in the deep night (hours 3–4), ramps as the US east coast (05:00 UTC) and west coast (08:00 UTC) unlock, peaks in the US daytime / European evening (hours 19–21), and decays into the night.

The curve's shape is stable across rolling-midnight launches; its amplitude relative to hour 1 is not — it saturates hard with the size of hour 1. Across the comparable launches, log(peak / hour-1) correlates about −0.93 with log(hour-1): a launch that already draws 875k in hour 1 barely grows, one that draws 25k roughly quadruples by the evening.

So at run time `main.py` fits, from the case's own `comparables`:

- the release-day curve of every comparable launch, keeping only rolling-midnight launches (unlock at midnight local) that are not early-access contaminated and not flagged weak, leaving out the GTA VI category's own pre-release history, which is not a launch;
- per hour, an OLS regression of log(viewers_h / viewers_1) on log(viewers_1), plus the median log-ratio across the launches;
- the forecast is a blend in log space — 70% regression, 30% median curve — with the quantile spread from the regression's residual standard deviation (floored at 0.20 log units), so the 5%–95% band is honest about how much launch days vary.

Validation, computed from the kit's own comparable launches (leave-one-out over the rolling-midnight set, scored with the contest's pinball loss on the log scale): mean loss 0.27, median 0.20, against the house persistence baseline's 0.38 / 0.25 on the same set. The house baseline scores about 0.387 on the practice; this model beats it by modelling the ramp-and-decay shape instead of repeating hour 1.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`input/` is the kit's `input/` directory (the practice cases `p00.in`–`p03.in`); the program reads every `*.in` there and writes one `*.out` per case to `out/`. In the grader the input and output directories are `/input` and `/output`. The run is deterministic: the same case always writes the same bytes, in well under a second.

---

<sub>Entered on [make.fun](https://make.fun/?ref=amsc6jjj&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
