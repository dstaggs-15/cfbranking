import os
import json
from dateutil import tz
import yaml
import pandas as pd
import numpy as np

import cfbd
from cfbd.rest import ApiException

from utils import (
    zscore, invert, scale_0_100, slugify_team, season_from_today, mean_ignore_none
)

# -------- CONFIG --------
YEAR = int(os.getenv("YEAR", season_from_today()))
OUT_DIR = os.path.join("docs", "data")
LOGO_DIR = os.path.join("docs", "logos")  # put logos here (step 5)
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join("src", "weights.yaml"), "r") as f:
    W = yaml.safe_load(f)

# -------- CFBD AUTH --------
api_key = os.getenv("CFBD_API_KEY")
if not api_key:
    raise SystemExit("Set CFBD_API_KEY environment variable or GitHub Secret.")

configuration = cfbd.Configuration()
configuration.api_key['Authorization'] = api_key
configuration.api_key_prefix['Authorization'] = 'Bearer'
api_client = cfbd.ApiClient(configuration)

games_api = cfbd.GamesApi(api_client)
stats_api = cfbd.StatsApi(api_client)
teams_api = cfbd.TeamsApi(api_client)

# -------- FETCH DATA --------
def fetch_fbs_games(year: int) -> pd.DataFrame:
    # Get all games; keep only games played with scores; keep only FBS vs FBS
    games = games_api.get_games(year=year)
    rows = []
    for g in games:
        if g.home_points is None or g.away_points is None:
            continue
        # Keep FBS games only (CFBD marks conference for FBS)
        if g.home_conference is None or g.away_conference is None:
            continue
        rows.append({
            "week": g.week,
            "date": g.start_date,
            "home_team": g.home_team,
            "home_conf": g.home_conference,
            "home_points": g.home_points,
            "away_team": g.away_team,
            "away_conf": g.away_conference,
            "away_points": g.away_points,
            "neutral_site": bool(g.neutral_site),
        })
    return pd.DataFrame(rows)

def fetch_advanced_season_stats(year: int) -> pd.DataFrame:
    # Advanced offense/defense season stats
    # CFBD Python client method: get_advanced_team_season_stats(year=YEAR)
    try:
        adv = stats_api.get_advanced_team_season_stats(year=year)
    except ApiException as e:
        # Backward compat name (rare): get_advanced_season_team_stats
        adv = stats_api.get_advanced_season_team_stats(year=year)
    data = []
    for t in adv:
        # offense/defense objects can be None early in season; guard it
        off = t.offense or {}
        de = t.defense or {}
        row = {
            "team": t.team,
            "conference": t.conference,
            "off_ppa": getattr(off, "ppa", None),
            "off_success_rate": getattr(off, "success_rate", None),
            "off_explosiveness": getattr(off, "explosiveness", None),
            "off_pts_per_opp": getattr(off, "points_per_opportunity", None),
            "def_ppa": getattr(de, "ppa", None),
            "def_success_rate": getattr(de, "success_rate", None),
            "def_explosiveness": getattr(de, "explosiveness", None),
            "def_pts_per_opp": getattr(de, "points_per_opportunity", None),
        }
        data.append(row)
    return pd.DataFrame(data)

def fetch_fbs_teams(year: int) -> pd.DataFrame:
    teams = teams_api.get_teams(year=year, classification="fbs")
    return pd.DataFrame([{"team": t.school, "conference": t.conference} for t in teams])

# -------- METRICS --------
def compute_records(gdf: pd.DataFrame) -> pd.DataFrame:
    rec = {}
    for _, r in gdf.iterrows():
        home, away = r["home_team"], r["away_team"]
        hp, ap = r["home_points"], r["away_points"]
        # update home
        rec.setdefault(home, {"team": home, "wins": 0, "losses": 0})
        rec.setdefault(away, {"team": away, "wins": 0, "losses": 0})
        if hp > ap:
            rec[home]["wins"] += 1
            rec[away]["losses"] += 1
        else:
            rec[away]["wins"] += 1
            rec[home]["losses"] += 1
    df = pd.DataFrame(rec.values())
    df["games"] = df["wins"] + df["losses"]
    df["win_pct"] = df["wins"] / df["games"].replace(0, np.nan)
    df["win_pct"] = df["win_pct"].fillna(0.0)
    return df

def performance_composite(stats: pd.DataFrame) -> pd.DataFrame:
    # Z-score each selected metric, invert defensive metrics where lower is better
    off_w = W["offense"]
    def_w = W["defense"]
    df = stats.copy()

    # Convert to z-scores
    df["z_off_ppa"] = zscore(df["off_ppa"])
    df["z_off_sr"] = zscore(df["off_success_rate"])
    df["z_off_expl"] = zscore(df["off_explosiveness"])
    df["z_off_ppo"] = zscore(df["off_pts_per_opp"])

    df["z_def_ppa"] = zscore(invert(df["def_ppa"]))
    df["z_def_sr"] = zscore(invert(df["def_success_rate"]))
    df["z_def_expl"] = zscore(invert(df["def_explosiveness"]))
    df["z_def_ppo"] = zscore(invert(df["def_pts_per_opp"]))

    # Weighted sums
    df["off_score_z"] = (
        off_w["ppa"] * df["z_off_ppa"]
        + off_w["success_rate"] * df["z_off_sr"]
        + off_w["explosiveness"] * df["z_off_expl"]
        + off_w["pts_per_opp"] * df["z_off_ppo"]
    )
    df["def_score_z"] = (
        def_w["ppa"] * df["z_def_ppa"]
        + def_w["success_rate"] * df["z_def_sr"]
        + def_w["explosiveness"] * df["z_def_expl"]
        + def_w["pts_per_opp"] * df["z_def_ppo"]
    )
    # Combine offense/defense evenly, then scale 0-100
    df["perf_z"] = 0.5 * df["off_score_z"] + 0.5 * df["def_score_z"]
    df["performance_score"] = scale_0_100(df["perf_z"])
    return df[["team", "conference", "performance_score"]]

def build_schedules(gdf: pd.DataFrame) -> dict:
    opps = {}
    for _, r in gdf.iterrows():
        a, b = r["home_team"], r["away_team"]
        opps.setdefault(a, set()).add(b)
        opps.setdefault(b, set()).add(a)
    return opps  # {team: set(opponents)}

def compute_ratings_based_sos(perf: pd.DataFrame, schedules: dict) -> pd.DataFrame:
    perf_map = dict(zip(perf["team"], perf["performance_score"]))
    values = []
    for team, oppset in schedules.items():
        opp_scores = [perf_map.get(o) for o in oppset if o in perf_map]
        sos = mean_ignore_none(opp_scores)
        values.append({"team": team, "sos_rating": sos})
    return pd.DataFrame(values)

def compute_quality_wins(gdf: pd.DataFrame, perf: pd.DataFrame, sos_df: pd.DataFrame, scalar: float) -> pd.DataFrame:
    perf_map = dict(zip(perf["team"], perf["performance_score"]))
    sos_map = dict(zip(sos_df["team"], sos_df["sos_rating"]))
    qw = {}
    for _, r in gdf.iterrows():
        home, away = r["home_team"], r["away_team"]
        hp, ap = r["home_points"], r["away_points"]

        winner = home if hp > ap else away
        loser = away if hp > ap else home

        opp_perf = perf_map.get(loser)
        opp_sos  = sos_map.get(loser)
        if opp_perf is None or opp_sos is None:
            bonus = 0.0
        else:
            # scale by opponent quality and opponent's SOS; average then scale
            bonus = scalar * ((opp_perf + opp_sos) / 2.0)

        qw[winner] = qw.get(winner, 0.0) + bonus
        qw.setdefault(loser, 0.0)

    return pd.DataFrame([{"team": t, "quality_wins": v} for t, v in qw.items()])

def head_to_head_nudges(gdf: pd.DataFrame) -> dict:
    # store who beat whom this year
    beaten = {}
    for _, r in gdf.iterrows():
        w = r["home_team"] if r["home_points"] > r["away_points"] else r["away_team"]
        l = r["away_team"] if w == r["home_team"] else r["home_team"]
        beaten.setdefault(w, set()).add(l)
    return beaten  # {winner: {losers}}

def apply_head_to_head_swaps(ranked_df: pd.DataFrame, beaten_map: dict, threshold: float, nudge: float) -> pd.DataFrame:
    # If A beat B and their scores are within threshold, nudge winner up slightly
    df = ranked_df.copy()
    score_map = dict(zip(df["team"], df["final_score"]))
    bonus = {t: 0.0 for t in df["team"]}

    for winner, losers in beaten_map.items():
        for loser in losers:
            if winner in score_map and loser in score_map:
                if abs(score_map[winner] - score_map[loser]) <= threshold:
                    bonus[winner] += nudge

    if bonus:
        df["final_score"] = df.apply(lambda r: r["final_score"] + bonus.get(r["team"], 0.0), axis=1)
        df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    return df

def main():
    print(f"Building rankings for {YEAR}")
    games = fetch_fbs_games(YEAR)
    if games.empty:
        raise SystemExit("No FBS games found. Are we early in the season?")

    records = compute_records(games)

    adv = fetch_advanced_season_stats(YEAR)
    # Merge with FBS team list to restrict to current FBS
    fbs = fetch_fbs_teams(YEAR)
    adv = adv.merge(fbs[["team", "conference"]], on=["team", "conference"], how="inner")

    perf = performance_composite(adv)  # team, conf, performance_score
    schedules = build_schedules(games)
    sos = compute_ratings_based_sos(perf, schedules)
    qw = compute_quality_wins(games, perf, sos, scalar=W["quality_win_scalar"])

    # Join everything
    df = fbs.merge(records, on="team", how="left") \
            .merge(perf, on=["team", "conference"], how="left") \
            .merge(sos, on="team", how="left") \
            .merge(qw, on="team", how="left")

    df["wins"] = df["wins"].fillna(0).astype(int)
    df["losses"] = df["losses"].fillna(0).astype(int)
    df["games"] = df["games"].fillna(0).astype(int)
    df["win_pct"] = df["win_pct"].fillna(0.0)
    df["performance_score"] = df["performance_score"].fillna(0.0)
    df["sos_rating"] = df["sos_rating"].fillna(0.0)
    df["quality_wins"] = df["quality_wins"].fillna(0.0)

    # Final score
    df["final_score"] = (
        W["final"]["performance"] * df["performance_score"]
        + W["final"]["sos"] * df["sos_rating"]
        + W["final"]["quality_wins"] * df["quality_wins"]
    )

    # Head-to-head nudge
    beaten_map = head_to_head_nudges(games)
    ranked = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    ranked = apply_head_to_head_swaps(
        ranked, beaten_map,
        threshold=W["h2h_threshold"],
        nudge=W["h2h_nudge"]
    )

    # Attach a simple logo slug (you drop files into docs/logos/{slug}.png)
    ranked["logo"] = ranked["team"].apply(lambda x: f"logos/{slugify_team(x)}.png")

    # Build Top-25 JSON
    top25 = ranked.head(25).copy()
    out = []
    for _, r in top25.iterrows():
        out.append({
            "rank": int(_ + 1),
            "team": r["team"],
            "conference": r["conference"],
            "record": f"{int(r['wins'])}-{int(r['losses'])}",
            "final_score": round(float(r["final_score"]), 3),
            "performance": round(float(r["performance_score"]), 2),
            "sos": round(float(r["sos_rating"]), 2),
            "quality_wins": round(float(r["quality_wins"]), 3),
            "logo": r["logo"],
        })

    # Write JSON
    with open(os.path.join(OUT_DIR, "rankings.json"), "w") as f:
        json.dump({"season": YEAR, "last_build_utc": pd.Timestamp.utcnow().isoformat(), "top25": out}, f, indent=2)

    print("Wrote docs/data/rankings.json")

if __name__ == "__main__":
    main()
