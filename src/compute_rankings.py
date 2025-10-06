# --- src/compute_rankings.py (REPLACE ENTIRE FILE) ---
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

YEAR = int(os.getenv("YEAR", season_from_today()))
OUT_DIR = os.path.join("docs", "data")
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join("src", "weights.yaml"), "r") as f:
    W = yaml.safe_load(f)

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

def season_fbs_games(year: int) -> pd.DataFrame:
    games = games_api.get_games(year=year)
    rows = []
    for g in games:
        if g.home_points is None or g.away_points is None:  # only played games
            continue
        if g.home_conference is None or g.away_conference is None:  # FBS vs FBS
            continue
        rows.append({
            "week": g.week,
            "date": g.start_date,
            "home_team": g.home_team, "home_conf": g.home_conference,
            "home_points": g.home_points,
            "away_team": g.away_team, "away_conf": g.away_conference,
            "away_points": g.away_points,
            "neutral_site": bool(g.neutral_site),
        })
    return pd.DataFrame(rows)

def advanced_season_stats(year: int) -> pd.DataFrame:
    try:
        adv = stats_api.get_advanced_team_season_stats(year=year)
    except ApiException:
        adv = stats_api.get_advanced_season_team_stats(year=year)
    data = []
    for t in adv:
        off = t.offense or {}
        de = t.defense or {}
        data.append({
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
        })
    return pd.DataFrame(data)

def fbs_teams(year: int) -> pd.DataFrame:
    teams = teams_api.get_teams(year=year, classification="fbs")
    return pd.DataFrame([{"team": t.school, "conference": t.conference} for t in teams])

def compute_records(gdf: pd.DataFrame) -> pd.DataFrame:
    rec = {}
    for _, r in gdf.iterrows():
        home, away = r["home_team"], r["away_team"]
        hp, ap = r["home_points"], r["away_points"]
        rec.setdefault(home, {"team": home, "wins": 0, "losses": 0})
        rec.setdefault(away, {"team": away, "wins": 0, "losses": 0})
        if hp > ap:
            rec[home]["wins"] += 1; rec[away]["losses"] += 1
        else:
            rec[away]["wins"] += 1; rec[home]["losses"] += 1
    df = pd.DataFrame(rec.values())
    df["games"] = df["wins"] + df["losses"]
    df["win_pct"] = (df["wins"] / df["games"].replace(0, np.nan)).fillna(0.0)
    return df

def performance_composite(stats: pd.DataFrame, W) -> pd.DataFrame:
    off_w, def_w = W["offense"], W["defense"]
    df = stats.copy()

    # z-scores
    df["z_off_ppa"]  = zscore(df["off_ppa"])
    df["z_off_sr"]   = zscore(df["off_success_rate"])
    df["z_off_expl"] = zscore(df["off_explosiveness"])
    df["z_off_ppo"]  = zscore(df["off_pts_per_opp"])

    df["z_def_ppa"]  = zscore(invert(df["def_ppa"]))
    df["z_def_sr"]   = zscore(invert(df["def_success_rate"]))
    df["z_def_expl"] = zscore(invert(df["def_explosiveness"]))
    df["z_def_ppo"]  = zscore(invert(df["def_pts_per_opp"]))

    df["off_score_z"] = (
        off_w["ppa"] * df["z_off_ppa"] +
        off_w["success_rate"] * df["z_off_sr"] +
        off_w["explosiveness"] * df["z_off_expl"] +
        off_w["pts_per_opp"] * df["z_off_ppo"]
    )
    df["def_score_z"] = (
        def_w["ppa"] * df["z_def_ppa"] +
        def_w["success_rate"] * df["z_def_sr"] +
        def_w["explosiveness"] * df["z_def_expl"] +
        def_w["pts_per_opp"] * df["z_def_ppo"]
    )

    df["perf_z"] = 0.5 * df["off_score_z"] + 0.5 * df["def_score_z"]
    df["performance_score"] = scale_0_100(df["perf_z"])
    return df

def build_schedules(gdf: pd.DataFrame) -> dict:
    opps = {}
    for _, r in gdf.iterrows():
        a, b = r["home_team"], r["away_team"]
        opps.setdefault(a, set()).add(b)
        opps.setdefault(b, set()).add(a)
    return opps

def compute_ratings_sos(perf: pd.DataFrame, schedules: dict) -> pd.DataFrame:
    perf_map = dict(zip(perf["team"], perf["performance_score"]))
    values = []
    for team, oppset in schedules.items():
        opp_scores = [perf_map.get(o) for o in oppset if o in perf_map]
        sos = mean_ignore_none(opp_scores)
        values.append({"team": team, "sos_rating": sos})
    return pd.DataFrame(values)

def compute_quality_wins(gdf: pd.DataFrame, perf: pd.DataFrame, sos_df: pd.DataFrame, scalar: float) -> pd.DataFrame:
    perf_map = dict(zip(perf["team"], perf["performance_score"]))
    sos_map  = dict(zip(sos_df["team"], sos_df["sos_rating"]))
    qw = {}
    for _, r in gdf.iterrows():
        home, away = r["home_team"], r["away_team"]
        hp, ap = r["home_points"], r["away_points"]
        winner = home if hp > ap else away
        loser  = away if hp > ap else home
        opp_perf = perf_map.get(loser)
        opp_sos  = sos_map.get(loser)
        bonus = 0.0 if (opp_perf is None or opp_sos is None) else scalar * ((opp_perf + opp_sos) / 2.0)
        qw[winner] = qw.get(winner, 0.0) + bonus
        qw.setdefault(loser, 0.0)
    return pd.DataFrame([{"team": t, "quality_wins": v} for t, v in qw.items()])

def head_to_head_map(gdf: pd.DataFrame) -> dict:
    beaten = {}
    for _, r in gdf.iterrows():
        w = r["home_team"] if r["home_points"] > r["away_points"] else r["away_team"]
        l = r["away_team"] if w == r["home_team"] else r["home_team"]
        beaten.setdefault(w, set()).add(l)
    return beaten

def apply_h2h(df: pd.DataFrame, beaten_map: dict, threshold: float, nudge: float) -> pd.DataFrame:
    ranked = df.copy()
    score_map = dict(zip(ranked["team"], ranked["final_score"]))
    bump = {t: 0.0 for t in ranked["team"]}
    for winner, losers in beaten_map.items():
        for loser in losers:
            if winner in score_map and loser in score_map:
                if abs(score_map[winner] - score_map[loser]) <= threshold:
                    bump[winner] += nudge
    if bump:
        ranked["final_score"] = ranked.apply(lambda r: r["final_score"] + bump.get(r["team"], 0.0), axis=1)
        ranked = ranked.sort_values("final_score", ascending=False).reset_index(drop=True)
    return ranked

def main():
    print(f"Building rankings for {YEAR}")
    games = season_fbs_games(YEAR)
    if games.empty:
        raise SystemExit("No FBS games found for the season.")

    records = compute_records(games)
    adv = advanced_season_stats(YEAR)
    fbs = fbs_teams(YEAR)

    adv = adv.merge(fbs[["team","conference"]], on=["team","conference"], how="inner")
    perf = performance_composite(adv, W)

    schedules = build_schedules(games)
    sos = compute_ratings_sos(perf, schedules)
    qw  = compute_quality_wins(games, perf, sos, scalar=W["quality_win_scalar"])

    df = fbs.merge(records, on="team", how="left") \
            .merge(perf[["team","conference","performance_score",
                         "off_ppa","off_success_rate","off_explosiveness","off_pts_per_opp",
                         "def_ppa","def_success_rate","def_explosiveness","def_pts_per_opp"]],
                   on=["team","conference"], how="left") \
            .merge(sos, on="team", how="left") \
            .merge(qw, on="team", how="left")

    # clean
    for col in ["wins","losses","games"]:
        df[col] = df[col].fillna(0).astype(int)
    for col in ["win_pct","performance_score","sos_rating","quality_wins",
                "off_ppa","off_success_rate","off_explosiveness","off_pts_per_opp",
                "def_ppa","def_success_rate","def_explosiveness","def_pts_per_opp"]:
        df[col] = df[col].fillna(0.0)

    # final score
    df["final_score"] = (
        W["final"]["performance"] * df["performance_score"]
        + W["final"]["sos"] * df["sos_rating"]
        + W["final"]["quality_wins"] * df["quality_wins"]
    )

    # component ranks
    df["performance_rank"] = df["performance_score"].rank(ascending=False, method="min").astype(int)
    df["sos_rank"]         = df["sos_rating"].rank(ascending=False, method="min").astype(int)

    ranked = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    ranked["overall_rank"] = (ranked.index + 1).astype(int)

    # head-to-head nudge
    h2h = head_to_head_map(games)
    ranked = apply_h2h(ranked, h2h, threshold=W["h2h_threshold"], nudge=W["h2h_nudge"])

    # Top 25 JSON (with rich stats)
    top25 = ranked.head(25).copy()
    out = []
    for _, r in top25.iterrows():
        out.append({
            "rank":            int(r["overall_rank"]),
            "team":            r["team"],
            "conference":      r["conference"],
            "record":          f"{int(r['wins'])}-{int(r['losses'])}",
            "games":           int(r["games"]),
            "win_pct":         round(float(r["win_pct"]), 3),
            "final_score":     round(float(r["final_score"]), 3),
            "performance":     round(float(r["performance_score"]), 2),
            "performance_rank":int(r["performance_rank"]),
            "sos":             round(float(r["sos_rating"]), 2),
            "sos_rank":        int(r["sos_rank"]),
            "quality_wins":    round(float(r["quality_wins"]), 3),
            "off_ppa":         round(float(r["off_ppa"]), 4),
            "off_success_rate":round(float(r["off_success_rate"]), 4),
            "off_explosiveness":round(float(r["off_explosiveness"]), 4),
            "off_pts_per_opp": round(float(r["off_pts_per_opp"]), 4),
            "def_ppa":         round(float(r["def_ppa"]), 4),
            "def_success_rate":round(float(r["def_success_rate"]), 4),
            "def_explosiveness":round(float(r["def_explosiveness"]), 4),
            "def_pts_per_opp": round(float(r["def_pts_per_opp"]), 4),
            "slug":            slugify_team(r["team"]),
        })

    with open(os.path.join(OUT_DIR, "rankings.json"), "w") as f:
        json.dump({
            "season": YEAR,
            "last_build_utc": pd.Timestamp.utcnow().isoformat(),
            "top25": out
        }, f, indent=2)

    print("Wrote docs/data/rankings.json")

if __name__ == "__main__":
    main()
