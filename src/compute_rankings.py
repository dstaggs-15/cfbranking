#!/usr/bin/env python3
"""
compute_rankings.py
Generates unbiased FBS-only college football rankings using CFBD API.
"""

import os
import json
import math
import datetime
import requests

CFBD_API_KEY = os.getenv("CFBD_API_KEY")

# ✅ Fixed YEAR variable handling
try:
    YEAR = int(os.getenv("YEAR") or datetime.datetime.now().year)
except ValueError:
    YEAR = datetime.datetime.now().year

HEADERS = {"Authorization": f"Bearer {CFBD_API_KEY}"}

DATA_DIR = "docs/data"
os.makedirs(DATA_DIR, exist_ok=True)
OUT_FILE = os.path.join(DATA_DIR, "rankings.json")

# -------------------------------------------------
# Helper: fetch CFBD data
# -------------------------------------------------

def fetch(endpoint, params=None):
    url = f"https://api.collegefootballdata.com/{endpoint}"
    r = requests.get(url, headers=HEADERS, params=params)
    r.raise_for_status()
    return r.json()

# -------------------------------------------------
# Data gathering
# -------------------------------------------------

def get_games(year):
    games = fetch("games", {"year": year, "seasonType": "regular", "division": "fbs"})
    fbs_games = []
    for g in games:
        if not g.get("homeTeam") or not g.get("awayTeam"):
            continue
        if g.get("homeConference") is None or g.get("awayConference") is None:
            # filter out FCS
            continue
        if g.get("homePoints") is None or g.get("awayPoints") is None:
            continue
        fbs_games.append(g)
    return fbs_games

def get_advanced(year):
    adv = fetch("stats/season/advanced", {"year": year})
    out = {}
    for t in adv:
        team = t.get("team")
        if not team:
            continue
        offense = t.get("offense", {})
        defense = t.get("defense", {})
        out[team] = {
            "off_ppa": offense.get("ppa", 0),
            "def_ppa": defense.get("ppa", 0),
            "off_sr": offense.get("successRate", 0),
            "def_sr": defense.get("successRate", 0)
        }
    return out

def get_sp_ratings(year):
    ratings = fetch("ratings/sp", {"year": year})
    out = {}
    for t in ratings:
        team = t.get("team")
        if not team:
            continue
        out[team] = {
            "sos": t.get("sos", 0.5)
        }
    return out

# -------------------------------------------------
# Scoring function
# -------------------------------------------------

def compute_team_score(team, metrics, results, rankings):
    wins = team["wins"]
    losses = team["losses"]
    games = wins + losses

    win_pct = wins / games if games > 0 else 0
    sos = metrics.get("sos", 0.5)
    avg_margin = metrics.get("avg_margin", 0.0)
    off_ppa = metrics.get("off_ppa", 0.0)
    def_ppa = metrics.get("def_ppa", 0.0)
    off_sr = metrics.get("off_sr", 0.0)
    def_sr = metrics.get("def_sr", 0.0)

    quality_wins = sum(1 for opp in results if opp["result"] == "W" and opp["opp_rank"] <= 25)
    bad_losses = sum(1 for opp in results if opp["result"] == "L" and opp["opp_rank"] > 75)

    # Stronger H2H logic
    h2h_bonus = 0
    for opp in results:
        if opp["result"] == "W" and opp["opp_rank"] < 30:
            h2h_bonus += 0.02
        if opp["result"] == "L" and opp["opp_rank"] > 70:
            h2h_bonus -= 0.03

    score = (
        0.50 * win_pct +
        0.20 * sos +
        0.10 * (avg_margin / 30) +
        0.10 * (off_ppa - def_ppa) +
        0.05 * (off_sr - def_sr) +
        0.05 * (0.02 * quality_wins - 0.02 * bad_losses) +
        h2h_bonus
    )

    return max(0, min(score, 1))

# -------------------------------------------------
# Ranking builder
# -------------------------------------------------

def build_rankings(year):
    print(f"Building rankings for {year}")
    games = get_games(year)
    adv = get_advanced(year)
    sp = get_sp_ratings(year)

    teams = {}
    for g in games:
        home, away = g["homeTeam"], g["awayTeam"]
        home_pts, away_pts = g["homePoints"], g["awayPoints"]

        # determine winner/loser
        if home_pts > away_pts:
            winner, loser = home, away
        elif away_pts > home_pts:
            winner, loser = away, home
        else:
            continue

        # initialize
        for t in [home, away]:
            if t not in teams:
                teams[t] = {"wins": 0, "losses": 0, "points_for": 0, "points_against": 0, "results": []}

        # record
        teams[winner]["wins"] += 1
        teams[loser]["losses"] += 1
        teams[home]["points_for"] += home_pts
        teams[home]["points_against"] += away_pts
        teams[away]["points_for"] += away_pts
        teams[away]["points_against"] += home_pts

        teams[winner]["results"].append({"opp": loser, "result": "W"})
        teams[loser]["results"].append({"opp": winner, "result": "L"})

    # compute average margin and merge metrics
    for t, info in teams.items():
        games_played = info["wins"] + info["losses"]
        avg_margin = 0
        if games_played > 0:
            avg_margin = (info["points_for"] - info["points_against"]) / games_played
        info["avg_margin"] = avg_margin
        info.update(adv.get(t, {}))
        info.update(sp.get(t, {}))

    # build provisional ranks
    scores = {}
    for t, info in teams.items():
        scores[t] = compute_team_score(info, info, info["results"], scores)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    rank_index = {t: i+1 for i, (t, _) in enumerate(ranked)}

    # update results with opponent ranks
    for t, info in teams.items():
        for r in info["results"]:
            r["opp_rank"] = rank_index.get(r["opp"], 130)

    # Re-score using updated ranks
    final_scores = {}
    for t, info in teams.items():
        final_scores[t] = compute_team_score(info, info, info["results"], rank_index)

    final = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    top25 = []
    for i, (team, score) in enumerate(final[:25]):
        info = teams[team]
        top25.append({
            "rank": i + 1,
            "team": team,
            "score": round(score, 4),
            "games": info["wins"] + info["losses"],
            "wins": info["wins"],
            "losses": info["losses"],
            "points_for": info["points_for"],
            "points_against": info["points_against"],
            "sos": round(info.get("sos", 0.5), 3),
            "avg_margin": round(info.get("avg_margin", 0), 1)
        })

    return {
        "season": year,
        "last_build_utc": datetime.datetime.utcnow().isoformat(),
        "top25": top25
    }

# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    data = build_rankings(YEAR)
    with open(OUT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Rankings written to {OUT_FILE}")

if __name__ == "__main__":
    main()
