#!/usr/bin/env python3
"""
compute_rankings.py
Generates unbiased FBS-only college football rankings using CFBD API.
Hardened against None values from CFBD (sos/metrics), safe YEAR handling,
and two-pass scoring with opponent-rank-aware adjustments.
"""

import os
import json
import datetime
import requests

CFBD_API_KEY = os.getenv("CFBD_API_KEY")

# ✅ Safe YEAR handling (never crashes on empty env)
try:
    YEAR = int(os.getenv("YEAR") or datetime.datetime.now().year)
except ValueError:
    YEAR = datetime.datetime.now().year

if not CFBD_API_KEY:
    raise RuntimeError("CFBD_API_KEY is not set in the environment.")

HEADERS = {"Authorization": f"Bearer {CFBD_API_KEY}"}

DATA_DIR = "docs/data"
os.makedirs(DATA_DIR, exist_ok=True)
OUT_FILE = os.path.join(DATA_DIR, "rankings.json")


# ----------------------------- Utilities -----------------------------

def safe_float(x, default=0.0):
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)

def fetch(endpoint, params=None):
    url = f"https://api.collegefootballdata.com/{endpoint}"
    r = requests.get(url, headers=HEADERS, params=params, timeout=45)
    r.raise_for_status()
    return r.json()


# ----------------------------- Data fetch -----------------------------

def get_games(year: int):
    """Pull regular-season FBS games with final scores only."""
    games = fetch("games", {"year": year, "seasonType": "regular", "division": "fbs"})
    fbs_games = []
    for g in games:
        # Must have teams
        home = g.get("homeTeam")
        away = g.get("awayTeam")
        if not home or not away:
            continue
        # Filter out non-FBS opponents by missing conference
        if g.get("homeConference") is None or g.get("awayConference") is None:
            continue
        # Need final score
        hp = g.get("homePoints")
        ap = g.get("awayPoints")
        if hp is None or ap is None:
            continue
        fbs_games.append(g)
    return fbs_games

def get_advanced(year: int):
    """Season-level advanced metrics."""
    adv = fetch("stats/season/advanced", {"year": year})
    out = {}
    for t in adv:
        team = t.get("team")
        if not team:
            continue
        offense = t.get("offense") or {}
        defense = t.get("defense") or {}
        out[team] = {
            "off_ppa": safe_float(offense.get("ppa"), 0.0),
            "def_ppa": safe_float(defense.get("ppa"), 0.0),
            "off_sr":  safe_float(offense.get("successRate"), 0.0),
            "def_sr":  safe_float(defense.get("successRate"), 0.0),
        }
    return out

def get_sp_ratings(year: int):
    """SP+ ratings (we only use SOS). Some teams may have sos=None; coerce to 0.5 midline."""
    ratings = fetch("ratings/sp", {"year": year})
    out = {}
    for t in ratings:
        team = t.get("team")
        if not team:
            continue
        out[team] = {
            "sos": safe_float(t.get("sos"), 0.5)
        }
    return out


# ----------------------------- Scoring -----------------------------

def compute_team_score(team, metrics, results, opp_ranks_known=False):
    """
    Two-pass scoring:
      - Pass 1: opp_ranks_known=False (don’t use quality/bad-loss/h2h features)
      - Pass 2: opp_ranks_known=True (quality wins / bad losses / h2h bonuses applied)
    All numeric inputs are safely coerced to floats.
    """
    wins = int(team.get("wins", 0))
    losses = int(team.get("losses", 0))
    games = wins + losses

    win_pct = (wins / games) if games > 0 else 0.0
    sos = safe_float(metrics.get("sos"), 0.5)
    avg_margin = safe_float(metrics.get("avg_margin"), 0.0)
    off_ppa = safe_float(metrics.get("off_ppa"), 0.0)
    def_ppa = safe_float(metrics.get("def_ppa"), 0.0)
    off_sr  = safe_float(metrics.get("off_sr"), 0.0)
    def_sr  = safe_float(metrics.get("def_sr"), 0.0)

    # Opp-rank-dependent features only on pass 2
    quality_wins = 0
    bad_losses = 0
    h2h_bonus = 0.0

    if opp_ranks_known:
        for opp in results:
            opp_rank = int(opp.get("opp_rank", 999))
            res = opp.get("result")
            if res == "W":
                if opp_rank <= 25:
                    quality_wins += 1
                if opp_rank < 30:
                    h2h_bonus += 0.02
            elif res == "L":
                if opp_rank > 75:
                    bad_losses += 1
                if opp_rank > 70:
                    h2h_bonus -= 0.03

    # Balanced, resume-first weighting
    score = (
        0.50 * win_pct +
        0.20 * sos +
        0.10 * (avg_margin / 30.0) +
        0.10 * (off_ppa - def_ppa) +
        0.05 * (off_sr - def_sr) +
        0.05 * (0.02 * quality_wins - 0.02 * bad_losses) +
        h2h_bonus
    )

    # Clamp
    if score < 0.0:
        score = 0.0
    elif score > 1.0:
        score = 1.0
    return score


# ----------------------------- Builder -----------------------------

def build_rankings(year: int):
    print(f"Building rankings for {year}")

    games = get_games(year)
    if not games:
        print("No finished FBS regular-season games found. Writing empty payload.")
        return {"season": year, "last_build_utc": datetime.datetime.utcnow().isoformat(), "top25": []}

    adv = get_advanced(year)
    sp = get_sp_ratings(year)

    # Roll up team records and results
    teams = {}
    for g in games:
        home, away = g["homeTeam"], g["awayTeam"]
        hp, ap = g["homePoints"], g["awayPoints"]

        if hp > ap:
            winner, loser = home, away
        elif ap > hp:
            winner, loser = away, home
        else:
            # Shouldn’t happen in FBS regular with final scores but guard anyway
            continue

        for t in (home, away):
            if t not in teams:
                teams[t] = {
                    "wins": 0, "losses": 0,
                    "points_for": 0, "points_against": 0,
                    "results": []  # list of {"opp": name, "result": "W"/"L", later "opp_rank": int}
                }

        teams[winner]["wins"] += 1
        teams[loser]["losses"] += 1

        teams[home]["points_for"] += safe_float(hp, 0.0)
        teams[home]["points_against"] += safe_float(ap, 0.0)
        teams[away]["points_for"] += safe_float(ap, 0.0)
        teams[away]["points_against"] += safe_float(hp, 0.0)

        teams[winner]["results"].append({"opp": loser, "result": "W"})
        teams[loser]["results"].append({"opp": winner, "result": "L"})

    # Merge metrics and compute margins
    for t, info in teams.items():
        games_played = int(info["wins"]) + int(info["losses"])
        if games_played > 0:
            avg_margin = (safe_float(info["points_for"]) - safe_float(info["points_against"])) / games_played
        else:
            avg_margin = 0.0
        info["avg_margin"] = avg_margin

        # Attach adv + sos with safe coercion
        adv_row = adv.get(t, {})
        info["off_ppa"] = safe_float(adv_row.get("off_ppa"), 0.0)
        info["def_ppa"] = safe_float(adv_row.get("def_ppa"), 0.0)
        info["off_sr"]  = safe_float(adv_row.get("off_sr"), 0.0)
        info["def_sr"]  = safe_float(adv_row.get("def_sr"), 0.0)

        sp_row = sp.get(t, {})
        info["sos"] = safe_float(sp_row.get("sos"), 0.5)

    # Pass 1: provisional scores (no opponent ranks)
    scores_pass1 = {t: compute_team_score(info, info, info["results"], opp_ranks_known=False)
                    for t, info in teams.items()}
    ranked_pass1 = sorted(scores_pass1.items(), key=lambda x: x[1], reverse=True)
    rank_index = {team: idx + 1 for idx, (team, _) in enumerate(ranked_pass1)}

    # Attach opponent ranks to results
    for t, info in teams.items():
        for r in info["results"]:
            r["opp_rank"] = int(rank_index.get(r["opp"], 999))

    # Pass 2: final scores with quality/bad-loss/h2h
    final_scores = {t: compute_team_score(info, info, info["results"], opp_ranks_known=True)
                    for t, info in teams.items()}
    final_sorted = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

    # Build Top 25 output
    top25 = []
    for i, (team, score) in enumerate(final_sorted[:25], start=1):
        info = teams[team]
        top25.append({
            "rank": i,
            "team": team,
            "score": round(safe_float(score), 4),
            "games": int(info["wins"]) + int(info["losses"]),
            "wins": int(info["wins"]),
            "losses": int(info["losses"]),
            "points_for": int(safe_float(info["points_for"], 0.0)),
            "points_against": int(safe_float(info["points_against"], 0.0)),
            "sos": round(safe_float(info.get("sos"), 0.5), 3),
            "avg_margin": round(safe_float(info.get("avg_margin"), 0.0), 1)
        })

    return {
        "season": year,
        "last_build_utc": datetime.datetime.utcnow().isoformat(),
        "top25": top25
    }


# ----------------------------- Main -----------------------------

def main():
    data = build_rankings(YEAR)
    with open(OUT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Rankings written to {OUT_FILE}")

if __name__ == "__main__":
    main()
