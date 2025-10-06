import os
import json
import datetime
import math
import statistics
import time
import requests

API_BASE = "https://api.collegefootballdata.com"
API_KEY = os.getenv("CFBD_API_KEY", "")
YEAR = int(os.getenv("YEAR") or datetime.datetime.now().year)

# ---------- HTTP helpers ----------

def cfbd_get(path, params=None, max_retries=3, backoff=1.5):
    """
    GET wrapper for CFBD REST v2 with Bearer auth.
    """
    if not API_KEY:
        raise RuntimeError("CFBD_API_KEY is not set in the environment.")
    url = f"{API_BASE}/{path.lstrip('/')}"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    params = params or {}

    for attempt in range(1, max_retries + 1):
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code == 200:
            try:
                return resp.json()
            except Exception:
                print("⚠️ JSON decode error; response begins:", resp.text[:200])
                return []
        # transient blocks (Cloudflare etc.)
        if resp.status_code in (429, 502, 503, 504):
            sleep_s = backoff ** attempt
            print(f"⚠️ {resp.status_code} from {url}. Retrying in {sleep_s:.1f}s...")
            time.sleep(sleep_s)
            continue
        # hard failures — show payload for debugging
        print(f"⚠️ CFBD {resp.status_code} from {url} :: {resp.text[:300]}")
        return []
    return []

# ---------- Data fetch ----------

def fetch_finished_games(year: int):
    """
    Use the v2 /games endpoint with minimal filters and rely on the
    'completed' flag. Do NOT pass division/classification to avoid
    empty responses if the param surface shifts.
    """
    params = {
        "year": year,
        "seasonType": "regular",   # v2 still accepts this casing
        # intentionally not passing week or division/classification
    }
    data = cfbd_get("games", params=params)
    if not isinstance(data, list):
        print("⚠️ Unexpected games payload type.")
        return []

    # finished games = completed == True (more reliable than points != None)
    finished = [g for g in data if g.get("completed") is True]
    print(f"Fetched {len(data)} games; {len(finished)} marked completed.")
    return finished

# ---------- Metrics ----------

def _init_team(teams, name):
    if name not in teams:
        teams[name] = {
            "games": 0,
            "wins": 0,
            "losses": 0,
            "pf": 0,
            "pa": 0,
            "sos": 0.0,
            "opp_set": set(),
        }

def compute_team_rollups(games):
    teams = {}
    for g in games:
        home = g.get("home_team") or g.get("homeTeam") or g.get("home")
        away = g.get("away_team") or g.get("awayTeam") or g.get("away")
        hp = g.get("home_points") if "home_points" in g else g.get("homePoints")
        ap = g.get("away_points") if "away_points" in g else g.get("awayPoints")

        if not home or not away:
            # defensive: skip malformed rows
            continue

        _init_team(teams, home)
        _init_team(teams, away)

        teams[home]["games"] += 1
        teams[away]["games"] += 1
        if hp is not None and ap is not None:
            teams[home]["pf"] += hp
            teams[home]["pa"] += ap
            teams[away]["pf"] += ap
            teams[away]["pa"] += hp
            if hp > ap:
                teams[home]["wins"] += 1
                teams[away]["losses"] += 1
            elif ap > hp:
                teams[away]["wins"] += 1
                teams[home]["losses"] += 1
            # ties are extremely rare but implicitly handled (no win counted)

        teams[home]["opp_set"].add(away)
        teams[away]["opp_set"].add(home)
    return teams

def compute_sos(teams):
    # Simple SoS = mean opponent win% (only opponents faced so far)
    for t, d in teams.items():
        opps = list(d["opp_set"])
        if not opps:
            d["sos"] = 0.0
            continue
        opp_wpcts = []
        for o in opps:
            if o not in teams or teams[o]["games"] == 0:
                continue
            opp_wpcts.append(teams[o]["wins"] / max(1, teams[o]["games"]))
        d["sos"] = statistics.mean(opp_wpcts) if opp_wpcts else 0.0

def score_teams(teams):
    scored = []
    for t, d in teams.items():
        if d["games"] < 1:
            continue
        win_pct = d["wins"] / d["games"]
        avg_margin = (d["pf"] - d["pa"]) / max(1, d["games"])
        sos = d["sos"]

        # Football-ish composite:
        #  - Win pct: 0.55
        #  - SoS:     0.30
        #  - Margin:  0.15 (scaled by ~25 pts per game)
        score = (0.55 * win_pct) + (0.30 * sos) + (0.15 * (avg_margin / 25.0))

        scored.append({
            "team": t,
            "score": round(score, 6),
            "games": d["games"],
            "wins": d["wins"],
            "losses": d["losses"],
            "points_for": d["pf"],
            "points_against": d["pa"],
            "sos": round(sos, 6),
            "avg_margin": round(avg_margin, 3),
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:25]

# ---------- Main ----------

def write_json(out):
    os.makedirs("docs/data", exist_ok=True)
    with open("docs/data/rankings.json", "w") as f:
        json.dump(out, f, indent=2)

def main():
    print(f"Building rankings for {YEAR}")
    games = fetch_finished_games(YEAR)

    if not games:
        print("No finished games returned by API. Writing placeholder JSON so site still loads.")
        write_json({
            "season": YEAR,
            "last_build_utc": datetime.datetime.utcnow().isoformat(),
            "top25": [],
            "note": "No completed games were returned by CFBD for the requested filters. This will auto-populate once results are available."
        })
        return

    teams = compute_team_rollups(games)
    compute_sos(teams)
    top25 = score_teams(teams)

    out = {
        "season": YEAR,
        "last_build_utc": datetime.datetime.utcnow().isoformat(),
        "top25": top25,
    }
    write_json(out)
    print(f"✅ Built Top 25 with {len(top25)} teams.")

if __name__ == "__main__":
    main()
