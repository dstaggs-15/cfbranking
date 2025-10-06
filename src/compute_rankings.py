import os
import json
import datetime
import math
import statistics
import time
from typing import Dict, List, Any, Set

import requests

API_BASE = "https://api.collegefootballdata.com"
API_KEY = os.getenv("CFBD_API_KEY", "")
YEAR = int(os.getenv("YEAR") or datetime.datetime.now().year)

def cfbd_get(path: str, params: dict | None = None, tries: int = 3, backoff: float = 1.5):
    if not API_KEY:
        raise RuntimeError("CFBD_API_KEY is not set.")
    url = f"{API_BASE}/{path.lstrip('/')}"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    params = params or {}
    for a in range(tries):
        r = requests.get(url, headers=headers, params=params, timeout=30)
        if r.status_code == 200:
            try:
                return r.json()
            except Exception:
                print("⚠️ JSON parse error; payload head:", r.text[:200])
                return []
        if r.status_code in (429, 502, 503, 504):
            sleep_s = backoff ** (a + 1)
            print(f"⚠️ {r.status_code} from {url}. Retry in {sleep_s:.1f}s")
            time.sleep(sleep_s)
            continue
        print(f"⚠️ CFBD {r.status_code}: {r.text[:300]}")
        return []
    return []

def fetch_fbs_names(year: int) -> Set[str]:
    # Official FBS list for the season
    data = cfbd_get("teams/fbs", {"year": year})
    names = {row.get("school") for row in (data or []) if row.get("school")}
    print(f"FBS teams for {year}: {len(names)}")
    return names

def fetch_all_regular_games(year: int) -> List[dict]:
    # Minimal filters; rely on 'completed' flag then filter to FBS later
    data = cfbd_get("games", {"year": year, "seasonType": "regular"})
    if not isinstance(data, list):
        print("⚠️ unexpected /games payload")
        return []
    finished = [g for g in data if g.get("completed") is True]
    # normalize keys used later
    norm = []
    for g in finished:
        norm.append({
            "home_team": g.get("home_team") or g.get("homeTeam") or g.get("home"),
            "away_team": g.get("away_team") or g.get("awayTeam") or g.get("away"),
            "home_points": g.get("home_points") if "home_points" in g else g.get("homePoints"),
            "away_points": g.get("away_points") if "away_points" in g else g.get("awayPoints"),
        })
    print(f"Fetched {len(data)} games; {len(finished)} completed.")
    return norm

def fbs_only_games(games: List[dict], fbs_names: Set[str]) -> List[dict]:
    kept = [g for g in games if g["home_team"] in fbs_names and g["away_team"] in fbs_names]
    print(f"FBS vs FBS kept: {len(kept)}")
    return kept

def roll_up(games: List[dict]) -> Dict[str, dict]:
    teams: Dict[str, dict] = {}
    def init(t: str):
        if t not in teams:
            teams[t] = {"games": 0, "wins": 0, "losses": 0, "pf": 0, "pa": 0, "opps": set()}
    for g in games:
        h, a = g["home_team"], g["away_team"]
        hp, ap = g["home_points"], g["away_points"]
        if not h or not a: 
            continue
        init(h); init(a)
        teams[h]["games"] += 1
        teams[a]["games"] += 1
        teams[h]["pf"] += int(hp or 0)
        teams[h]["pa"] += int(ap or 0)
        teams[a]["pf"] += int(ap or 0)
        teams[a]["pa"] += int(hp or 0)
        if hp is not None and ap is not None:
            if hp > ap: teams[h]["wins"] += 1; teams[a]["losses"] += 1
            elif ap > hp: teams[a]["wins"] += 1; teams[h]["losses"] += 1
        teams[h]["opps"].add(a); teams[a]["opps"].add(h)
    return teams

def compute_sos(teams: Dict[str, dict]) -> None:
    for t, d in teams.items():
        opps = list(d["opps"])
        wpcts = []
        for o in opps:
            if o not in teams: 
                continue
            g = max(1, teams[o]["games"])
            wpcts.append(teams[o]["wins"] / g)
        d["sos"] = float(statistics.mean(wpcts)) if wpcts else 0.0

def score_and_top25(teams: Dict[str, dict]) -> List[dict]:
    out = []
    for t, d in teams.items():
        if d["games"] < 1:
            continue
        win_pct = d["wins"] / d["games"]
        avg_margin = (d["pf"] - d["pa"]) / max(1, d["games"])
        sos = d.get("sos", 0.0)
        score = 0.55 * win_pct + 0.30 * sos + 0.15 * (avg_margin / 25.0)
        out.append({
            "team": t,
            "wins": d["wins"],
            "losses": d["losses"],
            "games": d["games"],
            "points_for": d["pf"],
            "points_against": d["pa"],
            "sos": round(sos, 6),
            "avg_margin": round(avg_margin, 3),
            "score": round(score, 6),
        })
    out.sort(key=lambda r: r["score"], reverse=True)
    # add ranks 1..N
    for i, row in enumerate(out, start=1):
        row["rank"] = i
    return out[:25]

def write_json(payload: dict):
    os.makedirs("docs/data", exist_ok=True)
    with open("docs/data/rankings.json", "w") as f:
        json.dump(payload, f, indent=2)

def main():
    print(f"Building FBS-only rankings for {YEAR}")
    fbs_names = fetch_fbs_names(YEAR)
    all_games = fetch_all_regular_games(YEAR)
    games = fbs_only_games(all_games, fbs_names)

    if not games:
        print("❌ No completed FBS vs FBS games returned. Writing placeholder so site loads.")
        write_json({
            "season": YEAR,
            "last_build_utc": datetime.datetime.utcnow().isoformat(),
            "top25": [],
            "note": "No completed FBS vs FBS games available yet."
        })
        return

    teams = roll_up(games)
    compute_sos(teams)
    top25 = score_and_top25(teams)

    out = {
        "season": YEAR,
        "last_build_utc": datetime.datetime.utcnow().isoformat(),
        "top25": top25
    }
    write_json(out)
    print(f"✅ Built Top 25 from {len(teams)} FBS teams. Wrote docs/data/rankings.json")

if __name__ == "__main__":
    main()
