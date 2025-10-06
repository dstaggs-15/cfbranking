import os
import json
import datetime
import time
import statistics
from typing import Dict, List, Set

import requests

API_BASE = "https://api.collegefootballdata.com"
API_KEY = os.getenv("CFBD_API_KEY", "")
YEAR = int(os.getenv("YEAR") or datetime.datetime.now().year)

# ---------------- HTTP ----------------

def cfbd_get(path: str, params: dict | None = None, tries: int = 3, backoff: float = 1.6):
    if not API_KEY:
        raise RuntimeError("CFBD_API_KEY is not set.")
    url = f"{API_BASE}/{path.lstrip('/')}"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    params = params or {}
    last_text = ""
    code = 0
    for a in range(tries):
        r = requests.get(url, headers=headers, params=params, timeout=40)
        code = r.status_code
        if code == 200:
            try:
                return r.json()
            except Exception:
                print("⚠️ JSON parse error; head:", r.text[:200])
                return []
        if code in (429, 502, 503, 504):
            sleep_s = backoff ** (a + 1)
            print(f"⚠️ {code} on {path} {params} — retry in {sleep_s:.1f}s")
            time.sleep(sleep_s)
            continue
        last_text = r.text
        break
    if last_text:
        print(f"⚠️ CFBD {code} {path} {params} :: {last_text[:300]}")
    return []

# --------------- FETCH HELPERS ----------------

def fetch_fbs_names(year: int) -> Set[str]:
    data = cfbd_get("teams/fbs", {"year": year})
    names = {row.get("school") for row in (data or []) if row.get("school")}
    print(f"[FBS] {len(names)} teams")
    return names

def fetch_current_week(year: int) -> int:
    """
    Use /calendar to find the max regular-season week that has started.
    Fallback to 20 if calendar not available.
    """
    data = cfbd_get("calendar", {"year": year})
    if not isinstance(data, list) or not data:
        print("[calendar] unavailable; fallback to 20 weeks")
        return 20
    now = datetime.datetime.utcnow()
    weeks = []
    for row in data:
        if int(row.get("season", 0)) != int(year):
            continue
        st = (row.get("season_type") or row.get("seasonType") or "").lower()
        if st != "regular":
            continue
        wk = int(row.get("week") or 0)
        start_str = row.get("first_game_start") or row.get("firstGameStart") or row.get("firstGameStartTime")
        if start_str:
            try:
                start = datetime.datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                if start <= now:
                    weeks.append(wk)
            except Exception:
                weeks.append(wk)
        else:
            weeks.append(wk)
    cur = max(weeks) if weeks else 20
    print(f"[calendar] current regular-season week guessed = {cur}")
    return cur

def is_finished(g: dict) -> bool:
    # robust finished detection
    if g.get("completed") is True:
        return True
    hp = g.get("home_points") if "home_points" in g else g.get("homePoints")
    ap = g.get("away_points") if "away_points" in g else g.get("awayPoints")
    return (hp is not None and ap is not None)

def norm(g: dict) -> dict:
    return {
        "home_team": g.get("home_team") or g.get("homeTeam") or g.get("home"),
        "away_team": g.get("away_team") or g.get("awayTeam") or g.get("away"),
        "home_points": g.get("home_points") if "home_points" in g else g.get("homePoints"),
        "away_points": g.get("away_points") if "away_points" in g else g.get("awayPoints"),
    }

def fetch_regular_games_by_week(year: int, up_to_week: int) -> List[dict]:
    all_games: List[dict] = []
    total_finished = 0
    for wk in range(1, up_to_week + 1):
        params = {"year": year, "seasonType": "regular", "week": wk}
        data = cfbd_get("games", params)
        if not isinstance(data, list):
            print(f"[games] week {wk}: unexpected payload")
            continue
        fin = [norm(g) for g in data if is_finished(g)]
        total_finished += len(fin)
        all_games.extend(fin)
        print(f"[games] week={wk} -> pulled={len(data)} finished={len(fin)}")
    print(f"[games] aggregated finished={total_finished}")
    return all_games

# --------------- FILTERING POLICY ----------------
# FBS teams only, but COUNT ALL THEIR GAMES (including vs non-FBS).
# For SoS, only FBS opponents contribute.

def keep_games_with_fbs_team(games: List[dict], fbs: Set[str]) -> List[dict]:
    kept = [g for g in games if (g["home_team"] in fbs) or (g["away_team"] in fbs)]
    print(f"[filter] games with at least one FBS team kept={len(kept)}")
    return kept

# --------------- METRICS ----------------

def roll_up(games: List[dict], fbs: Set[str]) -> Dict[str, dict]:
    teams: Dict[str, dict] = {}
    def init(t: str):
        if t not in teams:
            teams[t] = {"games": 0, "wins": 0, "losses": 0, "pf": 0, "pa": 0, "opps": set()}
    for g in games:
        h, a = g["home_team"], g["away_team"]
        hp, ap = int(g["home_points"] or 0), int(g["away_points"] or 0)

        # Only track records for FBS teams
        if h in fbs:
            init(h)
            teams[h]["games"] += 1
            teams[h]["pf"] += hp
            teams[h]["pa"] += ap
            if hp > ap: teams[h]["wins"] += 1
            elif ap > hp: teams[h]["losses"] += 1
            teams[h]["opps"].add(a)  # may be non-FBS

        if a in fbs:
            init(a)
            teams[a]["games"] += 1
            teams[a]["pf"] += ap
            teams[a]["pa"] += hp
            if ap > hp: teams[a]["wins"] += 1
            elif hp > ap: teams[a]["losses"] += 1
            teams[a]["opps"].add(h)  # may be non-FBS
    return teams

def compute_sos_fbs_only(teams: Dict[str, dict], fbs: Set[str]) -> None:
    # SoS = mean opponent Win% among FBS opponents only
    for t, d in teams.items():
        opps = [o for o in d["opps"] if o in fbs]  # ignore non-FBS for SoS
        wpcts = []
        for o in opps:
            if o not in teams or teams[o]["games"] == 0:
                continue
            wpcts.append(teams[o]["wins"] / teams[o]["games"])
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
    for i, r in enumerate(out, start=1):
        r["rank"] = i
    return out[:25]

# --------------- IO ----------------

def write_json(payload: dict):
    os.makedirs("docs/data", exist_ok=True)
    with open("docs/data/rankings.json", "w") as f:
        json.dump(payload, f, indent=2)

# --------------- MAIN ----------------

def main():
    print(f"Building FBS rankings (count all games) for {YEAR}")
    fbs = fetch_fbs_names(YEAR)
    cur_week = fetch_current_week(YEAR)
    games_all = fetch_regular_games_by_week(YEAR, cur_week)

    # keep any game where at least one team is FBS (so records match public 5-0, etc.)
    games = keep_games_with_fbs_team(games_all, fbs)

    if not games:
        print("❌ No completed games found for FBS teams. Writing placeholder.")
        write_json({
            "season": YEAR,
            "last_build_utc": datetime.datetime.utcnow().isoformat(),
            "top25": [],
            "note": "No completed games for FBS teams available."
        })
        return

    teams = roll_up(games, fbs)
    compute_sos_fbs_only(teams, fbs)
    top25 = score_and_top25(teams)

    out = {
        "season": YEAR,
        "last_build_utc": datetime.datetime.utcnow().isoformat(),
        "top25": top25
    }
    write_json(out)
    print(f"✅ Top 25 built from {len(teams)} FBS teams • weeks=1..{cur_week}")

if __name__ == "__main__":
    main()
