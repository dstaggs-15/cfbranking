# src/compute_rankings.py
import os, json, requests, statistics, datetime

# Get the CFBD API key from environment variable (GitHub secret)
CFBD_KEY = os.getenv("CFBD_API_KEY")
if not CFBD_KEY:
    raise RuntimeError("Missing CFBD_API_KEY environment variable.")

YEAR = int(os.getenv("YEAR", datetime.datetime.now().year))
API_BASE = "https://api.collegefootballdata.com"

HEADERS = {"Authorization": f"Bearer {CFBD_KEY}"}

def get_json(endpoint, params=None):
    url = f"{API_BASE}/{endpoint}"
    resp = requests.get(url, headers=HEADERS, params=params)
    resp.raise_for_status()
    return resp.json()

def get_team_stats():
    data = get_json("stats/season/advanced", {"year": YEAR})
    teams = {}
    for row in data:
        team = row["team"]
        off = row.get("offense") or {}
        deff = row.get("defense") or {}
        teams[team] = {
            "off_ppa": off.get("ppa", 0),
            "off_success": off.get("successRate", 0),
            "off_explosiveness": off.get("explosiveness", 0),
            "off_pts_per_opp": off.get("pointsPerOpportunity", 0),
            "def_ppa": deff.get("ppa", 0),
            "def_success": deff.get("successRate", 0),
            "def_explosiveness": deff.get("explosiveness", 0),
            "def_pts_per_opp": deff.get("pointsPerOpportunity", 0),
        }
    return teams

def get_strength_of_schedule():
    data = get_json("ratings/strength", {"year": YEAR})
    return {row["team"]: row["strengthOfSchedule"] for row in data}

def get_records():
    data = get_json("records", {"year": YEAR})
    return {r["team"]: (r["total"]["wins"], r["total"]["losses"]) for r in data}

def build_rankings():
    print("Building rankings for", YEAR)
    stats = get_team_stats()
    sos = get_strength_of_schedule()
    recs = get_records()

    results = []
    for team, s in stats.items():
        perf_score = (
            s["off_ppa"] * 40 +
            s["off_success"] * 25 +
            s["off_explosiveness"] * 10 +
            (1 - s["def_ppa"]) * 25
        )
        sos_score = sos.get(team, 50)
        wins, losses = recs.get(team, (0, 0))
        games = wins + losses
        win_pct = wins / games if games else 0
        final = 0.45 * perf_score + 0.35 * sos_score + 0.2 * (win_pct * 100)
        results.append({
            "team": team,
            "performance": round(perf_score, 2),
            "sos": round(sos_score, 2),
            "win_pct": round(win_pct, 3),
            "final_score": round(final, 2),
            "record": f"{wins}-{losses}",
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    for i, t in enumerate(results[:25], start=1):
        t["rank"] = i

    out = {
        "season": YEAR,
        "last_build_utc": datetime.datetime.utcnow().isoformat(),
        "top25": results[:25],
    }

    os.makedirs("docs/data", exist_ok=True)
    with open("docs/data/rankings.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote docs/data/rankings.json")

if __name__ == "__main__":
    build_rankings()
