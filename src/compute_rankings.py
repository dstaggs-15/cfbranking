#!/usr/bin/env python3
"""
Deterministic CFB Top 25 (FBS-only ranking set)
- Records count ALL games involving an FBS team (FBS vs FCS included).
- Résumé (SoS, quality wins, bad losses, head-to-head) counts ONLY FBS vs FBS.
- Primary order: LOSS BUCKETS (fewest losses first), then:
    1) Strict head-to-head inside same-loss bucket
    2) Composite score (Win%, SoS, Qual Wins, Bad Losses, Avg Margin, tiny adv spice)
    3) FBS wins, then alphabetical
Outputs per-team advanced raw fields for the frontend to compute z-scores.
"""

import os, json, datetime, requests
from collections import defaultdict

CFBD_API_KEY = os.getenv("CFBD_API_KEY")
if not CFBD_API_KEY:
    raise RuntimeError("CFBD_API_KEY is not set")

def _safe_year():
    try:
        return int(os.getenv("YEAR") or datetime.datetime.now().year)
    except Exception:
        return datetime.datetime.now().year

YEAR = _safe_year()
HEADERS = {"Authorization": f"Bearer {CFBD_API_KEY}"}

DATA_DIR = "docs/data"
os.makedirs(DATA_DIR, exist_ok=True)
OUT_FILE = os.path.join(DATA_DIR, "rankings.json")

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)

def fetch(endpoint: str, params=None):
    url = f"https://api.collegefootballdata.com/{endpoint}"
    r = requests.get(url, headers=HEADERS, params=params or {}, timeout=45)
    r.raise_for_status()
    return r.json()

def get_fbs_set(year: int) -> set:
    rows = fetch("teams/fbs", {"year": year})
    return {row.get("school") for row in rows if row.get("school")}

def get_regular_games(year: int):
    rows = fetch("games", {"year": year, "seasonType": "regular"})
    out = []
    for g in rows:
        hp = g.get("homePoints")
        ap = g.get("awayPoints")
        if hp is None or ap is None:
            continue
        out.append({
            "week": g.get("week"),
            "home": g.get("homeTeam"),
            "away": g.get("awayTeam"),
            "home_conf": g.get("homeConference"),
            "away_conf": g.get("awayConference"),
            "hp": safe_float(hp),
            "ap": safe_float(ap),
        })
    return out

def get_season_advanced(year: int) -> dict:
    """Light advanced metrics; we expose raw values to the frontend."""
    try:
        rows = fetch("stats/season/advanced", {"year": year})
    except Exception:
        return {}
    out = {}
    for r in rows:
        t = r.get("team")
        if not t: 
            continue
        off = r.get("offense") or {}
        de  = r.get("defense") or {}
        out[t] = {
            "off_ppa": safe_float(off.get("ppa"), 0.0),
            "def_ppa": safe_float(de.get("ppa"), 0.0),
            "off_sr":  safe_float(off.get("successRate"), 0.0),
            "def_sr":  safe_float(de.get("successRate"), 0.0),
        }
    return out

def rollup(games: list, fbs_set: set):
    teams = {}
    for g in games:
        h, a, hp, ap = g["home"], g["away"], g["hp"], g["ap"]
        if not h or not a:
            continue

        for t in (h, a):
            if t in fbs_set and t not in teams:
                teams[t] = {
                    "wins": 0, "losses": 0, "pf": 0.0, "pa": 0.0,
                    "fbs_wins": 0, "fbs_losses": 0,
                    "fbs_opps": set(),
                    "results_fbs": [],
                    "results_all": []
                }

        if h in teams:
            teams[h]["pf"] += hp
            teams[h]["pa"] += ap
            if hp > ap:
                teams[h]["wins"] += 1
                teams[h]["results_all"].append(("W", a))
            elif ap > hp:
                teams[h]["losses"] += 1
                teams[h]["results_all"].append(("L", a))

        if a in teams:
            teams[a]["pf"] += ap
            teams[a]["pa"] += hp
            if ap > hp:
                teams[a]["wins"] += 1
                teams[a]["results_all"].append(("W", h))
            elif hp > ap:
                teams[a]["losses"] += 1
                teams[a]["results_all"].append(("L", h))

        if h in teams and a in teams:
            teams[h]["fbs_opps"].add(a)
            teams[a]["fbs_opps"].add(h)
            if hp > ap:
                teams[h]["fbs_wins"] += 1
                teams[a]["fbs_losses"] += 1
                teams[h]["results_fbs"].append(("W", a))
                teams[a]["results_fbs"].append(("L", h))
            elif ap > hp:
                teams[a]["fbs_wins"] += 1
                teams[h]["fbs_losses"] += 1
                teams[a]["results_fbs"].append(("W", h))
                teams[h]["results_fbs"].append(("L", a))
    return teams

def compute_sos(teams: dict):
    fbs_wpct = {}
    for t, d in teams.items():
        g = d["fbs_wins"] + d["fbs_losses"]
        fbs_wpct[t] = (d["fbs_wins"] / g) if g > 0 else 0.5
    for t, d in teams.items():
        opps = [o for o in d["fbs_opps"] if o in teams]
        d["sos"] = (sum(fbs_wpct[o] for o in opps) / len(opps)) if opps else 0.5

def composite_score(d: dict, qual_w: int, bad_l: int, adv: dict):
    games = d["wins"] + d["losses"]
    win_pct = (d["wins"] / games) if games > 0 else 0.0
    sos = d.get("sos", 0.5)
    avg_margin = ((d["pf"] - d["pa"]) / games) if games > 0 else 0.0

    # Tiny advanced “spice”—doesn’t dominate
    delta_ppa = (adv.get("off_ppa", 0.0) - adv.get("def_ppa", 0.0))
    delta_sr  = (adv.get("off_sr", 0.0)  - adv.get("def_sr", 0.0))
    adv_term = 0.02 * max(-1.0, min(1.0, delta_ppa)) + 0.01 * max(-1.0, min(1.0, delta_sr))

    qw_term = 0.015 * qual_w
    bl_term = -0.02 * bad_l
    weak_sos_pen = -0.06 * max(0.0, 0.52 - sos)

    score = (
        0.50 * win_pct +
        0.22 * sos +
        0.10 * (avg_margin / 28.0) +
        qw_term + bl_term +
        adv_term +
        weak_sos_pen
    )
    return max(0.0, min(1.0, score))

def build_rankings(year: int):
    print(f"Building deterministic FBS rankings for {year}")
    fbs_set = get_fbs_set(year)
    games = get_regular_games(year)
    if not fbs_set or not games:
        return {"season": year, "last_build_utc": datetime.datetime.utcnow().isoformat(), "top25": []}

    teams = rollup(games, fbs_set)
    if not teams:
        return {"season": year, "last_build_utc": datetime.datetime.utcnow().isoformat(), "top25": []}

    compute_sos(teams)
    adv_all = get_season_advanced(year)

    # Seed order to classify quality/bad via provisional ranks
    seed = []
    for t, d in teams.items():
        games_cnt = d["wins"] + d["losses"]
        win_pct = (d["wins"] / games_cnt) if games_cnt > 0 else 0.0
        seed.append((t, 0.7*win_pct + 0.3*d.get("sos", 0.5)))
    seed.sort(key=lambda x: x[1], reverse=True)
    seed_rank = {t: i+1 for i, (t, _) in enumerate(seed)}

    qual = defaultdict(int)
    bad  = defaultdict(int)
    for t, d in teams.items():
        for res, opp in d["results_fbs"]:
            r = seed_rank.get(opp, 999)
            if res == "W":
                if r <= 40:
                    qual[t] += 1
            elif res == "L":
                if r >= 80:
                    bad[t] += 1

    rows = []
    for t, d in teams.items():
        s = composite_score(d, qual[t], bad[t], adv_all.get(t, {}))
        rows.append({
            "team": t,
            "wins": d["wins"],
            "losses": d["losses"],
            "games": d["wins"] + d["losses"],
            "points_for": int(d["pf"]),
            "points_against": int(d["pa"]),
            "sos": round(d.get("sos", 0.5), 6),
            "avg_margin": round(((d["pf"] - d["pa"]) / max(1, d["wins"] + d["losses"])), 3),
            "fbs_wins": d["fbs_wins"],
            "fbs_losses": d["fbs_losses"],
            "qual_wins": qual[t],
            "bad_losses": bad[t],
            "score": round(s, 6),
            # expose raw adv for frontend z-scores
            "off_ppa": adv_all.get(t, {}).get("off_ppa", 0.0),
            "def_ppa": adv_all.get(t, {}).get("def_ppa", 0.0),
            "off_sr":  adv_all.get(t, {}).get("off_sr", 0.0),
            "def_sr":  adv_all.get(t, {}).get("def_sr", 0.0),
        })

    # Loss buckets then composite within bucket
    rows.sort(key=lambda r: (r["losses"], -r["score"], -r["fbs_wins"], r["team"]))

    # Strict same-loss H2H inside bucket
    results_map = {t: dict() for t in teams.keys()}
    for t, d in teams.items():
        for res, opp in d["results_fbs"]:
            results_map[t][opp] = res

    changed = True
    while changed:
        changed = False
        for i in range(len(rows)):
            A = rows[i]["team"]; lossA = rows[i]["losses"]
            for j in range(i+1, len(rows)):
                if rows[j]["losses"] != lossA:
                    continue
                B = rows[j]["team"]
                # if A lost to B, swap B ahead of A
                if results_map.get(A, {}).get(B) == "L":
                    rows[i], rows[j] = rows[j], rows[i]
                    changed = True
                    break
            if changed: break

    for i, r in enumerate(rows, start=1):
        r["rank"] = i

    top25 = [{
        "rank": r["rank"],
        "team": r["team"],
        "score": round(r["score"], 4),
        "games": r["games"],
        "wins": r["wins"],
        "losses": r["losses"],
        "points_for": r["points_for"],
        "points_against": r["points_against"],
        "sos": round(r["sos"], 3),
        "avg_margin": round(r["avg_margin"], 1),
        "qual_wins": r["qual_wins"],
        "bad_losses": r["bad_losses"],
        # advanced raw for UI z-scores:
        "off_ppa": r["off_ppa"],
        "def_ppa": r["def_ppa"],
        "off_sr": r["off_sr"],
        "def_sr": r["def_sr"]
    } for r in rows[:25]]

    return {
        "season": year,
        "last_build_utc": datetime.datetime.utcnow().isoformat(),
        "notes": {
            "method": "Loss buckets > in-bucket head-to-head > composite (Win%, SoS, Quality Wins, Bad Losses, Margin). Records include FCS games; résumé counts only FBS vs FBS.",
            "weeks_included": "regular season to date",
            "weights": {"win_pct": 0.50, "sos": 0.22, "margin": 0.10, "qual_win_each": 0.015, "bad_loss_each": -0.02}
        },
        "top25": top25
    }

def main():
    data = build_rankings(YEAR)
    with open(OUT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Wrote {OUT_FILE}")

if __name__ == "__main__":
    main()
