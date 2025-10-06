#!/usr/bin/env python3
"""
Deterministic CFB Top 25 (FBS-only ranking set)
- Records count ALL games involving an FBS team (FBS vs FCS included) so records look normal.
- Résumé (SoS, quality wins, bad losses, H2H) counts ONLY FBS vs FBS.
- Primary ordering by LOSS BUCKETS (fewest losses first), then tie-breaking:
  1) Head-to-head (strict, same-loss bucket)
  2) Composite resume score: Win%, SoS, Quality Wins, Bad Losses, Avg Margin
  3) Secondary tiebreaks: FBS wins, then alphabetical for stability
"""

import os, json, datetime, requests, math
from collections import defaultdict

# ---------- Env ----------
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

# ---------- IO ----------
DATA_DIR = "docs/data"
os.makedirs(DATA_DIR, exist_ok=True)
OUT_FILE = os.path.join(DATA_DIR, "rankings.json")

# ---------- Helpers ----------
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

# ---------- Data pulls ----------
def get_fbs_set(year: int) -> set:
    # CFBD: teams/fbs returns current FBS schools for the season
    rows = fetch("teams/fbs", {"year": year})
    return {row.get("school") for row in rows if row.get("school")}

def get_regular_games(year: int):
    # seasonType=regular; include all divisions to keep FCS for record,
    # but we’ll classify later using FBS set.
    rows = fetch("games", {"year": year, "seasonType": "regular"})
    # keep finished only
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

# optional adv seasoning — super light, just for tie spice (not king)
def get_season_advanced(year: int) -> dict:
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

# ---------- Rollup ----------
def rollup(games: list, fbs_set: set):
    """
    records: include any game where team is FBS (opponent can be FCS)
    resume inputs (SoS, quality wins/losses): FBS vs FBS only
    """
    teams = {}
    for g in games:
        h, a, hp, ap = g["home"], g["away"], g["hp"], g["ap"]
        if not h or not a: 
            continue

        # init if FBS teams appear
        for t in (h, a):
            if t in fbs_set and t not in teams:
                teams[t] = {
                    "wins": 0, "losses": 0, "pf": 0.0, "pa": 0.0,
                    "fbs_wins": 0, "fbs_losses": 0,
                    "fbs_opps": set(),  # FBS-only opponents (for SoS)
                    "results_fbs": [],  # list of ("W"/"L", opp)
                    "results_all": []   # for record display only
                }

        # update record if FBS team participated
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

        # résumé only if BOTH are FBS
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
    """
    SoS = average opponent FBS win% over FBS games (0..1).
    If an opponent hasn't played many FBS games yet, it will float near 0.5.
    """
    fbs_wpct = {}
    for t, d in teams.items():
        fbs_g = d["fbs_wins"] + d["fbs_losses"]
        fbs_wpct[t] = (d["fbs_wins"] / fbs_g) if fbs_g > 0 else 0.5

    for t, d in teams.items():
        opps = [o for o in d["fbs_opps"] if o in teams]
        if not opps:
            d["sos"] = 0.5
        else:
            d["sos"] = sum(fbs_wpct[o] for o in opps) / len(opps)

# ---------- Scoring ----------
def composite_score(d: dict, qual_w: int, bad_l: int, adv: dict):
    """
    Transparent score. Ranges roughly 0..1. Weights are sane and football-y.
    """
    games = d["wins"] + d["losses"]
    win_pct = (d["wins"] / games) if games > 0 else 0.0
    sos = d.get("sos", 0.5)
    avg_margin = ((d["pf"] - d["pa"]) / games) if games > 0 else 0.0

    # small seasoning from advanced deltas; capped
    z = adv or {}
    delta_ppa = safe_float(z.get("off_ppa"), 0.0) - safe_float(z.get("def_ppa"), 0.0)
    delta_sr  = safe_float(z.get("off_sr"), 0.0)  - safe_float(z.get("def_sr"), 0.0)
    adv_term = 0.02 * max(-1.0, min(1.0, delta_ppa)) + 0.01 * max(-1.0, min(1.0, delta_sr))

    # quality & bad based only on FBS opponents (qual_w/bad_l already computed)
    qw_term = 0.015 * qual_w
    bl_term = -0.02 * bad_l

    # weak SoS nudge down (not huge, but stops cupcake schedules)
    weak_sos_pen = -0.06 * max(0.0, 0.52 - sos)

    score = (
        0.50 * win_pct +
        0.22 * sos +
        0.10 * (avg_margin / 28.0) +  # normalize ~ +/- 28 ppg
        qw_term + bl_term +
        adv_term +
        weak_sos_pen
    )
    # clamp
    return max(0.0, min(1.0, score))

def tier_from_rank(r: int):
    # Used after provisional ordering within bucket
    if r <= 15: return "Q1"
    if r <= 40: return "Q2"
    if r <= 60: return "Q3"
    if r <= 80: return "B1"  # not used directly, but kept for clarity
    return "B2"

# ---------- Ranking pipeline ----------
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

    # ---- Provisional in-bucket ordering using composite score with provisional opp value ----
    # First, make a simple provisional rank map by raw (win%, sos), to seed quality/bad counts.
    seed = []
    for t, d in teams.items():
        games = d["wins"] + d["losses"]
        win_pct = (d["wins"] / games) if games > 0 else 0.0
        seed.append((t, 0.7*win_pct + 0.3*d.get("sos", 0.5)))
    seed.sort(key=lambda x: x[1], reverse=True)
    seed_rank = {t: i+1 for i, (t, _) in enumerate(seed)}

    # compute quality wins / bad losses (FBS only) using seed ranks
    qual = defaultdict(int)
    bad  = defaultdict(int)
    for t, d in teams.items():
        for res, opp in d["results_fbs"]:
            r = seed_rank.get(opp, 999)
            if res == "W":
                if r <= 40:  # Q1/Q2 threshold
                    qual[t] += 1
            elif res == "L":
                if r >= 80:
                    bad[t] += 1

    # score rows
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
            "score": round(s, 6)
        })

    # ---- Loss buckets primary ----
    rows.sort(key=lambda r: (r["losses"], -r["score"], -r["fbs_wins"], r["team"]))
    # rank numbers will be assigned at the end
    # ---- Strict head-to-head inside same-loss buckets ----
    # build quick lookup of FBS results
    results_map = {t: dict() for t in teams.keys()}
    for t, d in teams.items():
        for res, opp in d["results_fbs"]:
            results_map[t][opp] = res  # "W" or "L"

    idx = {r["team"]: i for i, r in enumerate(rows)}
    changed = True
    while changed:
        changed = False
        for i in range(len(rows)):
            A = rows[i]["team"]
            lossA = rows[i]["losses"]
            for j in range(i+1, len(rows)):
                B = rows[j]["team"]
                if rows[j]["losses"] != lossA:
                    continue
                # If A lost to B within same-loss bucket, swap
                if results_map.get(A, {}).get(B) == "L":
                    rows[i], rows[j] = rows[j], rows[i]
                    # rebuild idx for stability
                    idx = {r["team"]: k for k, r in enumerate(rows)}
                    changed = True
                    break
            if changed:
                break

    # Final rank assign and cut Top 25
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

# ---------- Main ----------
def main():
    data = build_rankings(YEAR)
    with open(OUT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Wrote {OUT_FILE}")

if __name__ == "__main__":
    main()
