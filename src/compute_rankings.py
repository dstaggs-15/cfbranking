# src/compute_rankings.py
# ------------------------------------------------------------
# College Football in-season ranking builder (no preseason bias)
# Uses CFBD REST API via requests and a CFBD_API_KEY secret.
# Produces: docs/data/rankings.json
#
# Frontend expects per-team fields:
#   rank, team, conference, record, games, win_pct,
#   final_score, performance, performance_rank,
#   sos, sos_rank, quality_wins,
#   off_ppa, off_success_rate, off_explosiveness, off_pts_per_opp,
#   def_ppa, def_success_rate, def_explosiveness, def_pts_per_opp,
#   slug
# ------------------------------------------------------------

import os
import json
import math
import time
import datetime
import statistics
import requests
from typing import Dict, List, Any, Tuple, Optional

# -------------------- CONFIG / WEIGHTS ----------------------

def get_year() -> int:
    """
    Robust YEAR handling:
    - If env YEAR is numeric, use it
    - Otherwise use the current calendar year
    """
    y = os.getenv("YEAR")
    if y and y.strip().isdigit():
        return int(y)
    return datetime.datetime.now().year

YEAR: int = get_year()

CFBD_KEY = os.getenv("CFBD_API_KEY")
if not CFBD_KEY:
    raise RuntimeError("Missing CFBD_API_KEY (set as a GitHub Actions secret).")

API_BASE = "https://api.collegefootballdata.com"
HEADERS = {"Authorization": f"Bearer {CFBD_KEY}"}

# Football-centric weights (tune as you like)
WEIGHTS = {
    # Performance composite (z-scored then scaled)
    "offense": {
        "ppa": 0.40,
        "success_rate": 0.30,
        "explosiveness": 0.20,
        "pts_per_opp": 0.10
    },
    "defense": {
        # lower is better for defense metrics, we invert before z-scoring
        "ppa": 0.40,
        "success_rate": 0.30,
        "explosiveness": 0.20,
        "pts_per_opp": 0.10
    },
    # Final blend
    "final": {
        "performance": 0.45,
        "sos": 0.35,
        "quality_wins": 0.20
    },
    "quality_win_scalar": 0.05,  # per-win scaling by opponent quality
    "h2h_threshold": 1.0,        # if teams within threshold, nudge head-to-head winner
    "h2h_nudge": 0.15
}

# -------------------- HTTP HELPERS --------------------------

def _get(endpoint: str, params: Optional[Dict[str, Any]] = None, tries: int = 3) -> Any:
    """
    Basic GET with retries/backoff.
    """
    url = f"{API_BASE}/{endpoint.lstrip('/')}"
    for attempt in range(tries):
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 500, 502, 503, 504):
            # backoff
            time.sleep(1.5 * (attempt + 1))
            continue
        resp.raise_for_status()
    # last attempt raise
    resp.raise_for_status()

# -------------------- DATA FETCH ----------------------------

def fetch_fbs_teams(year: int) -> List[Dict[str, Any]]:
    # GET /teams/fbs?year=YYYY
    data = _get("teams/fbs", {"year": year})
    # normalize to {team, conference}
    out = []
    for t in data:
        out.append({
            "team": t.get("school"),
            "conference": t.get("conference")
        })
    return out

def fetch_games(year: int) -> List[Dict[str, Any]]:
    # GET /games?year=YYYY
    data = _get("games", {"year": year})
    # Keep only FBS vs FBS with final scores present
    good = []
    for g in data:
        hp = g.get("home_points")
        ap = g.get("away_points")
        if hp is None or ap is None:
            continue
        # FBS vs FBS - require conference fields
        if not g.get("home_conference") or not g.get("away_conference"):
            continue
        good.append({
            "week": g.get("week"),
            "date": g.get("start_date"),
            "home_team": g.get("home_team"),
            "home_conf": g.get("home_conference"),
            "home_points": hp,
            "away_team": g.get("away_team"),
            "away_conf": g.get("away_conference"),
            "away_points": ap,
            "neutral_site": bool(g.get("neutral_site")),
        })
    return good

def fetch_advanced_stats(year: int) -> Dict[str, Dict[str, float]]:
    # GET /stats/season/advanced?year=YYYY
    data = _get("stats/season/advanced", {"year": year})
    teams = {}
    for row in data:
        team = row.get("team")
        off = row.get("offense") or {}
        deff = row.get("defense") or {}
        teams[team] = {
            "off_ppa": off.get("ppa", 0.0),
            "off_success_rate": off.get("successRate", 0.0),
            "off_explosiveness": off.get("explosiveness", 0.0),
            "off_pts_per_opp": off.get("pointsPerOpportunity", 0.0),
            "def_ppa": deff.get("ppa", 0.0),
            "def_success_rate": deff.get("successRate", 0.0),
            "def_explosiveness": deff.get("explosiveness", 0.0),
            "def_pts_per_opp": deff.get("pointsPerOpportunity", 0.0),
        }
    return teams

def fetch_records(year: int) -> Dict[str, Tuple[int, int]]:
    # GET /records?year=YYYY
    data = _get("records", {"year": year})
    recs = {}
    for r in data:
        team = r.get("team")
        total = r.get("total") or {}
        wins = int(total.get("wins") or 0)
        losses = int(total.get("losses") or 0)
        recs[team] = (wins, losses)
    return recs

# -------------------- METRIC HELPERS ------------------------

def zscores(values: List[float]) -> List[float]:
    # Standard z-score with ddof=0; guard edge cases
    if not values:
        return []
    mean = statistics.fmean(values)
    # population std
    var = statistics.fmean([(v - mean) ** 2 for v in values])
    std = math.sqrt(var)
    if std == 0:
        return [0.0 for _ in values]
    return [(v - mean) / std for v in values]

def minmax01(values: List[float]) -> List[float]:
    if not values:
        return []
    mn = min(values)
    mx = max(values)
    if mx - mn == 0:
        return [0.5 for _ in values]
    return [(v - mn) / (mx - mn) for v in values]

def scale_0_100(values: List[float]) -> List[float]:
    return [round(v * 100.0, 2) for v in minmax01(values)]

def invert_list(values: List[float]) -> List[float]:
    return [-v if v is not None else None for v in values]

def mean_ignore_none(values: List[Optional[float]]) -> float:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    return float(statistics.fmean(clean)) if clean else 0.0

def slugify_team(name: str) -> str:
    s = (name or "").strip().lower()
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        elif ch.isspace() or ch in "-_":
            out.append("-")
        # else skip punctuation
    # collapse dashes
    slug = []
    last_dash = False
    for ch in out:
        if ch == "-":
            if not last_dash:
                slug.append("-")
            last_dash = True
        else:
            slug.append(ch)
            last_dash = False
    return "".join(slug).strip("-")

# -------------------- PIPELINE ------------------------------

def build_records(games: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    rec = {}
    for g in games:
        home = g["home_team"]; away = g["away_team"]
        hp = g["home_points"]; ap = g["away_points"]
        if home not in rec: rec[home] = {"team": home, "wins": 0, "losses": 0}
        if away not in rec: rec[away] = {"team": away, "wins": 0, "losses": 0}
        if hp > ap:
            rec[home]["wins"] += 1; rec[away]["losses"] += 1
        else:
            rec[away]["wins"] += 1; rec[home]["losses"] += 1
    # finalize
    for t in rec.values():
        games_played = t["wins"] + t["losses"]
        t["games"] = games_played
        t["win_pct"] = (t["wins"] / games_played) if games_played else 0.0
    return rec

def build_schedules(games: List[Dict[str, Any]]) -> Dict[str, set]:
    opps: Dict[str, set] = {}
    for g in games:
        a, b = g["home_team"], g["away_team"]
        opps.setdefault(a, set()).add(b)
        opps.setdefault(b, set()).add(a)
    return opps

def performance_composite(stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Build a performance score from advanced stats:
    - Offense: ppa, success_rate, explosiveness, pts_per_opp (weighted)
    - Defense: same metrics but inverted (lower is better)
    z-score each submetric across FBS, combine, then scale to 0..100
    """
    teams = list(stats.keys())

    off_ppa = [stats[t]["off_ppa"] for t in teams]
    off_sr = [stats[t]["off_success_rate"] for t in teams]
    off_ex = [stats[t]["off_explosiveness"] for t in teams]
    off_ppo = [stats[t]["off_pts_per_opp"] for t in teams]

    def_ppa = invert_list([stats[t]["def_ppa"] for t in teams])
    def_sr = invert_list([stats[t]["def_success_rate"] for t in teams])
    def_ex = invert_list([stats[t]["def_explosiveness"] for t in teams])
    def_ppo = invert_list([stats[t]["def_pts_per_opp"] for t in teams])

    z = {
        "z_off_ppa": zscores(off_ppa),
        "z_off_sr": zscores(off_sr),
        "z_off_ex": zscores(off_ex),
        "z_off_ppo": zscores(off_ppo),
        "z_def_ppa": zscores(def_ppa),
        "z_def_sr": zscores(def_sr),
        "z_def_ex": zscores(def_ex),
        "z_def_ppo": zscores(def_ppo),
    }

    off_w = WEIGHTS["offense"]
    def_w = WEIGHTS["defense"]

    perf_z: List[float] = []
    for i, t in enumerate(teams):
        off_score = (
            off_w["ppa"] * z["z_off_ppa"][i] +
            off_w["success_rate"] * z["z_off_sr"][i] +
            off_w["explosiveness"] * z["z_off_ex"][i] +
            off_w["pts_per_opp"] * z["z_off_ppo"][i]
        )
        def_score = (
            def_w["ppa"] * z["z_def_ppa"][i] +
            def_w["success_rate"] * z["z_def_sr"][i] +
            def_w["explosiveness"] * z["z_def_ex"][i] +
            def_w["pts_per_opp"] * z["z_def_ppo"][i]
        )
        perf_z.append(0.5 * off_score + 0.5 * def_score)

    perf_scaled = scale_0_100(perf_z)
    return {team: perf_scaled[i] for i, team in enumerate(teams)}

def compute_rating_sos(perf: Dict[str, float], schedules: Dict[str, set]) -> Dict[str, float]:
    """
    Rating-based SOS: average of opponents' performance scores.
    """
    sos = {}
    for team, opps in schedules.items():
        opp_scores = [perf.get(o) for o in opps if o in perf]
        sos[team] = mean_ignore_none(opp_scores)
    return sos

def compute_quality_wins(games: List[Dict[str, Any]],
                         perf: Dict[str, float],
                         sos: Dict[str, float]) -> Dict[str, float]:
    """
    For each victory, bonus = scalar * average(opponent's performance, opponent's SOS).
    """
    scalar = WEIGHTS["quality_win_scalar"]
    qw: Dict[str, float] = {}
    for g in games:
        home, away = g["home_team"], g["away_team"]
        hp, ap = g["home_points"], g["away_points"]
        winner = home if hp > ap else away
        loser = away if hp > ap else home
        opp_perf = perf.get(loser)
        opp_sos = sos.get(loser)
        if opp_perf is None or opp_sos is None:
            bonus = 0.0
        else:
            bonus = scalar * ((opp_perf + opp_sos) / 2.0)
        qw[winner] = qw.get(winner, 0.0) + bonus
        qw.setdefault(loser, 0.0)
    return qw

def head_to_head_map(games: List[Dict[str, Any]]) -> Dict[str, set]:
    beaten: Dict[str, set] = {}
    for g in games:
        winner = g["home_team"] if g["home_points"] > g["away_points"] else g["away_team"]
        loser  = g["away_team"] if winner == g["home_team"] else g["home_team"]
        beaten.setdefault(winner, set()).add(loser)
    return beaten

def apply_h2h_nudges(rows: List[Dict[str, Any]], beaten_map: Dict[str, set]) -> List[Dict[str, Any]]:
    """
    If winner and loser are within h2h_threshold in final_score, nudge winner by h2h_nudge.
    """
    threshold = WEIGHTS["h2h_threshold"]
    nudge = WEIGHTS["h2h_nudge"]
    score_map = {r["team"]: r["final_score"] for r in rows}
    bonus: Dict[str, float] = {r["team"]: 0.0 for r in rows}
    for winner, losers in beaten_map.items():
        for loser in losers:
            if winner in score_map and loser in score_map:
                if abs(score_map[winner] - score_map[loser]) <= threshold:
                    bonus[winner] += nudge
    for r in rows:
        r["final_score"] = r["final_score"] + bonus.get(r["team"], 0.0)
    rows.sort(key=lambda x: x["final_score"], reverse=True)
    return rows

# -------------------- MAIN BUILD ----------------------------

def main() -> None:
    print(f"Building rankings for {YEAR}")

    # Fetch data
    fbs = fetch_fbs_teams(YEAR)                # [{team, conference}]
    games = fetch_games(YEAR)                   # FBS vs FBS with scores
    stats = fetch_advanced_stats(YEAR)          # team -> metrics
    records_map = fetch_records(YEAR)           # team -> (wins, losses)

    # Short-circuit if no games yet
    if not games:
        raise SystemExit("No FBS vs FBS games with final scores found for the season.")

    # Build record table
    rec = build_records(games)  # team -> {wins, losses, games, win_pct}

    # Performance composite
    perf = performance_composite(stats)  # team -> 0..100

    # Schedules + SOS
    schedules = build_schedules(games)
    sos = compute_rating_sos(perf, schedules)  # team -> 0..100 approx

    # Quality wins
    qw = compute_quality_wins(games, perf, sos)

    # Assemble row per FBS team
    # (Use FBS list to keep name+conference normalized)
    table: List[Dict[str, Any]] = []
    for row in fbs:
        team = row["team"]
        conf = row["conference"]

        wins, losses = records_map.get(team, (0, 0))
        games_played = rec.get(team, {}).get("games", wins + losses)
        win_pct = rec.get(team, {}).get("win_pct", 0.0)

        # advanced metrics (0.0 default if missing)
        s = stats.get(team, {})
        off_ppa = float(s.get("off_ppa", 0.0))
        off_sr = float(s.get("off_success_rate", 0.0))
        off_ex = float(s.get("off_explosiveness", 0.0))
        off_ppo = float(s.get("off_pts_per_opp", 0.0))
        def_ppa = float(s.get("def_ppa", 0.0))
        def_sr = float(s.get("def_success_rate", 0.0))
        def_ex = float(s.get("def_explosiveness", 0.0))
        def_ppo = float(s.get("def_pts_per_opp", 0.0))

        perf_score = float(perf.get(team, 0.0))
        sos_score = float(sos.get(team, 0.0))
        qwins = float(qw.get(team, 0.0))

        final_score = (
            WEIGHTS["final"]["performance"] * perf_score
            + WEIGHTS["final"]["sos"] * sos_score
            + WEIGHTS["final"]["quality_wins"] * qwins
        )

        table.append({
            "team": team,
            "conference": conf,
            "wins": int(wins),
            "losses": int(losses),
            "games": int(games_played),
            "win_pct": round(win_pct, 3),
            "performance": round(perf_score, 2),
            "sos": round(sos_score, 2),
            "quality_wins": round(qwins, 3),
            "final_score": round(final_score, 3),
            "off_ppa": round(off_ppa, 4),
            "off_success_rate": round(off_sr, 4),
            "off_explosiveness": round(off_ex, 4),
            "off_pts_per_opp": round(off_ppo, 4),
            "def_ppa": round(def_ppa, 4),
            "def_success_rate": round(def_sr, 4),
            "def_explosiveness": round(def_ex, 4),
            "def_pts_per_opp": round(def_ppo, 4),
            "slug": slugify_team(team),
        })

    # Ranks
    # performance_rank, sos_rank are position ranks (1=best)
    def rank_by_key(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
        sorted_rows = sorted(rows, key=lambda r: r[key], reverse=True)
        rank_map = {}
        rank = 1
        last_val = None
        for r in sorted_rows:
            val = r[key]
            if last_val is None or val != last_val:
                rank_map[r["team"]] = rank
            else:
                rank_map[r["team"]] = rank  # ties share the same displayed rank
            last_val = val
            rank += 1
        return rank_map

    perf_rank_map = rank_by_key(table, "performance")
    sos_rank_map = rank_by_key(table, "sos")

    for r in table:
        r["performance_rank"] = perf_rank_map.get(r["team"], None) or 999
        r["sos_rank"] = sos_rank_map.get(r["team"], None) or 999
        r["record"] = f"{r['wins']}-{r['losses']}"

    # Sort by final_score and apply head-to-head nudges
    table.sort(key=lambda r: r["final_score"], reverse=True)
    beaten = head_to_head_map(games)
    table = apply_h2h_nudges(table, beaten)

    # Assign overall rank and slice Top 25
    for idx, r in enumerate(table, start=1):
        r["rank"] = idx
    top25 = table[:25]

    out = {
        "season": YEAR,
        "last_build_utc": datetime.datetime.utcnow().isoformat(),
        "top25": [
            {
                "rank": t["rank"],
                "team": t["team"],
                "conference": t["conference"],
                "record": t["record"],
                "games": t["games"],
                "win_pct": t["win_pct"],
                "final_score": t["final_score"],
                "performance": t["performance"],
                "performance_rank": t["performance_rank"],
                "sos": t["sos"],
                "sos_rank": t["sos_rank"],
                "quality_wins": t["quality_wins"],
                "off_ppa": t["off_ppa"],
                "off_success_rate": t["off_success_rate"],
                "off_explosiveness": t["off_explosiveness"],
                "off_pts_per_opp": t["off_pts_per_opp"],
                "def_ppa": t["def_ppa"],
                "def_success_rate": t["def_success_rate"],
                "def_explosiveness": t["def_explosiveness"],
                "def_pts_per_opp": t["def_pts_per_opp"],
                "slug": t["slug"],
            }
            for t in top25
        ]
    }

    os.makedirs(os.path.join("docs", "data"), exist_ok=True)
    out_path = os.path.join("docs", "data", "rankings.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")

# -------------------- ENTRY --------------------------------

if __name__ == "__main__":
    main()
