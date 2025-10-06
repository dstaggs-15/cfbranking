# src/compute_rankings.py
# ------------------------------------------------------------
# College Football in-season ranking builder (no preseason bias)
# 2025-ready: uses CFBD GraphQL for finished games + REST v2 for stats/records
# Output -> docs/data/rankings.json
# ------------------------------------------------------------

import os, json, math, time, datetime, statistics, requests
from typing import Dict, List, Any, Tuple, Optional

# -------------------- YEAR & AUTH ---------------------------

def get_year() -> int:
    y = os.getenv("YEAR")
    if y and y.strip().isdigit():
        return int(y)
    return datetime.datetime.now().year

YEAR: int = get_year()

CFBD_KEY = os.getenv("CFBD_API_KEY")
if not CFBD_KEY:
    raise RuntimeError("Missing CFBD_API_KEY (set it as a GitHub Actions secret).")

# v2 REST + GraphQL endpoints (2025)
REST_BASE = "https://api.collegefootballdata.com"
GQL_URL   = "https://graphql.collegefootballdata.com/v1/graphql"

REST_HEADERS = {"Authorization": f"Bearer {CFBD_KEY}"}
GQL_HEADERS  = {"Authorization": f"Bearer {CFBD_KEY}", "Content-Type": "application/json"}

# -------------------- WEIGHTS -------------------------------

WEIGHTS = {
    "offense": {"ppa": 0.40, "success_rate": 0.30, "explosiveness": 0.20, "pts_per_opp": 0.10},
    "defense": {"ppa": 0.40, "success_rate": 0.30, "explosiveness": 0.20, "pts_per_opp": 0.10},  # invert later
    "final":   {"performance": 0.45, "sos": 0.35, "quality_wins": 0.20},
    "quality_win_scalar": 0.05,
    "h2h_threshold": 1.0,
    "h2h_nudge": 0.15,
}

# -------------------- HTTP HELPERS --------------------------

def _rest_get(endpoint: str, params: Optional[Dict[str, Any]] = None, tries: int = 4) -> Any:
    url = f"{REST_BASE}/{endpoint.lstrip('/')}"
    for attempt in range(tries):
        r = requests.get(url, headers=REST_HEADERS, params=params, timeout=45)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(1.5 * (attempt + 1))
            continue
        r.raise_for_status()
    r.raise_for_status()

def _graphql(query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    body = {"query": query, "variables": variables}
    r = requests.post(GQL_URL, headers=GQL_HEADERS, json=body, timeout=60)
    r.raise_for_status()
    data = r.json()
    if "errors" in data and data["errors"]:
        raise RuntimeError(f"GraphQL errors: {data['errors']}")
    return data["data"]

# -------------------- FETCHERS ------------------------------

def fetch_fbs_teams(year: int) -> List[Dict[str, Any]]:
    # REST v2
    data = _rest_get("teams/fbs", {"year": year}) or []
    return [{"team": t.get("school"), "conference": t.get("conference")} for t in data]

def fetch_finished_games_gql(year: int) -> List[Dict[str, Any]]:
    """
    GraphQL: finished (status='final') games for given season, any week, seasonType=regular, FBS only.
    """
    query = """
    query FinishedGames($yr: smallint!) {
      game(
        where: {
          season: { _eq: $yr },
          seasonType: { _eq: REGULAR },
          status: { _eq: final },
          homeClassification: { _eq: FBS },
          awayClassification: { _eq: FBS }
        }
        orderBy: { startDate: ASC }
      ) {
        id
        week
        startDate
        homeTeam
        homeConference
        homePoints
        awayTeam
        awayConference
        awayPoints
        neutralSite
      }
    }
    """
    data = _graphql(query, {"yr": year})
    games = data.get("game", []) or []
    finished = []
    for g in games:
        hp = g.get("homePoints"); ap = g.get("awayPoints")
        if hp is None or ap is None:
            continue
        finished.append({
            "week": g.get("week"),
            "date": g.get("startDate"),
            "home_team": g.get("homeTeam"),
            "home_conf": g.get("homeConference"),
            "home_points": hp,
            "away_team": g.get("awayTeam"),
            "away_conf": g.get("awayConference"),
            "away_points": ap,
            "neutral_site": bool(g.get("neutralSite")),
        })
    print(f"[GQL games] finished={len(finished)}")
    return finished

def fetch_advanced_stats(year: int) -> Dict[str, Dict[str, float]]:
    # REST v2
    data = _rest_get("stats/season/advanced", {"year": year}) or []
    teams: Dict[str, Dict[str, float]] = {}
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
    # REST v2
    data = _rest_get("records", {"year": year}) or []
    recs: Dict[str, Tuple[int, int]] = {}
    for r in data:
        team = r.get("team")
        total = r.get("total") or {}
        wins = int(total.get("wins") or 0)
        losses = int(total.get("losses") or 0)
        recs[team] = (wins, losses)
    return recs

# -------------------- METRIC HELPERS ------------------------

def zscores(vals: List[float]) -> List[float]:
    if not vals: return []
    mean = statistics.fmean(vals)
    var  = statistics.fmean([(v - mean) ** 2 for v in vals])
    std  = math.sqrt(var)
    if std == 0: return [0.0 for _ in vals]
    return [(v - mean) / std for v in vals]

def minmax01(vals: List[float]) -> List[float]:
    if not vals: return []
    mn, mx = min(vals), max(vals)
    if mx - mn == 0: return [0.5 for _ in vals]
    return [(v - mn) / (mx - mn) for v in vals]

def scale_0_100(vals: List[float]) -> List[float]:
    return [round(v * 100.0, 2) for v in minmax01(vals)]

def invert_list(vals: List[float]) -> List[float]:
    return [-v if v is not None else None for v in vals]

def mean_ignore_none(vals: List[Optional[float]]) -> float:
    clean = [v for v in vals if v is not None and not math.isnan(v)]
    return float(statistics.fmean(clean)) if clean else 0.0

def slugify_team(name: str) -> str:
    s = (name or "").strip().lower()
    out = []
    for ch in s:
        if ch.isalnum(): out.append(ch)
        elif ch.isspace() or ch in "-_": out.append("-")
    slug, last_dash = [], False
    for ch in out:
        if ch == "-":
            if not last_dash: slug.append("-")
            last_dash = True
        else:
            slug.append(ch); last_dash = False
    return "".join(slug).strip("-")

# -------------------- PIPELINE ------------------------------

def build_records(games: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    rec: Dict[str, Dict[str, Any]] = {}
    for g in games:
        home, away = g["home_team"], g["away_team"]
        hp, ap = g["home_points"], g["away_points"]
        rec.setdefault(home, {"team": home, "wins": 0, "losses": 0})
        rec.setdefault(away, {"team": away, "wins": 0, "losses": 0})
        if hp > ap:
            rec[home]["wins"] += 1; rec[away]["losses"] += 1
        else:
            rec[away]["wins"] += 1; rec[home]["losses"] += 1
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
    teams = list(stats.keys())
    if not teams: return {}
    off_ppa = [stats[t]["off_ppa"] for t in teams]
    off_sr  = [stats[t]["off_success_rate"] for t in teams]
    off_ex  = [stats[t]["off_explosiveness"] for t in teams]
    off_ppo = [stats[t]["off_pts_per_opp"] for t in teams]
    def_ppa = invert_list([stats[t]["def_ppa"] for t in teams])
    def_sr  = invert_list([stats[t]["def_success_rate"] for t in teams])
    def_ex  = invert_list([stats[t]["def_explosiveness"] for t in teams])
    def_ppo = invert_list([stats[t]["def_pts_per_opp"] for t in teams])

    z = {
        "z_off_ppa": zscores(off_ppa),
        "z_off_sr":  zscores(off_sr),
        "z_off_ex":  zscores(off_ex),
        "z_off_ppo": zscores(off_ppo),
        "z_def_ppa": zscores(def_ppa),
        "z_def_sr":  zscores(def_sr),
        "z_def_ex":  zscores(def_ex),
        "z_def_ppo": zscores(def_ppo),
    }
    ow, dw = WEIGHTS["offense"], WEIGHTS["defense"]
    perf_z = []
    for i in range(len(teams)):
        off_score = ow["ppa"]*z["z_off_ppa"][i] + ow["success_rate"]*z["z_off_sr"][i] + \
                    ow["explosiveness"]*z["z_off_ex"][i] + ow["pts_per_opp"]*z["z_off_ppo"][i]
        def_score = dw["ppa"]*z["z_def_ppa"][i] + dw["success_rate"]*z["z_def_sr"][i] + \
                    dw["explosiveness"]*z["z_def_ex"][i] + dw["pts_per_opp"]*z["z_def_ppo"][i]
        perf_z.append(0.5*off_score + 0.5*def_score)

    perf_scaled = scale_0_100(perf_z)
    return {teams[i]: perf_scaled[i] for i in range(len(teams))}

def compute_rating_sos(perf: Dict[str, float], schedules: Dict[str, set]) -> Dict[str, float]:
    sos: Dict[str, float] = {}
    for team, opps in schedules.items():
        opp_scores = [perf.get(o) for o in opps if o in perf]
        sos[team] = mean_ignore_none(opp_scores)
    return sos

def compute_quality_wins(games: List[Dict[str, Any]], perf: Dict[str, float], sos: Dict[str, float]) -> Dict[str, float]:
    scalar = WEIGHTS["quality_win_scalar"]
    qw: Dict[str, float] = {}
    for g in games:
        home, away = g["home_team"], g["away_team"]
        hp, ap = g["home_points"], g["away_points"]
        winner, loser = (home, away) if hp > ap else (away, home)
        opp_perf, opp_sos = perf.get(loser), sos.get(loser)
        bonus = 0.0 if (opp_perf is None or opp_sos is None) else scalar * ((opp_perf + opp_sos) / 2.0)
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
    threshold, nudge = WEIGHTS["h2h_threshold"], WEIGHTS["h2h_nudge"]
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

    fbs = fetch_fbs_teams(YEAR)
    fbs_names = {t["team"] for t in fbs if t.get("team")}

    # Finished FBS vs FBS games via GraphQL (2025-compatible)
    games = fetch_finished_games_gql(YEAR)
    games = [g for g in games if g["home_team"] in fbs_names and g["away_team"] in fbs_names]
    print(f"[FBS filter] finished games kept: {len(games)}")
    if not games:
        raise SystemExit("No finished FBS vs FBS games returned by GraphQL for this season.")

    # Season-level stats & records (REST v2)
    stats = fetch_advanced_stats(YEAR)
    records_map = fetch_records(YEAR)

    rec = build_records(games)
    schedules = build_schedules(games)
    perf = performance_composite(stats)
    sos  = compute_rating_sos(perf, schedules)
    qw   = compute_quality_wins(games, perf, sos)

    # Build table
    table: List[Dict[str, Any]] = []
    for row in fbs:
        team, conf = row["team"], row["conference"]
        wins, losses = records_map.get(team, (0, 0))
        games_played = rec.get(team, {}).get("games", wins + losses)
        win_pct = rec.get(team, {}).get("win_pct", 0.0)

        s = stats.get(team, {})
        off_ppa = float(s.get("off_ppa", 0.0));    off_sr  = float(s.get("off_success_rate", 0.0))
        off_ex  = float(s.get("off_explosiveness", 0.0));  off_ppo = float(s.get("off_pts_per_opp", 0.0))
        def_ppa = float(s.get("def_ppa", 0.0));    def_sr  = float(s.get("def_success_rate", 0.0))
        def_ex  = float(s.get("def_explosiveness", 0.0));  def_ppo = float(s.get("def_pts_per_opp", 0.0))

        perf_score = float(perf.get(team, 0.0))
        sos_score  = float(sos.get(team, 0.0))
        qwins      = float(qw.get(team, 0.0))

        final_score = WEIGHTS["final"]["performance"] * perf_score + \
                      WEIGHTS["final"]["sos"] * sos_score + \
                      WEIGHTS["final"]["quality_wins"] * qwins

        table.append({
            "team": team, "conference": conf,
            "wins": int(wins), "losses": int(losses),
            "games": int(games_played), "win_pct": round(win_pct, 3),
            "performance": round(perf_score, 2), "sos": round(sos_score, 2),
            "quality_wins": round(qwins, 3), "final_score": round(final_score, 3),
            "off_ppa": round(off_ppa, 4), "off_success_rate": round(off_sr, 4),
            "off_explosiveness": round(off_ex, 4), "off_pts_per_opp": round(off_ppo, 4),
            "def_ppa": round(def_ppa, 4), "def_success_rate": round(def_sr, 4),
            "def_explosiveness": round(def_ex, 4), "def_pts_per_opp": round(def_ppo, 4),
            "slug": slugify_team(team),
        })

    # Rank components
    def rank_by_key(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
        sorted_rows = sorted(rows, key=lambda r: r[key], reverse=True)
        rank_map, rank, last_val = {}, 1, None
        for r in sorted_rows:
            val = r[key]
            if last_val is None or val != last_val:
                rank_map[r["team"]] = rank
            else:
                rank_map[r["team"]] = rank
            last_val = val; rank += 1
        return rank_map

    perf_rank = rank_by_key(table, "performance")
    sos_rank  = rank_by_key(table, "sos")
    for r in table:
        r["performance_rank"] = perf_rank.get(r["team"], 999)
        r["sos_rank"] = sos_rank.get(r["team"], 999)
        r["record"] = f"{r['wins']}-{r['losses']}"

    # Sort, apply H2H nudges, finalize ranks
    table.sort(key=lambda r: r["final_score"], reverse=True)
    table = apply_h2h_nudges(table, head_to_head_map(games))
    for i, r in enumerate(table, start=1):
        r["rank"] = i

    top25 = table[:25]
    out = {
        "season": YEAR,
        "last_build_utc": datetime.datetime.utcnow().isoformat(),
        "top25": [{
            "rank": t["rank"], "team": t["team"], "conference": t["conference"],
            "record": t["record"], "games": t["games"], "win_pct": t["win_pct"],
            "final_score": t["final_score"], "performance": t["performance"],
            "performance_rank": t["performance_rank"], "sos": t["sos"],
            "sos_rank": t["sos_rank"], "quality_wins": t["quality_wins"],
            "off_ppa": t["off_ppa"], "off_success_rate": t["off_success_rate"],
            "off_explosiveness": t["off_explosiveness"], "off_pts_per_opp": t["off_pts_per_opp"],
            "def_ppa": t["def_ppa"], "def_success_rate": t["def_success_rate"],
            "def_explosiveness": t["def_explosiveness"], "def_pts_per_opp": t["def_pts_per_opp"],
            "slug": t["slug"],
        } for t in top25]
    }

    os.makedirs("docs/data", exist_ok=True)
    with open("docs/data/rankings.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote docs/data/rankings.json")

# -------------------- ENTRY --------------------------------

if __name__ == "__main__":
    main()
