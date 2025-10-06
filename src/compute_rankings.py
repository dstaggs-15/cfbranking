import os
import json
import datetime
import time
import statistics
from typing import Dict, List, Set, Tuple

import requests

API_BASE = "https://api.collegefootballdata.com"
API_KEY = os.getenv("CFBD_API_KEY", "")
YEAR = int(os.getenv("YEAR") or datetime.datetime.now().year)

# ====================== HTTP ======================

def cfbd_get(path: str, params: dict | None = None, tries: int = 3, backoff: float = 1.6):
    """GET helper with CFBD Bearer auth + simple retry."""
    if not API_KEY:
        raise RuntimeError("CFBD_API_KEY is not set.")
    url = f"{API_BASE}/{path.lstrip('/')}"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    params = params or {}
    last_text = ""
    code = 0
    for a in range(tries):
        r = requests.get(url, headers=headers, params=params, timeout=45)
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

# ====================== FETCHERS ======================

def fetch_fbs_names(year: int) -> Set[str]:
    data = cfbd_get("teams/fbs", {"year": year})
    names = {row.get("school") for row in (data or []) if row.get("school")}
    print(f"[FBS] {len(names)} teams")
    return names

def fetch_current_week(year: int) -> int:
    """Current regular-season week based on calendar entries that have started."""
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
    if g.get("completed") is True:
        return True
    hp = g.get("home_points") if "home_points" in g else g.get("homePoints")
    ap = g.get("away_points") if "away_points" in g else g.get("awayPoints")
    return (hp is not None and ap is not None)

def norm_game(g: dict) -> dict:
    return {
        "id": g.get("id"),
        "week": g.get("week"),
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
        fin = [norm_game(g) for g in data if is_finished(g)]
        total_finished += len(fin)
        all_games.extend(fin)
        print(f"[games] week={wk} -> pulled={len(data)} finished={len(fin)}")
    print(f"[games] aggregated finished={total_finished}")
    return all_games

# Policy: keep any game where at least one team is FBS (records reflect public W/L)
def keep_games_with_fbs_team(games: List[dict], fbs: Set[str]) -> List[dict]:
    kept = [g for g in games if (g["home_team"] in fbs) or (g["away_team"] in fbs)]
    print(f"[filter] games with at least one FBS team kept={len(kept)}")
    return kept

# ====================== ADVANCED (Layer 1) ======================

def fetch_season_advanced(year: int) -> Dict[str, dict]:
    """team -> {off_ppa, def_ppa, off_sr, def_sr, off_expl, def_expl}"""
    data = cfbd_get("stats/season/advanced", {"year": year})
    if not isinstance(data, list):
        print("[adv] unexpected payload")
        return {}
    out: Dict[str, dict] = {}
    for row in data:
        team = row.get("team")
        off = row.get("offense") or {}
        de  = row.get("defense") or {}
        if not team:
            continue
        out[team] = {
            "off_ppa": off.get("ppa"),
            "def_ppa": de.get("ppa"),
            "off_sr": off.get("successRate"),
            "def_sr": de.get("successRate"),
            "off_expl": off.get("explosiveness"),
            "def_expl": de.get("explosiveness"),
        }
    print(f"[adv] season advanced rows: {len(out)}")
    return out

def zstats(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 1.0
    mu = statistics.fmean(values)
    try:
        sd = statistics.pstdev(values)
    except statistics.StatisticsError:
        sd = 1.0
    if sd == 0:
        sd = 1.0
    return mu, sd

def standardize_advanced(adv: Dict[str, dict], fbs: Set[str]) -> Dict[str, dict]:
    keys = ["off_ppa", "def_ppa", "off_sr", "def_sr", "off_expl", "def_expl"]
    arrays: Dict[str, List[float]] = {k: [] for k in keys}
    for t in fbs:
        row = adv.get(t) or {}
        for k in keys:
            v = row.get(k)
            if isinstance(v, (int, float)):
                arrays[k].append(float(v))
    mu_sigma = {k: zstats(arrays[k]) for k in keys}
    std: Dict[str, dict] = {}
    for t in fbs:
        row = adv.get(t) or {}
        zrow = {}
        for k in keys:
            mu, sd = mu_sigma[k]
            v = row.get(k)
            z = (float(v) - mu) / sd if isinstance(v, (int, float)) else 0.0
            zrow[k] = z
        std[t] = zrow
    return std

# ====================== ROLLUP & SOS ======================

def roll_up(games: List[dict], fbs: Set[str]) -> Dict[str, dict]:
    teams: Dict[str, dict] = {}
    def init(t: str):
        if t not in teams:
            teams[t] = {
                "games": 0, "wins": 0, "losses": 0, "pf": 0, "pa": 0,
                "opps": set(), "results": []  # results: [("W"/"L", opponent)]
            }
    for g in games:
        h, a = g["home_team"], g["away_team"]
        hp, ap = int(g["home_points"] or 0), int(g["away_points"] or 0)

        if h in fbs:
            init(h)
            teams[h]["games"] += 1
            teams[h]["pf"] += hp
            teams[h]["pa"] += ap
            teams[h]["opps"].add(a)
            if hp > ap: teams[h]["wins"] += 1; teams[h]["results"].append(("W", a))
            elif ap > hp: teams[h]["losses"] += 1; teams[h]["results"].append(("L", a))

        if a in fbs:
            init(a)
            teams[a]["games"] += 1
            teams[a]["pf"] += ap
            teams[a]["pa"] += hp
            teams[a]["opps"].add(h)
            if ap > hp: teams[a]["wins"] += 1; teams[a]["results"].append(("W", h))
            elif hp > ap: teams[a]["losses"] += 1; teams[a]["results"].append(("L", h))
    return teams

def compute_sos_fbs_only(teams: Dict[str, dict], fbs: Set[str]) -> None:
    for t, d in teams.items():
        opps = [o for o in d["opps"] if o in fbs]
        wpcts = []
        for o in opps:
            if o not in teams or teams[o]["games"] == 0:
                continue
            wpcts.append(teams[o]["wins"] / teams[o]["games"])
        d["sos"] = float(statistics.mean(wpcts)) if wpcts else 0.0

# ====================== SCORING ======================

# Base weights (sum=1.00)
W_WIN = 0.52
W_SOS = 0.20
W_MARGIN = 0.08
W_PPA = 0.08
W_SR  = 0.06
W_EX  = 0.06

# Second-order tuning
QW_TIER = { "top15": 0.07, "r16_40": 0.04, "r41_60": 0.02 }   # per win
BL_TIER = { "r50_79": -0.04, "r80p": -0.06 }                  # per loss
H2H_ENFORCE_DELTA = 0.08  # if two teams are within this after adjustments, enforce H2H

def base_score(win_pct: float, sos: float, avg_margin: float, z: dict) -> float:
    return (
        W_WIN * win_pct +
        W_SOS * sos +
        W_MARGIN * (avg_margin / 25.0) +
        W_PPA * (z.get("off_ppa", 0.0) - z.get("def_ppa", 0.0)) +
        W_SR  * (z.get("off_sr",  0.0) - z.get("def_sr",  0.0)) +
        W_EX  * (z.get("off_expl",0.0) - z.get("def_expl",0.0))
    )

def tier_quality_bad(r: int) -> Tuple[str | None, str | None]:
    """Return (quality_win_tier, bad_loss_tier) for opponent rank r."""
    qw = None
    bl = None
    if r <= 15: qw = "top15"
    elif r <= 40: qw = "r16_40"
    elif r <= 60: qw = "r41_60"

    if r >= 80: bl = "r80p"
    elif r >= 50: bl = "r50_79"

    return qw, bl

def second_order(team: str, row: dict, rank_map: Dict[str, int]) -> Tuple[float, dict]:
    """Compute Quality-Win/Bad-Loss adjustments based on provisional ranks."""
    q_counts = {"top15": 0, "r16_40": 0, "r41_60": 0}
    bl_counts = {"r50_79": 0, "r80p": 0}

    for res, opp in row.get("results", []):
        r = rank_map.get(opp, 999)
        qw_tier, bl_tier = tier_quality_bad(r)
        if res == "W" and qw_tier:
            q_counts[qw_tier] += 1
        elif res == "L" and bl_tier:
            bl_counts[bl_tier] += 1

    q_bonus = (
        q_counts["top15"]  * QW_TIER["top15"] +
        q_counts["r16_40"] * QW_TIER["r16_40"] +
        q_counts["r41_60"] * QW_TIER["r41_60"]
    )
    bl_pen = (
        bl_counts["r50_79"] * BL_TIER["r50_79"] +
        bl_counts["r80p"]   * BL_TIER["r80p"]
    )

    details = {
        "quality_wins": q_counts,
        "bad_losses": bl_counts,
        "q_bonus": round(q_bonus, 6),
        "badloss_pen": round(bl_pen, 6),
        "h2h_enforced": []  # filled later
    }
    return q_bonus + bl_pen, details

def score_pass(teams: Dict[str, dict], zadv: Dict[str, dict], rank_map: Dict[str, int] | None = None) -> List[dict]:
    """Score teams; if rank_map is provided, add second-order adjustments."""
    rows = []
    for t, d in teams.items():
        if d["games"] < 1:
            continue
        win_pct = d["wins"] / d["games"]
        avg_margin = (d["pf"] - d["pa"]) / max(1, d["games"])
        sos = d.get("sos", 0.0)
        z = zadv.get(t, {})

        s = base_score(win_pct, sos, avg_margin, z)
        second = {"quality_wins": {}, "bad_losses": {}, "q_bonus": 0.0, "badloss_pen": 0.0, "h2h_enforced": []}
        if rank_map is not None:
            adj, second = second_order(t, d, rank_map)
            s += adj

        rows.append({
            "team": t,
            "wins": d["wins"],
            "losses": d["losses"],
            "games": d["games"],
            "points_for": d["pf"],
            "points_against": d["pa"],
            "sos": round(sos, 6),
            "avg_margin": round(avg_margin, 3),
            "score": round(s, 6),
            "components": {
                "win_pct": round(win_pct, 6),
                "sos": round(sos, 6),
                "margin_scaled": round(avg_margin / 25.0, 6),
                "z": {
                    "off_ppa": round(z.get("off_ppa", 0.0), 4),
                    "def_ppa": round(z.get("def_ppa", 0.0), 4),
                    "off_sr":  round(z.get("off_sr", 0.0), 4),
                    "def_sr":  round(z.get("def_sr", 0.0), 4),
                    "off_expl":round(z.get("off_expl",0.0), 4),
                    "def_expl":round(z.get("def_expl",0.0), 4),
                },
                "second_order": second
            }
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    return rows

def make_rank_map(rows: List[dict]) -> Dict[str, int]:
    return {r["team"]: r["rank"] for r in rows}

def enforce_head_to_head(rows: List[dict], teams: Dict[str, dict]) -> List[dict]:
    """
    If Team A beat Team B and their scores are within H2H_ENFORCE_DELTA,
    force A to be ranked above B by swapping neighbors iteratively.
    """
    rank = make_rank_map(rows)
    name_to_row = {r["team"]: r for r in rows}

    def beat(a: str, b: str) -> bool:
        # did a beat b at least once?
        for res, opp in teams[a].get("results", []):
            if res == "W" and opp == b:
                return True
        return False

    changed = True
    while changed:
        changed = False
        for i in range(len(rows) - 1):
            A = rows[i]["team"]
            B = rows[i+1]["team"]
            sA = rows[i]["score"]; sB = rows[i+1]["score"]
            if abs(sA - sB) <= H2H_ENFORCE_DELTA:
                if beat(B, A) and not beat(A, B):
                    # swap to put B above A
                    rows[i], rows[i+1] = rows[i+1], rows[i]
                    rows[i]["rank"], rows[i+1]["rank"] = i+1, i+2
                    # annotate
                    rows[i]["components"]["second_order"]["h2h_enforced"].append(f"{rows[i]['team']} > {rows[i+1]['team']}")
                    changed = True
                elif beat(A, B) and not beat(B, A):
                    # already in correct order (A above B); annotate
                    rows[i]["components"]["second_order"]["h2h_enforced"].append(f"{A} > {B}")
            # else: difference is big; no enforcement
    return rows

# ====================== IO ======================

def write_json(payload: dict):
    os.makedirs("docs/data", exist_ok=True)
    with open("docs/data/rankings.json", "w") as f:
        json.dump(payload, f, indent=2)

# ====================== MAIN ======================

def main():
    print(f"Building FBS rankings (reweighted + tiered quality/bad + H2H) for {YEAR}")
    fbs = fetch_fbs_names(YEAR)
    cur_week = fetch_current_week(YEAR)
    games_all = fetch_regular_games_by_week(YEAR, cur_week)
    games = keep_games_with_fbs_team(games_all, fbs)

    teams = roll_up(games, fbs)
    if not teams:
        print("❌ No completed games for FBS teams. Writing placeholder.")
        write_json({
            "season": YEAR,
            "last_build_utc": datetime.datetime.utcnow().isoformat(),
            "top25": [],
            "note": "No completed games for FBS teams available."
        })
        return

    compute_sos_fbs_only(teams, fbs)

    adv = fetch_season_advanced(YEAR)
    zadv = standardize_advanced(adv, fbs)

    # Pass 1: base table (no second-order)
    pass1 = score_pass(teams, zadv, rank_map=None)
    rmap1 = make_rank_map(pass1)

    # Pass 2: add quality/bad-loss adjustments using ranks from Pass 1
    pass2 = score_pass(teams, zadv, rank_map=rmap1)

    # Enforce head-to-head within proximity band
    final_rows = enforce_head_to_head(pass2, teams)

    out = {
        "season": YEAR,
        "last_build_utc": datetime.datetime.utcnow().isoformat(),
        "notes": {
            "weeks_included": f"1..{cur_week}",
            "weights_base": {
                "win_pct": W_WIN, "sos": W_SOS, "margin_scaled": W_MARGIN,
                "delta_ppa": W_PPA, "delta_sr": W_SR, "delta_expl": W_EX
            },
            "second_order": {
                "quality_win_tiers": QW_TIER, "bad_loss_tiers": BL_TIER,
                "h2h_enforcement_delta": H2H_ENFORCE_DELTA
            }
        },
        "top25": final_rows[:25]
    }
    write_json(out)
    print(f"✅ Top 25 built from {len(teams)} FBS teams • weeks=1..{cur_week}")

if __name__ == "__main__":
    main()
