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

# --------------- FETCH HELPERS ----------------

def fetch_fbs_names(year: int) -> Set[str]:
    data = cfbd_get("teams/fbs", {"year": year})
    names = {row.get("school") for row in (data or []) if row.get("school")}
    print(f"[FBS] {len(names)} teams")
    return names

def fetch_current_week(year: int) -> int:
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

# --------------- POLICY: count all games for FBS teams (SoS uses FBS opps only) -----

def keep_games_with_fbs_team(games: List[dict], fbs: Set[str]) -> List[dict]:
    kept = [g for g in games if (g["home_team"] in fbs) or (g["away_team"] in fbs)]
    print(f"[filter] games with at least one FBS team kept={len(kept)}")
    return kept

# --------------- ADVANCED STATS (Layer 1) ----------------

def fetch_season_advanced(year: int) -> Dict[str, dict]:
    """
    Returns team -> {off_ppa, def_ppa, off_sr, def_sr, off_expl, def_expl}
    """
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

def zscore_map(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 1.0
    mean = statistics.fmean(values)
    try:
        stdev = statistics.pstdev(values)
    except statistics.StatisticsError:
        stdev = 1.0
    if stdev == 0:
        stdev = 1.0
    return mean, stdev

def standardize_advanced(adv: Dict[str, dict], fbs: Set[str]) -> Dict[str, dict]:
    keys = ["off_ppa", "def_ppa", "off_sr", "def_sr", "off_expl", "def_expl"]
    arrays: Dict[str, List[float]] = {k: [] for k in keys}
    for t in fbs:
        row = adv.get(t) or {}
        for k in keys:
            v = row.get(k)
            if isinstance(v, (int, float)):
                arrays[k].append(float(v))
    mu_sigma = {k: zscore_map(arrays[k]) for k in keys}
    std: Dict[str, dict] = {}
    for t in fbs:
        row = adv.get(t) or {}
        zrow = {}
        for k in keys:
            mu, sd = mu_sigma[k]
            v = row.get(k)
            if isinstance(v, (int, float)):
                z = (float(v) - mu) / sd
            else:
                z = 0.0
            zrow[k] = z
        std[t] = zrow
    return std

# --------------- METRICS ----------------

def roll_up(games: List[dict], fbs: Set[str]) -> Dict[str, dict]:
    teams: Dict[str, dict] = {}
    def init(t: str):
        if t not in teams:
            teams[t] = {"games": 0, "wins": 0, "losses": 0, "pf": 0, "pa": 0, "opps": set(), "results": []}
    for g in games:
        h, a = g["home_team"], g["away_team"]
        hp, ap = int(g["home_points"] or 0), int(g["away_points"] or 0)

        if h in fbs:
            init(h)
            teams[h]["games"] += 1
            teams[h]["pf"] += hp
            teams[h]["pa"] += ap
            teams[h]["opps"].add(a)
            if hp > ap:
                teams[h]["wins"] += 1
                teams[h]["results"].append(("W", a))
            elif ap > hp:
                teams[h]["losses"] += 1
                teams[h]["results"].append(("L", a))

        if a in fbs:
            init(a)
            teams[a]["games"] += 1
            teams[a]["pf"] += ap
            teams[a]["pa"] += hp
            teams[a]["opps"].add(h)
            if ap > hp:
                teams[a]["wins"] += 1
                teams[a]["results"].append(("W", h))
            elif hp > ap:
                teams[a]["losses"] += 1
                teams[a]["results"].append(("L", h))
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

# ---- Pass 1 base score (efficiency + SoS + margin + results)

def base_score(win_pct: float, sos: float, avg_margin: float, z: dict) -> float:
    # Slightly more weight on results than before
    return (
        0.40 * win_pct +
        0.22 * sos +
        0.12 * (avg_margin / 25.0) +
        0.10 * (z.get("off_ppa", 0.0) - z.get("def_ppa", 0.0)) +
        0.08 * (z.get("off_sr",  0.0) - z.get("def_sr",  0.0)) +
        0.08 * (z.get("off_expl",0.0) - z.get("def_expl",0.0))
    )

# ---- Quality wins, bad losses, and head-to-head nudges

def second_order_adjustments(team: str, team_row: dict, rank_map: Dict[str, int]) -> Tuple[float, dict]:
    """
    Compute:
      - quality wins bonus: wins vs top40 (scaled)
      - bad loss penalty: losses vs rank > 80 (scaled)
      - head-to-head nudge: beat a higher-ranked team gets small bonus
    Returns (adj_score, details_dict)
    """
    qwins = 0
    badloss = 0
    h2h = 0
    details = {"quality_wins": 0, "bad_losses": 0, "h2h_upsets": 0}

    for res, opp in team_row.get("results", []):
        r = rank_map.get(opp, 999)
        if res == "W":
            if r <= 40:
                qwins += 1
                details["quality_wins"] += 1
            # head-to-head nudge if you beat someone ranked above you
            # actual H2H bonus applied later once we know this team's rank vs opp
        elif res == "L":
            if r >= 80:
                badloss += 1
                details["bad_losses"] += 1

    # scale: assume 0..4 quality wins typical mid-season
    q_bonus = 0.04 * min(qwins, 6)  # cap at 6
    # penalties mild but real
    bl_pen = -0.03 * min(badloss, 6)

    # For h2h, give small bonus for each opponent ranked above you at time of pass1
    # We don't know "you vs opp" ordering here, so award if opp rank < your provisional rank
    # We'll adapt this in a final pass.
    # Placeholder: count upsets as wins vs opp rank <= your rank + 5
    # Then scale small:
    details["h2h_upsets"] = 0  # set later in final stitching
    h2h_bonus = 0.0

    return (q_bonus + bl_pen + h2h_bonus), {**details, "q_bonus": q_bonus, "badloss_pen": bl_pen, "h2h_bonus": h2h_bonus}

def score_pass(teams: Dict[str, dict], zadv: Dict[str, dict], rank_map: Dict[str, int] | None = None) -> List[dict]:
    out = []
    for t, d in teams.items():
        if d["games"] < 1:
            continue
        win_pct = d["wins"] / d["games"]
        avg_margin = (d["pf"] - d["pa"]) / max(1, d["games"])
        sos = d.get("sos", 0.0)
        z = zadv.get(t, {})

        s = base_score(win_pct, sos, avg_margin, z)
        comp = {
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
            }
        }

        adj = 0.0
        adj_details = {"quality_wins": 0, "bad_losses": 0, "q_bonus": 0.0, "badloss_pen": 0.0, "h2h_bonus": 0.0}
        if rank_map is not None:
            adj, adj_details = second_order_adjustments(t, d, rank_map)
            s += adj

        out.append({
            "team": t,
            "wins": d["wins"],
            "losses": d["losses"],
            "games": d["games"],
            "points_for": d["pf"],
            "points_against": d["pa"],
            "sos": round(sos, 6),
            "avg_margin": round(avg_margin, 3),
            "score": round(s, 6),
            "components": {**comp, "second_order": adj_details}
        })
    out.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(out, start=1):
        r["rank"] = i
    return out

def make_rank_map(rows: List[dict]) -> Dict[str, int]:
    return {r["team"]: r["rank"] for r in rows}

# --------------- IO ----------------

def write_json(payload: dict):
    os.makedirs("docs/data", exist_ok=True)
    with open("docs/data/rankings.json", "w") as f:
        json.dump(payload, f, indent=2)

# --------------- MAIN ----------------

def main():
    print(f"Building FBS rankings (two-pass) for {YEAR}")
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

    # Pass 1: base table
    pass1 = score_pass(teams, zadv, rank_map=None)
    rmap1 = make_rank_map(pass1)

    # Pass 2: add quality wins, bad losses (and placeholder h2h small nudge via qwins)
    pass2 = score_pass(teams, zadv, rank_map=rmap1)

    out = {
        "season": YEAR,
        "last_build_utc": datetime.datetime.utcnow().isoformat(),
        "notes": {
            "weeks_included": f"1..{cur_week}",
            "model_layers": "Win%, SoS (FBS only), Avg margin, Advanced efficiency (PPA/SR/Expl), quality wins, bad-loss penalty",
            "weights": "Base: 0.40*Win% + 0.22*SoS + 0.12*Margin/25 + 0.10*(OffPPA-DefPPA) + 0.08*(OffSR-DefSR) + 0.08*(OffExpl-DefExpl); Second-order: +0.04*qualityWins -0.03*badLosses (caps applied)"
        },
        "top25": pass2[:25]
    }
    write_json(out)
    print(f"✅ Built Top 25 from {len(teams)} FBS teams • weeks=1..{cur_week}")

if __name__ == "__main__":
    main()
