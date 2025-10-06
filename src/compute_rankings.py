import os
import json
import datetime
import time
import statistics
from typing import Dict, List, Set, Tuple, Optional

import requests

API_BASE = "https://api.collegefootballdata.com"
API_KEY = os.getenv("CFBD_API_KEY", "")
YEAR = int(os.getenv("YEAR") or datetime.datetime.now().year)

# ====================== HTTP ======================

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

# ====================== FETCHERS ======================

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

# Keep any game with at least one FBS team (public W/L optics)
def keep_games_with_fbs_team(games: List[dict], fbs: Set[str]) -> List[dict]:
    kept = [g for g in games if (g["home_team"] in fbs) or (g["away_team"] in fbs)]
    print(f"[filter] games with at least one FBS team kept={len(kept)}")
    return kept

# ====================== ADVANCED (Layer 1) ======================

def fetch_season_advanced(year: int) -> Dict[str, dict]:
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

class TeamRow(dict):
    pass

def roll_up(games: List[dict], fbs: Set[str]) -> Dict[str, TeamRow]:
    teams: Dict[str, TeamRow] = {}
    def init(t: str):
        if t not in teams:
            teams[t] = TeamRow({
                "games": 0, "wins": 0, "losses": 0,
                "pf": 0, "pa": 0,
                "opps": set(), "results": [],  # [("W"/"L", opponent)]
                "fbs_games": 0, "fbs_wins": 0, "fbs_losses": 0
            })
    for g in games:
        h, a = g["home_team"], g["away_team"]
        hp, ap = int(g["home_points"] or 0), int(g["away_points"] or 0)

        if h in fbs:
            init(h)
            teams[h]["games"] += 1
            teams[h]["pf"] += hp
            teams[h]["pa"] += ap
            teams[h]["opps"].add(a)
            if a in fbs:
                teams[h]["fbs_games"] += 1
            if hp > ap:
                teams[h]["wins"] += 1
                teams[h]["results"].append(("W", a))
                if a in fbs: teams[h]["fbs_wins"] += 1
            elif ap > hp:
                teams[h]["losses"] += 1
                teams[h]["results"].append(("L", a))
                if a in fbs: teams[h]["fbs_losses"] += 1

        if a in fbs:
            init(a)
            teams[a]["games"] += 1
            teams[a]["pf"] += ap
            teams[a]["pa"] += hp
            teams[a]["opps"].add(h)
            if h in fbs:
                teams[a]["fbs_games"] += 1
            if ap > hp:
                teams[a]["wins"] += 1
                teams[a]["results"].append(("W", h))
                if h in fbs: teams[a]["fbs_wins"] += 1
            elif hp > ap:
                teams[a]["losses"] += 1
                teams[a]["results"].append(("L", h))
                if h in fbs: teams[a]["fbs_losses"] += 1
    return teams

def compute_sos_fbs_only(teams: Dict[str, TeamRow], fbs: Set[str]) -> None:
    for t, d in teams.items():
        opps = [o for o in d["opps"] if o in fbs]
        wpcts = []
        for o in opps:
            if o not in teams or teams[o]["games"] == 0:
                continue
            wpcts.append(teams[o]["wins"] / teams[o]["games"])
        d["sos"] = float(statistics.mean(wpcts)) if wpcts else 0.0

# ====================== SCORING ======================

# Base weights (sum = 1.00) — results-first
W_WIN = 0.45
W_SOS = 0.22
W_MARGIN = 0.03
W_PPA = 0.05
W_SR  = 0.03
W_EX  = 0.02
W_RESUME = 0.20

# Tiers
QW_TIER = { "top15": 0.10, "r16_40": 0.06, "r41_60": 0.03 }
BL_TIER = { "r50_79": -0.06, "r80p": -0.09 }

# Rules
LOSS_BUCKET_SORT = True
H2H_SAME_LOSSES_STRICT = True
WEAK_SOS_PENALTY = True

# Eligibility gate inside each loss bucket:
# (a) >=1 quality win (beat Top-40 provisional), OR (b) SoS>=0.52, OR (c) Undefeated & SoS>=0.48
def passes_gate(team_row: dict, qwin_count: int, sos: float) -> bool:
    undefeated = (team_row["losses"] == 0)
    return (qwin_count >= 1) or (sos >= 0.52) or (undefeated and sos >= 0.48)

def base_score(win_pct: float, sos: float, avg_margin: float, z: dict) -> Tuple[float, float]:
    sched_pen = 0.0
    if WEAK_SOS_PENALTY and sos < 0.50:
        # up to ~ -0.075 at sos near 0.0
        sched_pen = -0.15 * (0.50 - sos)

    score = (
        W_WIN * win_pct +
        W_SOS * sos +
        W_MARGIN * (avg_margin / 25.0) +
        W_PPA * (z.get("off_ppa", 0.0) - z.get("def_ppa", 0.0)) +
        W_SR  * (z.get("off_sr",  0.0) - z.get("def_sr",  0.0)) +
        W_EX  * (z.get("off_expl",0.0) - z.get("def_expl",0.0)) +
        sched_pen
    )
    return score, sched_pen

def make_rank_map(rows: List[dict]) -> Dict[str, int]:
    return {r["team"]: r["rank"] for r in rows}

def opponent_strength(rank_map: Dict[str, int], team: str, nteams: int) -> float:
    r = rank_map.get(team, nteams)
    return 1.0 - (r - 1) / max(1, nteams - 1)

def compute_resume(teams: Dict[str, TeamRow], rank_map: Dict[str, int], nteams: int) -> Dict[str, float]:
    # Resume raw: sum(s) for wins minus sum(1-s) for losses; normalize to [0,1]
    raw: Dict[str, float] = {}
    for t, d in teams.items():
        val = 0.0
        for res, opp in d.get("results", []):
            s = opponent_strength(rank_map, opp, nteams)
            if res == "W":
                val += s
            else:
                val -= (1.0 - s) * 1.05  # small extra sting for losing to weak teams
        raw[t] = val
    vals = list(raw.values())
    mn, mx = (min(vals), max(vals)) if vals else (0.0, 1.0)
    denom = (mx - mn) if (mx - mn) != 0 else 1.0
    return {t: (raw[t] - mn) / denom for t in raw}

def tier_quality_bad(r: int) -> Tuple[Optional[str], Optional[str]]:
    qw = None
    bl = None
    if r <= 15: qw = "top15"
    elif r <= 40: qw = "r16_40"
    elif r <= 60: qw = "r41_60"
    if r >= 80: bl = "r80p"
    elif r >= 50: bl = "r50_79"
    return qw, bl

def second_order(team: str, row: TeamRow, rank_map: Dict[str, int]) -> Tuple[float, dict, int]:
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

    total = q_bonus + bl_pen
    details = {
        "quality_wins": q_counts,
        "bad_losses": bl_counts,
        "q_bonus": round(q_bonus, 6),
        "badloss_pen": round(bl_pen, 6),
        "h2h_enforced": []
    }
    q_total = sum(q_counts.values())
    return total, details, q_total

def score_pass(teams: Dict[str, TeamRow], zadv: Dict[str, dict],
               rank_map: Optional[Dict[str, int]] = None,
               resume_map: Optional[Dict[str, float]] = None) -> List[dict]:
    rows = []
    for t, d in teams.items():
        if d["games"] < 1:
            continue
        win_pct = d["wins"] / d["games"]
        avg_margin = (d["pf"] - d["pa"]) / max(1, d["games"])
        sos = d.get("sos", 0.0)
        z = zadv.get(t, {})

        base, sched_pen = base_score(win_pct, sos, avg_margin, z)

        second_adj = 0.0
        comp_second = {"quality_wins": {}, "bad_losses": {}, "q_bonus": 0.0, "badloss_pen": 0.0, "h2h_enforced": []}
        qwin_count = 0
        if rank_map is not None:
            second_adj, comp_second, qwin_count = second_order(t, d, rank_map)
            base += second_adj

        resume_part = 0.0
        if resume_map is not None:
            resume_part = resume_map.get(t, 0.0)
            base += W_RESUME * resume_part

        # "no-quality gate tax": tiny push down if SoS weak and zero quality wins
        if (rank_map is not None) and (qwin_count == 0) and (sos < 0.52):
            base -= 0.02

        rows.append({
            "team": t,
            "wins": d["wins"],
            "losses": d["losses"],
            "games": d["games"],
            "points_for": d["pf"],
            "points_against": d["pa"],
            "sos": round(sos, 6),
            "avg_margin": round(avg_margin, 3),
            "score": round(base, 6),
            "components": {
                "win_pct": round(win_pct, 6),
                "sos": round(sos, 6),
                "margin_scaled": round(avg_margin / 25.0, 6),
                "schedule_penalty": round(sched_pen, 6),
                "resume": round(resume_part, 6),
                "z": {
                    "off_ppa": round(z.get("off_ppa", 0.0), 4),
                    "def_ppa": round(z.get("def_ppa", 0.0), 4),
                    "off_sr":  round(z.get("off_sr", 0.0), 4),
                    "def_sr":  round(z.get("def_sr", 0.0), 4),
                    "off_expl":round(z.get("off_expl",0.0), 4),
                    "def_expl":round(z.get("def_expl",0.0), 4),
                },
                "second_order": comp_second
            }
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    return rows

def enforce_loss_buckets(rows: List[dict]) -> List[dict]:
    rows.sort(key=lambda r: (r["losses"], -r["score"]))
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    return rows

def enforce_h2h_same_losses(rows: List[dict], teams: Dict[str, TeamRow]) -> List[dict]:
    pos = {r["team"]: i for i, r in enumerate(rows)}
    changed = True
    while changed:
        changed = False
        for i in range(len(rows)):
            A = rows[i]["team"]
            losses_A = rows[i]["losses"]
            for res, opp in teams[A].get("results", []):
                if res != "W" or opp not in pos:
                    continue
                j = pos[opp]
                if j <= i:
                    continue
                if rows[j]["losses"] == losses_A:
                    rows[i], rows[j] = rows[j], rows[i]
                    for k, r in enumerate(rows):
                        pos[r["team"]] = k
                        r["rank"] = k + 1
                    rows[pos[A]]["components"]["second_order"]["h2h_enforced"].append(f"{A} > {opp} (same-loss)")
                    changed = True
                    break
            if changed:
                break
    return rows

def apply_bucket_gate(rows: List[dict], teams: Dict[str, TeamRow], rmap_prov: Dict[str, int]) -> List[dict]:
    """
    Within each loss bucket:
      - split teams into GATE PASS and GATE FAIL using passes_gate()
      - keep internal order by score desc inside each partition
      - concatenate PASS first, then FAIL
    """
    # Pre-compute quality win counts from second_order details
    qwin_by_team: Dict[str, int] = {}
    for r in rows:
        sec = r["components"]["second_order"]
        q = sec.get("quality_wins", {})
        qwin_by_team[r["team"]] = int(q.get("top15", 0)) + int(q.get("r16_40", 0)) + int(q.get("r41_60", 0))

    # group by losses
    buckets: Dict[int, List[dict]] = {}
    for r in rows:
        buckets.setdefault(r["losses"], []).append(r)

    new_rows: List[dict] = []
    for losses in sorted(buckets.keys()):
        bucket = buckets[losses]
        # preserve score order
        bucket.sort(key=lambda x: -x["score"])
        passed, failed = [], []
        for r in bucket:
            t = r["team"]
            sos = r["sos"]
            qwins = qwin_by_team.get(t, 0)
            if passes_gate(teams[t], qwins, sos):
                passed.append(r)
            else:
                failed.append(r)
        new_rows.extend(passed + failed)

    # re-rank
    for i, r in enumerate(new_rows, start=1):
        r["rank"] = i
    return new_rows

# ====================== IO ======================

def write_json(payload: dict):
    os.makedirs("docs/data", exist_ok=True)
    with open("docs/data/rankings.json", "w") as f:
        json.dump(payload, f, indent=2)

# ====================== MAIN ======================

def main():
    print(f"Building FBS rankings (resume-first gate + strict H2H + loss buckets) for {YEAR}")
    fbs = fetch_fbs_names(YEAR)
    cur_week = fetch_current_week(YEAR)
    games_all = fetch_regular_games_by_week(YEAR, cur_week)
    games = keep_games_with_fbs_team(games_all, fbs)

    teams = roll_up(games, fbs)
    if not teams:
        print("❌ No completed games. Writing placeholder.")
        write_json({"season": YEAR, "last_build_utc": datetime.datetime.utcnow().isoformat(), "top25": []})
        return

    compute_sos_fbs_only(teams, fbs)
    adv = fetch_season_advanced(YEAR)
    zadv = standardize_advanced(adv, fbs)

    # Pass 1: base (no resume, no second-order)
    pass1 = score_pass(teams, zadv, rank_map=None, resume_map=None)
    rmap1 = make_rank_map(pass1)

    # Build resume from Pass 1
    resume = compute_resume(teams, rmap1, nteams=len(pass1))

    # Pass 2: add second-order and resume
    pass2 = score_pass(teams, zadv, rank_map=rmap1, resume_map=resume)

    # Structural ordering
    if LOSS_BUCKET_SORT:
        pass2 = enforce_loss_buckets(pass2)
    # Eligibility gate within each loss bucket
    pass2 = apply_bucket_gate(pass2, teams, rmap1)
    # Strict head-to-head inside bucket
    if H2H_SAME_LOSSES_STRICT:
        pass2 = enforce_h2h_same_losses(pass2, teams)

    out = {
        "season": YEAR,
        "last_build_utc": datetime.datetime.utcnow().isoformat(),
        "notes": {
            "weeks_included": f"1..{cur_week}",
            "weights": {
                "win_pct": W_WIN, "sos": W_SOS, "margin": W_MARGIN,
                "delta_ppa": W_PPA, "delta_sr": W_SR, "delta_expl": W_EX,
                "resume": W_RESUME
            },
            "tiers": {"quality": QW_TIER, "bad": BL_TIER},
            "rules": {
                "loss_bucket": True,
                "h2h_strict_same_losses": True,
                "weak_sos_penalty": WEAK_SOS_PENALTY,
                "bucket_gate": ">=1 quality win OR SoS>=0.52 OR (0-loss & SoS>=0.48)"
            }
        },
        "top25": pass2[:25]
    }
    write_json(out)
    print(f"✅ Top 25 built • teams={len(pass2)} • weeks=1..{cur_week}")

if __name__ == "__main__":
    main()
