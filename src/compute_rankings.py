#!/usr/bin/env python3
"""
BCS-style deterministic CFB Top 25 (FBS résumé) — SOS heavier
Ordering:
  (A) Loss buckets (fewest total losses first; records include FCS opponents)
  (B) Head-to-head inside the same loss bucket (strict swap)
  (C) Composite score (BCS-like components) with heavier SOS:
        • Win% (overall FBS team record, includes FCS)                      [W_WINPCT]
        • SOS1: Opponent strength (avg FBS-opponent win%)                   [W_SOS1]
        • SOS2: Opponent’s opponent strength                                [W_SOS2]
        • Quality wins / Bad losses                                         [W_QUAL, W_BAD]
        • Average scoring margin (capped influence)                          [W_MARGIN]
        • Location adjustment (road/neutral/home)                            (direct nudge)
        • Tiny efficiency seasoning (off_ppa-def_ppa, off_sr-def_sr)        [W_ADV_*]
        • OPTIONAL: Polls (AP/Coaches) tiny nudge                           [POLLS_WEIGHT]

Notes:
  - Résumé math (SOS, H2H, QW/BL, location) uses only FBS vs FBS.
  - Overall record counts all games an FBS team played (so it matches public records).
  - Stronger penalty for weak schedules near/below average (W_WEAK_SOS more negative).
  - Weights are configurable via environment variables; see defaults below.
"""

import os, json, datetime, requests, math, re
from collections import defaultdict

# --------- Env + defaults ----------
def _f(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)

def _i(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)

CFBD_API_KEY = os.getenv("CFBD_API_KEY")
if not CFBD_API_KEY:
    raise RuntimeError("CFBD_API_KEY is not set")

def _safe_int(x, fallback):
    try:
        return int(x)
    except Exception:
        return fallback

YEAR = _safe_int(os.getenv("YEAR"), datetime.datetime.now().year)

# Heavier SOS defaults (tunable via env)
W_WINPCT   = _f("W_WINPCT",   0.36)   # was 0.44
W_SOS1     = _f("W_SOS1",     0.34)   # was 0.22
W_SOS2     = _f("W_SOS2",     0.14)   # was 0.10
W_MARGIN   = _f("W_MARGIN",   0.06)   # was 0.08
W_QUAL     = _f("W_QUAL",     0.020)  # per quality win (up from 0.015)
W_BAD      = _f("W_BAD",     -0.030)  # per bad loss (more negative)
W_ADV_PPA  = _f("W_ADV_PPA",  0.018)
W_ADV_SR   = _f("W_ADV_SR",   0.010)
W_WEAK_SOS = _f("W_WEAK_SOS", -0.100) # stronger penalty for SOS1 < ~0.52
W_CONF_CH  = _f("W_CONF_CH",  0.010)
POLLS_WEIGHT = _f("POLLS_WEIGHT", 0.08)  # small; set 0 to disable

# Location nudges (kept modest)
ROAD_WIN_BONUS   = _f("ROAD_WIN_BONUS",    0.007)
ROAD_LOSS_PEN    = _f("ROAD_LOSS_PEN",    -0.010)
HOME_WIN_BONUS   = _f("HOME_WIN_BONUS",    0.000)
HOME_LOSS_PEN    = _f("HOME_LOSS_PEN",    -0.012)
NEUTRAL_WIN_BONUS= _f("NEUTRAL_WIN_BONUS", 0.003)
NEUTRAL_LOSS_PEN = _f("NEUTRAL_LOSS_PEN", -0.006)

HEADERS = {"Authorization": f"Bearer {CFBD_API_KEY}"}
DATA_DIR = "docs/data"
os.makedirs(DATA_DIR, exist_ok=True)
OUT_FILE = os.path.join(DATA_DIR, "rankings.json")

# ---------- helpers ----------
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
        hp = g.get("homePoints"); ap = g.get("awayPoints")
        if hp is None or ap is None:
            continue
        out.append({
            "id": g.get("id"),
            "week": g.get("week"),
            "home": g.get("homeTeam"),
            "away": g.get("awayTeam"),
            "home_conf": g.get("homeConference"),
            "away_conf": g.get("awayConference"),
            "hp": safe_float(hp),
            "ap": safe_float(ap),
            "neutral": bool(g.get("neutralSite")),
            "venue": g.get("venue"),
            "notes": g.get("notes") or ""
        })
    return out

def get_postseason_games(year: int):
    try:
        rows = fetch("games", {"year": year, "seasonType": "postseason"})
    except Exception:
        rows = []
    out = []
    for g in rows:
        hp = g.get("homePoints"); ap = g.get("awayPoints")
        if hp is None or ap is None:
            continue
        out.append({
            "id": g.get("id"),
            "week": g.get("week"),
            "home": g.get("homeTeam"),
            "away": g.get("awayTeam"),
            "home_conf": g.get("homeConference"),
            "away_conf": g.get("awayConference"),
            "hp": safe_float(hp),
            "ap": safe_float(ap),
            "neutral": bool(g.get("neutralSite")),
            "venue": g.get("venue"),
            "notes": g.get("notes") or ""
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

def latest_polls(year: int):
    """Return dict: { team: {ap: rank|None, coaches: rank|None} } from /rankings."""
    out = defaultdict(lambda: {"ap": None, "coaches": None})
    if POLLS_WEIGHT <= 0:
        return out
    try:
        rows = fetch("rankings", {"year": year})
    except Exception:
        return out

    for wk in rows:
        polls = wk.get("polls", [])
        for p in polls:
            poll = (p.get("poll") or "").lower()
            ranks = p.get("ranks") or []
            if poll == "ap top 25":
                for r in ranks:
                    team = r.get("school"); rank = r.get("rank")
                    if team and rank: out[team]["ap"] = int(rank)
            elif poll == "coaches poll":
                for r in ranks:
                    team = r.get("school"); rank = r.get("rank")
                    if team and rank: out[team]["coaches"] = int(rank)
    return out

def rollup_regular(games: list, fbs_set: set):
    """
    Build per-team tallies:
      - wins/losses (overall, includes FCS opponents when FBS team involved)
      - FBS-only wins/losses, H2H map, opponent sets, location adj
    """
    teams = {}
    def ensure(t):
        if t not in teams:
            teams[t] = {
                "wins": 0, "losses": 0, "pf": 0.0, "pa": 0.0,
                "fbs_wins": 0, "fbs_losses": 0,
                "fbs_opps": set(),
                "results_fbs": [],
                "results_all": [],
                "loc_adj": 0.0
            }

    for g in games:
        h, a, hp, ap = g["home"], g["away"], g["hp"], g["ap"]
        if not h or not a: continue
        h_fbs = h in fbs_set; a_fbs = a in fbs_set
        if not (h_fbs or a_fbs): continue

        if h_fbs: ensure(h)
        if a_fbs: ensure(a)

        # Overall record (FBS team counts all games)
        if h_fbs:
            teams[h]["pf"] += hp; teams[h]["pa"] += ap
            if hp > ap: teams[h]["wins"] += 1; teams[h]["results_all"].append(("W", a))
            elif ap > hp: teams[h]["losses"] += 1; teams[h]["results_all"].append(("L", a))
        if a_fbs:
            teams[a]["pf"] += ap; teams[a]["pa"] += hp
            if ap > hp: teams[a]["wins"] += 1; teams[a]["results_all"].append(("W", h))
            elif hp > ap: teams[a]["losses"] += 1; teams[a]["results_all"].append(("L", h))

        # FBS résumé + location nudges
        if h_fbs and a_fbs:
            teams[h]["fbs_opps"].add(a); teams[a]["fbs_opps"].add(h)
            if hp > ap:
                teams[h]["fbs_wins"] += 1; teams[a]["fbs_losses"] += 1
                teams[h]["results_fbs"].append(("W", a)); teams[a]["results_fbs"].append(("L", h))
                if g["neutral"]:
                    teams[h]["loc_adj"] += NEUTRAL_WIN_BONUS
                    teams[a]["loc_adj"] += NEUTRAL_LOSS_PEN
                else:
                    teams[h]["loc_adj"] += HOME_WIN_BONUS
                    teams[a]["loc_adj"] += ROAD_LOSS_PEN
            elif ap > hp:
                teams[a]["fbs_wins"] += 1; teams[h]["fbs_losses"] += 1
                teams[a]["results_fbs"].append(("W", h)); teams[h]["results_fbs"].append(("L", a))
                if g["neutral"]:
                    teams[a]["loc_adj"] += NEUTRAL_WIN_BONUS
                    teams[h]["loc_adj"] += NEUTRAL_LOSS_PEN
                else:
                    teams[a]["loc_adj"] += ROAD_WIN_BONUS
                    teams[h]["loc_adj"] += HOME_LOSS_PEN
    return teams

def detect_conf_champs(postseason_games, fbs_set):
    """Identify winners of games labeled like 'Championship'."""
    champs = set()
    for g in postseason_games:
        h, a, hp, ap = g["home"], g["away"], g["hp"], g["ap"]
        if not (h in fbs_set and a in fbs_set):
            continue
        txt = (g.get("notes") or "") + " " + (g.get("venue") or "")
        if re.search(r"Championship", txt, re.IGNORECASE):
            if hp > ap: champs.add(h)
            elif ap > hp: champs.add(a)
    return champs

def compute_sos(teams: dict):
    """ SOS1: avg FBS-opponent win%; SOS2: avg of opponents' SOS1. """
    fbs_wpct = {}
    for t, d in teams.items():
        g = d["fbs_wins"] + d["fbs_losses"]
        fbs_wpct[t] = (d["fbs_wins"] / g) if g > 0 else 0.5

    for t, d in teams.items():
        opps = [o for o in d["fbs_opps"] if o in teams]
        d["sos1"] = (sum(fbs_wpct[o] for o in opps) / len(opps)) if opps else 0.5

    for t, d in teams.items():
        opps = [o for o in d["fbs_opps"] if o in teams]
        d["sos2"] = (sum(teams[o]["sos1"] for o in opps) / len(opps)) if opps else 0.5

def provisional_seed(teams):
    """Seed list for QW/BL classification (not final rank)."""
    seed = []
    for t, d in teams.items():
        games_cnt = d["wins"] + d["losses"]
        win_pct = (d["wins"] / games_cnt) if games_cnt > 0 else 0.0
        seed.append((t, 0.55 * win_pct + 0.45 * d.get("sos1", 0.5)))  # lean a bit more to SOS even in seed
    seed.sort(key=lambda x: x[1], reverse=True)
    return {t: i + 1 for i, (t, _) in enumerate(seed)}

def build_qw_bl(teams, seed_rank):
    qual = defaultdict(int); bad = defaultdict(int)
    for t, d in teams.items():
        for res, opp in d["results_fbs"]:
            r = seed_rank.get(opp, 999)
            if res == "W":
                if r <= 40: qual[t] += 1
            elif res == "L":
                if r >= 80: bad[t] += 1
    return qual, bad

def polls_component_for(team, polls_map):
    if POLLS_WEIGHT <= 0:
        return None, None, None
    entry = polls_map.get(team, {})
    ap = entry.get("ap"); coaches = entry.get("coaches")
    def nrm(rank):
        if rank is None: return None
        if rank > 25: return 0.0
        return (26 - rank) / 25.0
    ap_n = nrm(ap); co_n = nrm(coaches)
    if ap_n is None and co_n is None:
        return None, ap, coaches
    poll_comp = co_n if ap_n is None else ap_n if co_n is None else 0.5 * (ap_n + co_n)
    return poll_comp, ap, coaches

def composite_for_team(t, d, qual, bad, adv, poll_comp, conf_champ_winner):
    games = d["wins"] + d["losses"]
    win_pct = (d["wins"] / games) if games > 0 else 0.0
    sos1 = d.get("sos1", 0.5)
    sos2 = d.get("sos2", 0.5)
    avg_margin = ((d["pf"] - d["pa"]) / games) if games > 0 else 0.0
    avg_margin_norm = avg_margin / 28.0

    adv_term = 0.0
    if adv:
        delta_ppa = max(-1.0, min(1.0, adv.get("off_ppa", 0.0) - adv.get("def_ppa", 0.0)))
        delta_sr  = max(-1.0, min(1.0,  adv.get("off_sr", 0.0)  - adv.get("def_sr", 0.0)))
        adv_term = W_ADV_PPA * delta_ppa + W_ADV_SR * delta_sr

    weak_pen = W_WEAK_SOS * max(0.0, 0.52 - sos1)   # punish cupcake slates harder
    conf_bonus = W_CONF_CH if conf_champ_winner else 0.0

    score = (
        W_WINPCT * win_pct +
        W_SOS1   * sos1 +
        W_SOS2   * sos2 +
        W_MARGIN * avg_margin_norm +
        W_QUAL   * qual +
        W_BAD    * bad +
        d.get("loc_adj", 0.0) +    # direct small nudges
        adv_term +
        weak_pen +
        conf_bonus +
        (POLLS_WEIGHT * poll_comp if poll_comp is not None else 0.0)
    )
    return max(0.0, min(1.0, score))

def human_readable_why(parts):
    reasons = []
    reasons.append(f"Record strength: {parts['win_pct']:.1%} win rate.")
    reasons.append(f"Schedule strength: SOS1 {parts['sos1']:.3f} (opponents), SOS2 {parts['sos2']:.3f} (opponents’ opponents).")
    reasons.append(f"Scoring margin: {parts['avg_margin']:.1f} points per game.")
    if parts["qual_wins"] > 0: reasons.append(f"Quality wins: {parts['qual_wins']} vs roughly Top-40 teams.")
    if parts["bad_losses"] > 0: reasons.append(f"Bad losses: {parts['bad_losses']} vs ~80+ teams.")
    if abs(parts["loc_adj"]) >= 0.004:
        reasons.append("Road/neutral outcomes added a location adjustment." if parts["loc_adj"] > 0
                       else "Home/road outcomes incurred a small location penalty.")
    if parts.get("conf_champ", False):
        reasons.append("Conference championship win adds a small bonus.")
    if parts.get("poll_ap") or parts.get("poll_coaches"):
        reasons.append("Tiny human-poll nudge applied (does not override résumé).")
    return reasons

def build_rankings(year: int):
    print(f"Building heavier-SOS FBS rankings for {year}")
    fbs_set = get_fbs_set(year)
    reg = get_regular_games(year)
    post = get_postseason_games(year)
    if not fbs_set or not reg:
        return {"season": year, "last_build_utc": datetime.datetime.utcnow().isoformat(), "top25": []}

    teams = rollup_regular(reg, fbs_set)
    if not teams:
        return {"season": year, "last_build_utc": datetime.datetime.utcnow().isoformat(), "top25": []}

    compute_sos(teams)
    seed_rank = provisional_seed(teams)
    qual_map, bad_map = build_qw_bl(teams, seed_rank)
    adv_all = get_season_advanced(year)
    champs = detect_conf_champs(post, fbs_set) if post else set()
    polls_map = latest_polls(year)

    rows = []
    for t, d in teams.items():
        poll_comp, ap, co = polls_component_for(t, polls_map)
        score = composite_for_team(
            t, d,
            qual=qual_map[t], bad=bad_map[t],
            adv=adv_all.get(t, {}),
            poll_comp=poll_comp,
            conf_champ_winner=(t in champs)
        )
        games = d["wins"] + d["losses"]
        avg_margin = ((d["pf"] - d["pa"]) / games) if games > 0 else 0.0
        parts = {
            "win_pct": (d["wins"] / games) if games > 0 else 0.0,
            "sos1": d.get("sos1", 0.5),
            "sos2": d.get("sos2", 0.5),
            "avg_margin": avg_margin,
            "qual_wins": qual_map[t],
            "bad_losses": bad_map[t],
            "loc_adj": d.get("loc_adj", 0.0),
            "poll_ap": ap,
            "poll_coaches": co,
            "conf_champ": (t in champs)
        }
        rows.append({
            "team": t,
            "wins": d["wins"],
            "losses": d["losses"],
            "games": games,
            "points_for": int(d["pf"]),
            "points_against": int(d["pa"]),
            "sos": round(d.get("sos1", 0.5), 6),
            "sos2": round(d.get("sos2", 0.5), 6),
            "avg_margin": round(avg_margin, 3),
            "fbs_wins": d["fbs_wins"],
            "fbs_losses": d["fbs_losses"],
            "qual_wins": qual_map[t],
            "bad_losses": bad_map[t],
            "location_adj": round(d.get("loc_adj", 0.0), 6),
            "conf_champ": (t in champs),
            "poll_ap": ap,
            "poll_coaches": co,
            "score": round(score, 6),
            "off_ppa": adv_all.get(t, {}).get("off_ppa", 0.0),
            "def_ppa": adv_all.get(t, {}).get("def_ppa", 0.0),
            "off_sr":  adv_all.get(t, {}).get("off_sr", 0.0),
            "def_sr":  adv_all.get(t, {}).get("def_sr", 0.0),
            "why": human_readable_why(parts)
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
        "sos2": round(r["sos2"], 3),
        "avg_margin": round(r["avg_margin"], 1),
        "qual_wins": r["qual_wins"],
        "bad_losses": r["bad_losses"],
        "location_adj": r["location_adj"],
        "conf_champ": r["conf_champ"],
        "poll_ap": r["poll_ap"],
        "poll_coaches": r["poll_coaches"],
        "off_ppa": r["off_ppa"],
        "def_ppa": r["def_ppa"],
        "off_sr": r["off_sr"],
        "def_sr": r["def_sr"],
        "why": r["why"]
    } for r in rows[:25]]

    return {
        "season": year,
        "last_build_utc": datetime.datetime.utcnow().isoformat(),
        "notes": {
            "ordering": "Loss buckets > H2H within bucket > composite (heavier SOS).",
            "weights": {
                "win_pct": W_WINPCT, "sos1": W_SOS1, "sos2": W_SOS2, "margin": W_MARGIN,
                "qual_each": W_QUAL, "bad_each": W_BAD,
                "adv_ppa": W_ADV_PPA, "adv_sr": W_ADV_SR,
                "weak_sos_pen": W_WEAK_SOS, "conf_champ": W_CONF_CH,
                "polls_weight": POLLS_WEIGHT
            }
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
