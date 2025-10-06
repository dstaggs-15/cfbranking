#!/usr/bin/env python3
"""
CFB Top 25 — SOS-first composite (no hard loss-bucket ordering)

Key changes vs prior version:
  • Sort by COMPOSITE SCORE directly (not by losses first).
  • Heavier SOS (SOS1 + SOS2) weights.
  • Undefeated soft-schedule brake so 5–0 cupcakes don't sit over elite 4–1's.
  • Head-to-head is a SOFT tiebreaker: only flips when scores are basically tied.

Résumé math is still FBS-vs-FBS for fairness. Overall records include FCS so TV
records match fan expectations.

Env knobs (no code edits needed):
  W_WINPCT, W_SOS1, W_SOS2, W_MARGIN, W_QUAL, W_BAD, W_ADV_PPA, W_ADV_SR,
  W_WEAK_SOS, W_CONF_CH, POLLS_WEIGHT,
  H2H_EPS, UNDEF_SOS_BRAKE, UNDEF_SOS_EDGE, YEAR
"""

import os, json, datetime, requests, math, re
from collections import defaultdict

# ---------- ENV & DEFAULTS ----------
def _f(name, default):
    try: return float(os.getenv(name, str(default)))
    except Exception: return float(default)

def _i(name, default):
    try: return int(os.getenv(name, str(default)))
    except Exception: return int(default)

CFBD_API_KEY = os.getenv("CFBD_API_KEY")
if not CFBD_API_KEY:
    raise RuntimeError("CFBD_API_KEY is not set")

def _safe_int(x, fallback):
    try: return int(x)
    except Exception: return fallback

YEAR = _safe_int(os.getenv("YEAR"), datetime.datetime.now().year)

# Heavier SOS defaults
W_WINPCT   = _f("W_WINPCT",   0.32)
W_SOS1     = _f("W_SOS1",     0.40)
W_SOS2     = _f("W_SOS2",     0.18)
W_MARGIN   = _f("W_MARGIN",   0.05)
W_QUAL     = _f("W_QUAL",     0.020)   # per quality win
W_BAD      = _f("W_BAD",     -0.030)   # per bad loss
W_ADV_PPA  = _f("W_ADV_PPA",  0.015)
W_ADV_SR   = _f("W_ADV_SR",   0.008)
W_WEAK_SOS = _f("W_WEAK_SOS", -0.10)   # penalty for SOS1 below ~0.52
W_CONF_CH  = _f("W_CONF_CH",  0.010)
POLLS_WEIGHT = _f("POLLS_WEIGHT", 0.04)  # set 0 to disable

# Location nudges
ROAD_WIN_BONUS    = _f("ROAD_WIN_BONUS",    0.007)
ROAD_LOSS_PEN     = _f("ROAD_LOSS_PEN",    -0.010)
HOME_WIN_BONUS    = _f("HOME_WIN_BONUS",    0.000)
HOME_LOSS_PEN     = _f("HOME_LOSS_PEN",    -0.012)
NEUTRAL_WIN_BONUS = _f("NEUTRAL_WIN_BONUS", 0.003)
NEUTRAL_LOSS_PEN  = _f("NEUTRAL_LOSS_PEN", -0.006)

# New: soft tiebreak and undefeated brake
H2H_EPS         = _f("H2H_EPS", 0.015)     # only flip on H2H if score diff < this
UNDEF_SOS_BRAKE = _f("UNDEF_SOS_BRAKE", -0.06)  # extra penalty for undefeated w/ weak SOS
UNDEF_SOS_EDGE  = _f("UNDEF_SOS_EDGE", 0.52)    # "average" SOS threshold

HEADERS = {"Authorization": f"Bearer {CFBD_API_KEY}"}
DATA_DIR = "docs/data"
os.makedirs(DATA_DIR, exist_ok=True)
OUT_FILE = os.path.join(DATA_DIR, "rankings.json")

# ---------- HTTP helpers ----------
def safe_float(x, default=0.0):
    try: return float(x)
    except Exception: return float(default)

def fetch(endpoint: str, params=None):
    url = f"https://api.collegefootballdata.com/{endpoint}"
    r = requests.get(url, headers=HEADERS, params=params or {}, timeout=45)
    r.raise_for_status()
    return r.json()

# ---------- Data pulls ----------
def get_fbs_set(year: int) -> set:
    rows = fetch("teams/fbs", {"year": year})
    return {row.get("school") for row in rows if row.get("school")}

def get_regular_games(year: int):
    rows = fetch("games", {"year": year, "seasonType": "regular"})
    out = []
    for g in rows:
        hp = g.get("homePoints"); ap = g.get("awayPoints")
        if hp is None or ap is None: continue
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
        if hp is None or ap is None: continue
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
    try:
        rows = fetch("stats/season/advanced", {"year": year})
    except Exception:
        return {}
    out = {}
    for r in rows:
        t = r.get("team")
        if not t: continue
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
    out = defaultdict(lambda: {"ap": None, "coaches": None})
    if POLLS_WEIGHT <= 0: return out
    try:
        rows = fetch("rankings", {"year": year})
    except Exception:
        return out
    for wk in rows:
        for p in (wk.get("polls") or []):
            name = (p.get("poll") or "").lower()
            ranks = p.get("ranks") or []
            if name == "ap top 25":
                for r in ranks:
                    t = r.get("school"); k = r.get("rank")
                    if t and k: out[t]["ap"] = int(k)
            elif name == "coaches poll":
                for r in ranks:
                    t = r.get("school"); k = r.get("rank")
                    if t and k: out[t]["coaches"] = int(k)
    return out

# ---------- Build résumé ----------
def rollup_regular(games, fbs_set):
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
        h_fbs, a_fbs = h in fbs_set, a in fbs_set
        if not (h_fbs or a_fbs): continue

        if h_fbs: ensure(h)
        if a_fbs: ensure(a)

        # overall record & points (FBS teams count all games to match public records)
        if h_fbs:
            teams[h]["pf"] += hp; teams[h]["pa"] += ap
            (teams[h]["wins"] if hp>ap else teams[h]["losses"]).__iadd__(1) if hp!=ap else None
            teams[h]["results_all"].append(("W" if hp>ap else "L", a) if hp!=ap else ("", a))
        if a_fbs:
            teams[a]["pf"] += ap; teams[a]["pa"] += hp
            (teams[a]["wins"] if ap>hp else teams[a]["losses"]).__iadd__(1) if ap!=hp else None
            teams[a]["results_all"].append(("W" if ap>hp else "L", h) if ap!=hp else ("", h))

        # FBS résumé + location
        if h_fbs and a_fbs:
            teams[h]["fbs_opps"].add(a); teams[a]["fbs_opps"].add(h)
            if hp > ap:
                teams[h]["fbs_wins"] += 1; teams[a]["fbs_losses"] += 1
                teams[h]["results_fbs"].append(("W", a)); teams[a]["results_fbs"].append(("L", h))
                if g["neutral"]:
                    teams[h]["loc_adj"] += NEUTRAL_WIN_BONUS; teams[a]["loc_adj"] += NEUTRAL_LOSS_PEN
                else:
                    teams[h]["loc_adj"] += HOME_WIN_BONUS; teams[a]["loc_adj"] += ROAD_LOSS_PEN
            elif ap > hp:
                teams[a]["fbs_wins"] += 1; teams[h]["fbs_losses"] += 1
                teams[a]["results_fbs"].append(("W", h)); teams[h]["results_fbs"].append(("L", a))
                if g["neutral"]:
                    teams[a]["loc_adj"] += NEUTRAL_WIN_BONUS; teams[h]["loc_adj"] += NEUTRAL_LOSS_PEN
                else:
                    teams[a]["loc_adj"] += ROAD_WIN_BONUS; teams[h]["loc_adj"] += HOME_LOSS_PEN
    return teams

def detect_conf_champs(postseason_games, fbs_set):
    champs = set()
    for g in postseason_games:
        h, a, hp, ap = g["home"], g["away"], g["hp"], g["ap"]
        if not (h in fbs_set and a in fbs_set): continue
        txt = (g.get("notes") or "") + " " + (g.get("venue") or "")
        if re.search(r"Championship", txt, re.IGNORECASE):
            champs.add(h if hp>ap else a)
    return champs

def compute_sos(teams):
    # SOS1: avg opponent FBS win%; SOS2: avg of opponents' SOS1
    fbs_wpct = {}
    for t, d in teams.items():
        g = d["fbs_wins"] + d["fbs_losses"]
        fbs_wpct[t] = (d["fbs_wins"]/g) if g>0 else 0.5
    for t, d in teams.items():
        opps = [o for o in d["fbs_opps"] if o in teams]
        d["sos1"] = sum(fbs_wpct[o] for o in opps)/len(opps) if opps else 0.5
    for t, d in teams.items():
        opps = [o for o in d["fbs_opps"] if o in teams]
        d["sos2"] = sum(teams[o]["sos1"] for o in opps)/len(opps) if opps else 0.5

def provisional_seed(teams):
    seed = []
    for t, d in teams.items():
        games = d["wins"] + d["losses"]
        wp = (d["wins"]/games) if games>0 else 0.0
        seed.append((t, 0.5*wp + 0.5*d.get("sos1",0.5)))  # seed only for QW/BL buckets
    seed.sort(key=lambda x: x[1], reverse=True)
    return {t:i+1 for i,(t,_) in enumerate(seed)}

def build_qw_bl(teams, seed_rank):
    qual = defaultdict(int); bad = defaultdict(int)
    for t, d in teams.items():
        for res, opp in d["results_fbs"]:
            r = seed_rank.get(opp, 999)
            if res == "W" and r <= 40: qual[t] += 1
            if res == "L" and r >= 80: bad[t]  += 1
    return qual, bad

def polls_component_for(team, polls_map):
    if POLLS_WEIGHT <= 0: return None, None, None
    entry = polls_map.get(team, {})
    ap = entry.get("ap"); coaches = entry.get("coaches")
    def nrm(rank):
        if rank is None: return None
        if rank > 25: return 0.0
        return (26 - rank)/25.0
    ap_n = nrm(ap); co_n = nrm(coaches)
    if ap_n is None and co_n is None: return None, ap, coaches
    poll_comp = co_n if ap_n is None else ap_n if co_n is None else 0.5*(ap_n+co_n)
    return poll_comp, ap, coaches

def composite_for_team(team, d, qual, bad, adv, poll_comp, conf_champ_winner):
    games = d["wins"] + d["losses"]
    win_pct = (d["wins"]/games) if games>0 else 0.0
    sos1 = d.get("sos1", 0.5); sos2 = d.get("sos2", 0.5)
    avg_margin = ((d["pf"]-d["pa"])/games) if games>0 else 0.0
    avg_margin_norm = avg_margin / 28.0

    # tiny efficiency seasoning
    adv_term = 0.0
    if adv:
        delta_ppa = max(-1.0, min(1.0, adv.get("off_ppa",0.0) - adv.get("def_ppa",0.0)))
        delta_sr  = max(-1.0, min(1.0, adv.get("off_sr",0.0)  - adv.get("def_sr",0.0)))
        adv_term = W_ADV_PPA*delta_ppa + W_ADV_SR*delta_sr

    weak_pen = W_WEAK_SOS * max(0.0, UNDEF_SOS_EDGE - sos1)

    # NEW: Undefeated soft-schedule brake
    undefeated = (d["losses"] == 0 and games > 0)
    undef_pen = 0.0
    if undefeated and sos1 < UNDEF_SOS_EDGE:
        # scales with how far below average the schedule is
        gap = (UNDEF_SOS_EDGE - sos1)  # ~0..0.2 typical early
        undef_pen = UNDEF_SOS_BRAKE * gap  # negative

    conf_bonus = W_CONF_CH if conf_champ_winner else 0.0

    score = (
        W_WINPCT*win_pct +
        W_SOS1*sos1 + W_SOS2*sos2 +
        W_MARGIN*avg_margin_norm +
        W_QUAL*qual + W_BAD*bad +
        d.get("loc_adj", 0.0) +
        adv_term + weak_pen + undef_pen + conf_bonus +
        (POLLS_WEIGHT * poll_comp if poll_comp is not None else 0.0)
    )
    # clamp
    return max(0.0, min(1.0, score)), {
        "win_pct": win_pct, "sos1": sos1, "sos2": sos2, "avg_margin": avg_margin,
        "qual_wins": qual, "bad_losses": bad, "loc_adj": d.get("loc_adj",0.0),
        "undef_pen": undef_pen, "weak_pen": weak_pen,
        "conf_champ": conf_champ_winner
    }

def human_readable_why(parts, ap, co):
    reasons = []
    reasons.append(f"Record strength: {parts['win_pct']:.1%} win rate.")
    reasons.append(f"Schedule strength: SOS1 {parts['sos1']:.3f} (opponents), SOS2 {parts['sos2']:.3f} (opponents’ opponents).")
    reasons.append(f"Scoring margin: {parts['avg_margin']:.1f} points per game.")
    if parts["qual_wins"] > 0: reasons.append(f"Quality wins: {parts['qual_wins']} vs roughly Top-40.")
    if parts["bad_losses"] > 0: reasons.append(f"Bad losses: {parts['bad_losses']} vs ~80+.")
    if abs(parts["loc_adj"]) >= 0.004:
        reasons.append("Road/neutral results improved résumé." if parts["loc_adj"] > 0 else "Home/road results slightly penalized résumé.")
    if parts["conf_champ"]: reasons.append("Conference championship adds a small bonus.")
    if parts["undef_pen"] < -1e-9:
        reasons.append("Undefeated vs softer schedule received a small adjustment.")
    if POLLS_WEIGHT > 0 and (ap or co):
        reasons.append("Tiny poll nudge applied; résumé remains primary.")
    return reasons

# ---------- Build rankings ----------
def build_rankings(year: int):
    print(f"Building SOS-first composite rankings for {year}")
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
        score, parts = composite_for_team(
            t, d,
            qual=qual_map[t], bad=bad_map[t],
            adv=adv_all.get(t, {}),
            poll_comp=poll_comp,
            conf_champ_winner=(t in champs)
        )
        games = d["wins"] + d["losses"]
        avg_margin = parts["avg_margin"]
        rows.append({
            "team": t,
            "wins": d["wins"], "losses": d["losses"], "games": games,
            "points_for": int(d["pf"]), "points_against": int(d["pa"]),
            "sos": round(parts["sos1"], 6), "sos2": round(parts["sos2"], 6),
            "avg_margin": round(avg_margin, 3),
            "fbs_wins": d["fbs_wins"], "fbs_losses": d["fbs_losses"],
            "qual_wins": qual_map[t], "bad_losses": bad_map[t],
            "location_adj": round(parts["loc_adj"], 6),
            "conf_champ": (t in champs),
            "poll_ap": polls_map.get(t, {}).get("ap"),
            "poll_coaches": polls_map.get(t, {}).get("coaches"),
            "off_ppa": adv_all.get(t, {}).get("off_ppa", 0.0),
            "def_ppa": adv_all.get(t, {}).get("def_ppa", 0.0),
            "off_sr":  adv_all.get(t, {}).get("off_sr", 0.0),
            "def_sr":  adv_all.get(t, {}).get("def_sr", 0.0),
            "score": round(score, 6),
            "why": human_readable_why(parts, polls_map.get(t, {}).get("ap"), polls_map.get(t, {}).get("coaches"))
        })

    # PRIMARY ORDER: composite score (desc)
    rows.sort(key=lambda r: (-r["score"], -r["fbs_wins"], r["losses"], r["team"]))

    # SOFT H2H tiebreak if basically tied
    # Build a quick lookup of FBS results
    results_map = defaultdict(dict)
    for t, d in teams.items():
        for res, opp in d["results_fbs"]:
            results_map[t][opp] = res

    changed = True
    while changed:
        changed = False
        for i in range(len(rows)-1):
            A, B = rows[i], rows[i+1]
            diff = abs(A["score"] - B["score"])
            if diff <= H2H_EPS:
                a, b = A["team"], B["team"]
                if results_map.get(a, {}).get(b) == "L":  # B beat A
                    rows[i], rows[i+1] = rows[i+1], rows[i]
                    changed = True

    # rank labels
    for i, r in enumerate(rows, start=1):
        r["rank"] = i

    top25 = [{
        "rank": r["rank"], "team": r["team"], "score": round(r["score"], 4),
        "games": r["games"], "wins": r["wins"], "losses": r["losses"],
        "points_for": r["points_for"], "points_against": r["points_against"],
        "sos": round(r["sos"], 3), "sos2": round(r["sos2"], 3),
        "avg_margin": round(r["avg_margin"], 1),
        "qual_wins": r["qual_wins"], "bad_losses": r["bad_losses"],
        "location_adj": r["location_adj"], "conf_champ": r["conf_champ"],
        "poll_ap": r["poll_ap"], "poll_coaches": r["poll_coaches"],
        "off_ppa": r["off_ppa"], "def_ppa": r["def_ppa"],
        "off_sr": r["off_sr"], "def_sr": r["def_sr"],
        "why": r["why"]
    } for r in rows[:25]]

    return {
        "season": year,
        "last_build_utc": datetime.datetime.utcnow().isoformat(),
        "notes": {
            "ordering": "Composite (SOS-first) > soft H2H if within epsilon (no hard loss buckets).",
            "weights": {
                "win_pct": W_WINPCT, "sos1": W_SOS1, "sos2": W_SOS2, "margin": W_MARGIN,
                "qual_each": W_QUAL, "bad_each": W_BAD,
                "adv_ppa": W_ADV_PPA, "adv_sr": W_ADV_SR,
                "weak_sos_pen": W_WEAK_SOS, "undef_brake": UNDEF_SOS_BRAKE,
                "undef_edge": UNDEF_SOS_EDGE, "conf_champ": W_CONF_CH,
                "polls_weight": POLLS_WEIGHT, "h2h_eps": H2H_EPS
            }
        },
        "top25": top25
    }

def main():
    data = build_rankings(YEAR)
    with open(OUT_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Wrote", OUT_FILE)

if __name__ == "__main__":
    main()
