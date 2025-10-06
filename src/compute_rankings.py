import os, json, datetime, requests, statistics, math

API_KEY = os.getenv("CFBD_API_KEY")
YEAR = int(os.getenv("YEAR") or datetime.datetime.now().year)
BASE_URL = "https://api.collegefootballdata.com"

def cfbd_get(endpoint, params=None):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    url = f"{BASE_URL}/{endpoint}"
    r = requests.get(url, headers=headers, params=params or {})
    if r.status_code != 200:
        print(f"⚠️ CFBD error {r.status_code}: {r.text[:200]}")
        return []
    return r.json()

def fetch_finished_games(year):
    all_games = []
    for week in range(1, 20):
        params = {"year": year, "seasonType": "regular", "week": week, "division": "fbs"}
        data = cfbd_get("games", params)
        finished = [g for g in data if g.get("home_points") is not None and g.get("away_points") is not None]
        all_games.extend(finished)
        print(f"✅ Week {week}: {len(finished)} finished games")
    print(f"Total finished games: {len(all_games)}")
    return all_games

def compute_team_metrics(games):
    teams = {}
    for g in games:
        home, away = g["home_team"], g["away_team"]
        h_pts, a_pts = g["home_points"], g["away_points"]
        if h_pts is None or a_pts is None:
            continue
        for t in [home, away]:
            if t not in teams:
                teams[t] = {"games": 0, "wins": 0, "points_for": 0, "points_against": 0, "sos": 0}
        teams[home]["games"] += 1
        teams[away]["games"] += 1
        teams[home]["points_for"] += h_pts
        teams[home]["points_against"] += a_pts
        teams[away]["points_for"] += a_pts
        teams[away]["points_against"] += h_pts
        if h_pts > a_pts:
            teams[home]["wins"] += 1
        else:
            teams[away]["wins"] += 1
    return teams

def compute_sos(teams, games):
    for team in teams:
        opps = []
        for g in games:
            if g["home_team"] == team:
                opps.append(g["away_team"])
            elif g["away_team"] == team:
                opps.append(g["home_team"])
        if not opps:
            continue
        opp_win_rates = [teams[o]["wins"] / max(teams[o]["games"], 1) for o in opps if o in teams]
        teams[team]["sos"] = statistics.mean(opp_win_rates) if opp_win_rates else 0
    return teams

def build_rankings(teams):
    rankings = []
    for t, data in teams.items():
        if data["games"] < 3:
            continue
        win_pct = data["wins"] / data["games"]
        margin = (data["points_for"] - data["points_against"]) / data["games"]
        score = (win_pct * 0.6) + (data["sos"] * 0.3) + ((margin / 25) * 0.1)
        rankings.append({"team": t, "score": score, **data})
    rankings.sort(key=lambda x: x["score"], reverse=True)
    return rankings[:25]

def main():
    print(f"Building rankings for {YEAR}")
    games = fetch_finished_games(YEAR)
    if not games:
        print("❌ No games found.")
        return
    teams = compute_team_metrics(games)
    teams = compute_sos(teams, games)
    top25 = build_rankings(teams)
    result = {
        "season": YEAR,
        "last_build_utc": datetime.datetime.utcnow().isoformat(),
        "top25": top25
    }
    os.makedirs("docs/data", exist_ok=True)
    with open("docs/data/rankings.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"✅ Rankings built for {len(top25)} teams")

if __name__ == "__main__":
    main()
