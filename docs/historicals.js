const INDEX_URL = "./data/historicals/index.json?v=" + Date.now();
const CURRENT_URL = "./data/rankings.json?v=" + Date.now();

const seasonSelect = document.getElementById("season-select");
const snapshotSelect = document.getElementById("snapshot-select");
const teamSearch = document.getElementById("team-search");
const teamOptions = document.getElementById("team-options");
const showTeamButton = document.getElementById("show-team");
const teamPanel = document.getElementById("team-panel");
const teamTitle = document.getElementById("team-title");
const teamSummary = document.getElementById("team-summary");
const teamKpis = document.getElementById("team-kpis");
const teamHistoryBody = document.getElementById("team-history-body");
const rankChart = document.getElementById("rank-chart");
const scoreChart = document.getElementById("score-chart");
const movementGrid = document.getElementById("movement-grid");
const analyticsGrid = document.getElementById("analytics-grid");
const snapshotRankings = document.getElementById("snapshot-rankings");
const historyMeta = document.getElementById("history-meta");
const historyError = document.getElementById("history-error");

const statSnapshots = document.getElementById("stat-snapshots");
const statTeams = document.getElementById("stat-teams");
const statNumberOne = document.getElementById("stat-number-one");
const statConsistent = document.getElementById("stat-consistent");

let state = {
  index: [],
  snapshots: [],
  season: null
};

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function number(value, digits = 3) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : "—";
}

function formatDate(value) {
  if (!value) return "—";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

function periodLabel(snapshot) {
  return snapshot.week ? `Week ${snapshot.week}` : `Snapshot ${snapshot.snapshot_number || "?"}`;
}

function snapshotLabel(snapshot) {
  return `${periodLabel(snapshot)} · ${formatDate(snapshot.captured_at)}`;
}

async function json(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`${url} returned HTTP ${response.status}`);
  return response.json();
}

function teamNames(snapshots) {
  return [...new Set(
    snapshots.flatMap(snapshot => (snapshot.data.top25 || []).map(team => team.team))
  )].sort((a, b) => a.localeCompare(b));
}

function recordsForTeam(teamName) {
  return state.snapshots.map(snapshot => ({
    snapshot,
    team: (snapshot.data.top25 || []).find(team => team.team.toLowerCase() === teamName.toLowerCase()) || null
  }));
}

function statsForSeason() {
  const snapshots = state.snapshots;
  const teams = teamNames(snapshots);
  const counts = new Map();
  const numberOnes = new Map();

  for (const snapshot of snapshots) {
    for (const team of snapshot.data.top25 || []) {
      const row = counts.get(team.team) || { appearances: 0, ranks: [] };
      row.appearances += 1;
      row.ranks.push(Number(team.rank));
      counts.set(team.team, row);
    }

    const first = (snapshot.data.top25 || [])[0];
    if (first) numberOnes.set(first.team, (numberOnes.get(first.team) || 0) + 1);
  }

  const mostNumberOne = [...numberOnes.entries()]
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))[0];

  const consistent = [...counts.entries()]
    .filter(([, value]) => value.ranks.length >= 2)
    .map(([team, value]) => {
      const avg = value.ranks.reduce((a, b) => a + b, 0) / value.ranks.length;
      const variance = value.ranks.reduce((sum, rank) => sum + Math.pow(rank - avg, 2), 0) / value.ranks.length;
      return { team, sd: Math.sqrt(variance), appearances: value.appearances };
    })
    .sort((a, b) => a.sd - b.sd || b.appearances - a.appearances)[0];

  return { teams, counts, mostNumberOne, consistent };
}

function renderMeta() {
  const latest = state.snapshots[state.snapshots.length - 1];
  historyMeta.innerHTML = latest ? `
    <span class="meta-chip">Season <strong>${esc(state.season)}</strong></span>
    <span class="meta-chip">Latest <strong>${esc(periodLabel(latest))} · ${esc(formatDate(latest.captured_at))}</strong></span>
    <span class="meta-chip">Archive <strong>${esc(state.snapshots.length)} snapshots</strong></span>
  ` : "";
}

function renderSummary() {
  const { teams, mostNumberOne, consistent } = statsForSeason();

  statSnapshots.textContent = String(state.snapshots.length);
  statTeams.textContent = String(teams.length);
  statNumberOne.textContent = mostNumberOne ? `${mostNumberOne[0]} (${mostNumberOne[1]})` : "—";
  statConsistent.textContent = consistent ? consistent.team : "—";
}

function movementBetween(previous, latest) {
  const prevMap = new Map((previous?.data.top25 || []).map(team => [team.team, team]));
  const latestMap = new Map((latest?.data.top25 || []).map(team => [team.team, team]));
  const movement = [];

  for (const [team, current] of latestMap) {
    const before = prevMap.get(team);
    movement.push({
      team,
      currentRank: current.rank,
      previousRank: before?.rank ?? null,
      delta: before ? Number(before.rank) - Number(current.rank) : null,
      entered: !before
    });
  }

  const dropped = [...prevMap.entries()]
    .filter(([team]) => !latestMap.has(team))
    .map(([team, before]) => ({ team, previousRank: before.rank }));

  return { movement, dropped };
}

function movementList(items, mode) {
  if (!items.length) return '<div class="empty-history">No movement data yet.</div>';

  return `<ul class="movement-list">${items.map(item => {
    let value = "";
    let cls = "trend-flat";

    if (mode === "up") {
      value = `+${item.delta} to #${item.currentRank}`;
      cls = "trend-up";
    } else if (mode === "down") {
      value = `${item.delta} to #${item.currentRank}`;
      cls = "trend-down";
    } else if (mode === "new") {
      value = `New at #${item.currentRank}`;
      cls = "trend-up";
    } else {
      value = `From #${item.previousRank}`;
      cls = "trend-down";
    }

    return `<li><span>${esc(item.team)}</span><span class="movement-value ${cls}">${esc(value)}</span></li>`;
  }).join("")}</ul>`;
}

function renderMovement() {
  const latest = state.snapshots[state.snapshots.length - 1];
  const previous = state.snapshots[state.snapshots.length - 2];

  if (!latest || !previous) {
    movementGrid.innerHTML = '<div class="empty-history">A second archived snapshot is needed to calculate movement.</div>';
    return;
  }

  const { movement, dropped } = movementBetween(previous, latest);
  const risers = movement.filter(x => x.delta > 0).sort((a, b) => b.delta - a.delta).slice(0, 5);
  const fallers = movement.filter(x => x.delta < 0).sort((a, b) => a.delta - b.delta).slice(0, 5);
  const newcomers = movement.filter(x => x.entered).slice(0, 5);

  movementGrid.innerHTML = `
    <article class="movement-card"><h3>Biggest risers</h3>${movementList(risers, "up")}</article>
    <article class="movement-card"><h3>Biggest fallers</h3>${movementList(fallers, "down")}</article>
    <article class="movement-card"><h3>Entered Top 25</h3>${movementList(newcomers, "new")}</article>
    <article class="movement-card"><h3>Dropped out</h3>${movementList(dropped.slice(0, 5), "drop")}</article>
  `;
}

function renderAnalytics() {
  const { counts } = statsForSeason();

  const rows = [...counts.entries()].map(([team, value]) => ({
    team,
    appearances: value.appearances,
    best: Math.min(...value.ranks),
    average: value.ranks.reduce((a, b) => a + b, 0) / value.ranks.length,
    range: Math.max(...value.ranks) - Math.min(...value.ranks)
  }));

  const mostRanked = [...rows].sort((a, b) => b.appearances - a.appearances || a.average - b.average).slice(0, 5);
  const bestAverage = [...rows].filter(x => x.appearances >= Math.min(2, state.snapshots.length))
    .sort((a, b) => a.average - b.average).slice(0, 5);
  const biggestRange = [...rows].filter(x => x.appearances >= 2)
    .sort((a, b) => b.range - a.range).slice(0, 5);
  const latest = state.snapshots[state.snapshots.length - 1];

  const strongestResume = [...(latest?.data.top25 || [])]
    .sort((a, b) => Number(b.sos) - Number(a.sos) || Number(a.rank) - Number(b.rank))
    .slice(0, 5);

  const list = (items, valueFn) => `<ul class="analytics-list">${items.map(item =>
    `<li><span>${esc(item.team)}</span><span class="analytics-value">${esc(valueFn(item))}</span></li>`
  ).join("")}</ul>`;

  analyticsGrid.innerHTML = `
    <article class="analytics-card">
      <h3>Most Top 25 appearances</h3>
      ${list(mostRanked, item => `${item.appearances} / ${state.snapshots.length}`)}
    </article>
    <article class="analytics-card">
      <h3>Best average rank</h3>
      ${list(bestAverage, item => `#${item.average.toFixed(1)}`)}
    </article>
    <article class="analytics-card">
      <h3>Largest rank range</h3>
      ${list(biggestRange, item => `${item.range} places`)}
    </article>
    <article class="analytics-card">
      <h3>Highest latest SOS1</h3>
      ${list(strongestResume, item => number(item.sos, 3))}
    </article>
  `;
}

function renderSnapshot(snapshot) {
  snapshotRankings.innerHTML = (snapshot?.data.top25 || []).map(team => `
    <li class="rank-item snapshot-ranking-row">
      <div class="rank-head">
        <div class="rank-num">${esc(team.rank)}</div>
        <div>
          <div class="snapshot-team">${esc(team.team)}</div>
          <div class="snapshot-sub">${esc(team.wins)}-${esc(team.losses)} · SOS ${esc(number(team.sos, 3))}</div>
        </div>
        <div class="score">
          ${esc(number(team.score, 3))}
          <span class="score-label">Power Rating</span>
        </div>
      </div>
    </li>
  `).join("");
}

function svgChart(records, field, rankMode = false) {
  const width = Math.max(620, records.length * 95);
  const height = 250;
  const pad = { left: 42, right: 18, top: 18, bottom: 42 };

  const values = records.map(record => {
    if (!record.team) return rankMode ? 26 : null;
    const value = Number(record.team[field]);
    return Number.isFinite(value) ? value : null;
  });

  const numeric = values.filter(value => value !== null);
  if (!numeric.length) return '<div class="empty-history">No chart data available.</div>';

  let min = rankMode ? 1 : Math.min(...numeric);
  let max = rankMode ? 26 : Math.max(...numeric);
  if (min === max) { min -= .05; max += .05; }

  const x = i => pad.left + (records.length === 1 ? 0 : i * ((width - pad.left - pad.right) / (records.length - 1)));
  const y = value => {
    if (rankMode) return pad.top + ((value - min) / (max - min)) * (height - pad.top - pad.bottom);
    return pad.top + ((max - value) / (max - min)) * (height - pad.top - pad.bottom);
  };

  const points = values.map((value, i) => value === null ? null : [x(i), y(value), value, !records[i].team]).filter(Boolean);
  const path = points.map((p, i) => `${i ? "L" : "M"} ${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(" ");

  const ticks = rankMode ? [1, 5, 10, 15, 20, 25, 26] : [0, .25, .5, .75, 1].map(t => min + (max - min) * t);

  return `
    <svg viewBox="0 0 ${width} ${height}" role="img">
      ${ticks.map(value => {
        const yy = y(value);
        const label = rankMode ? (value === 26 ? "NR" : `#${value}`) : value.toFixed(3);
        return `<line class="chart-grid" x1="${pad.left}" x2="${width - pad.right}" y1="${yy}" y2="${yy}"></line>
          <text class="chart-axis-label" x="4" y="${yy + 3}">${esc(label)}</text>`;
      }).join("")}
      <path class="chart-line" d="${path}"></path>
      ${points.map(p => `<circle class="${p[3] ? "chart-nr-point" : "chart-point"}" cx="${p[0]}" cy="${p[1]}" r="4"></circle>`).join("")}
      ${records.map((record, i) => `<text class="chart-axis-label" text-anchor="middle" x="${x(i)}" y="${height - 12}">${esc(record.snapshot.week ? "W" + record.snapshot.week : "S" + (record.snapshot.snapshot_number || i + 1))}</text>`).join("")}
    </svg>
  `;
}

function renderTeam(teamName) {
  const canonical = teamNames(state.snapshots).find(name => name.toLowerCase() === teamName.trim().toLowerCase());

  if (!canonical) {
    teamPanel.hidden = false;
    teamTitle.textContent = teamName || "Team not found";
    teamSummary.textContent = "No archived Top 25 appearances for this team in the selected season.";
    teamKpis.innerHTML = "";
    rankChart.innerHTML = '<div class="empty-history">No rank history available.</div>';
    scoreChart.innerHTML = '<div class="empty-history">No score history available.</div>';
    teamHistoryBody.innerHTML = "";
    return;
  }

  teamSearch.value = canonical;
  const records = recordsForTeam(canonical);
  const ranked = records.filter(record => record.team);
  const ranks = ranked.map(record => Number(record.team.rank));
  const scores = ranked.map(record => Number(record.team.score));

  teamPanel.hidden = false;
  teamTitle.textContent = canonical;
  teamSummary.textContent = `${ranked.length} Top 25 appearances across ${state.snapshots.length} snapshots`;

  const best = ranks.length ? Math.min(...ranks) : null;
  const latest = [...ranked].pop();
  const avg = ranks.length ? ranks.reduce((a, b) => a + b, 0) / ranks.length : null;
  const highScore = scores.length ? Math.max(...scores) : null;

  teamKpis.innerHTML = `
    <div class="team-kpi"><span>Best rank</span><strong>${best ? "#" + best : "NR"}</strong></div>
    <div class="team-kpi"><span>Latest rank</span><strong>${latest ? "#" + latest.team.rank : "NR"}</strong></div>
    <div class="team-kpi"><span>Average rank</span><strong>${avg ? "#" + avg.toFixed(1) : "—"}</strong></div>
    <div class="team-kpi"><span>Top 25 rate</span><strong>${state.snapshots.length ? Math.round(ranked.length / state.snapshots.length * 100) + "%" : "—"}</strong></div>
    <div class="team-kpi"><span>High score</span><strong>${highScore === null ? "—" : highScore.toFixed(3)}</strong></div>
  `;

  rankChart.innerHTML = svgChart(records, "rank", true);
  scoreChart.innerHTML = svgChart(ranked, "score", false);

  teamHistoryBody.innerHTML = records.map(record => {
    const team = record.team;
    return `<tr>
      <td><strong>${esc(periodLabel(record.snapshot))}</strong></td>
      <td>${esc(formatDate(record.snapshot.captured_at))}</td>
      <td>${team ? "#" + esc(team.rank) : "NR"}</td>
      <td>${team ? esc(number(team.score, 3)) : "—"}</td>
      <td>${team ? esc(team.wins) + "-" + esc(team.losses) : "—"}</td>
      <td>${team ? esc(number(team.sos, 3)) : "—"}</td>
      <td>${team ? esc(number(team.sos2, 3)) : "—"}</td>
      <td>${team ? esc(number(team.avg_margin, 1)) : "—"}</td>
      <td>${team ? esc(team.qual_wins ?? 0) : "—"}</td>
    </tr>`;
  }).join("");
}

async function loadSeason(season) {
  state.season = Number(season);
  const entries = state.index.filter(item => Number(item.season) === state.season);

  state.snapshots = await Promise.all(entries.map(async entry => ({
    ...entry,
    data: await json("./data/" + entry.file + "?v=" + encodeURIComponent(entry.id))
  })));

  state.snapshots.sort((a, b) => new Date(a.captured_at) - new Date(b.captured_at));

  const names = teamNames(state.snapshots);
  teamOptions.innerHTML = names.map(name => `<option value="${esc(name)}"></option>`).join("");

  snapshotSelect.innerHTML = state.snapshots.map((snapshot, i) =>
    `<option value="${i}">${esc(snapshotLabel(snapshot))}</option>`
  ).join("");
  if (state.snapshots.length) snapshotSelect.value = String(state.snapshots.length - 1);

  renderMeta();
  renderSummary();
  renderMovement();
  renderAnalytics();
  renderSnapshot(state.snapshots[state.snapshots.length - 1]);

  if (teamSearch.value) renderTeam(teamSearch.value);
}

async function main() {
  try {
    let indexPayload;
    try {
      indexPayload = await json(INDEX_URL);
    } catch {
      const current = await json(CURRENT_URL);
      const captured = current.last_build_utc || new Date().toISOString();
      indexPayload = {
        snapshots: [{
          id: "current",
          season: current.season,
          captured_at: captured,
          date: captured.slice(0, 10),
          file: "rankings.json",
          team_count: (current.top25 || []).length,
          top_team: current.top25?.[0]?.team || null,
          snapshot_number: 1
        }]
      };
    }

    state.index = (indexPayload.snapshots || []).slice().sort((a, b) =>
      new Date(a.captured_at) - new Date(b.captured_at)
    );

    const seasons = [...new Set(state.index.map(item => Number(item.season)))].sort((a, b) => b - a);
    if (!seasons.length) throw new Error("No ranking snapshots are available.");

    seasonSelect.innerHTML = seasons.map(season => `<option value="${season}">${season}</option>`).join("");
    await loadSeason(seasons[0]);

    const florida = teamNames(state.snapshots).find(name => name === "Florida");
    if (florida) renderTeam(florida);
  } catch (error) {
    console.error(error);
    historyError.hidden = false;
    historyError.innerHTML = `<strong>Unable to load ranking history.</strong><br>${esc(error.message || error)}`;
  }
}

seasonSelect.addEventListener("change", () => loadSeason(seasonSelect.value));
snapshotSelect.addEventListener("change", () => renderSnapshot(state.snapshots[Number(snapshotSelect.value)]));
showTeamButton.addEventListener("click", () => renderTeam(teamSearch.value));
teamSearch.addEventListener("keydown", event => {
  if (event.key === "Enter") renderTeam(teamSearch.value);
});

main();
