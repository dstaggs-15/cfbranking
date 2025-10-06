// Minimal, framework-free frontend.
// - Loads data/rankings.json
// - Renders dark list with team-colored pills
// - Expand row to show stats
// - Computes z-scores for advanced fields from the loaded set

const TEAM_COLORS = {
  // Add/adjust as you like; sensible defaults + fallbacks are applied.
  "Alabama": "#9e0b0f",
  "Georgia": "#cc0000",
  "Ohio State": "#bb0000",
  "Michigan": "#00274c",
  "Texas": "#bf5700",
  "Oregon": "#004f39",
  "Florida State": "#782f40",
  "Notre Dame": "#0c2340",
  "Clemson": "#f56600",
  "LSU": "#461d7c",
  "Ole Miss": "#14213d",
  "Miami": "#005030",
  "Oklahoma": "#841617",
  "Tennessee": "#ff8200",
  "Auburn": "#0a1f44",
  "Penn State": "#041e42",
  "USC": "#990000",
  "Utah": "#cc0000",
  "Texas A&M": "#500000",
  "Washington": "#4b2e83",
  "Florida": "#0021a5",
  "North Carolina": "#7BAFD4",
  "Duke": "#012169",
  "Kansas State": "#512888",
  "Missouri": "#f1b82d"
};

function fmt(n, digits = 2) {
  return (typeof n === "number" && isFinite(n)) ? n.toFixed(digits) : "—";
}

function mean(arr) {
  if (!arr.length) return 0;
  return arr.reduce((a,b)=>a+b,0)/arr.length;
}
function std(arr) {
  if (arr.length < 2) return 1;
  const m = mean(arr);
  const v = mean(arr.map(x => (x-m)*(x-m)));
  return v === 0 ? 1 : Math.sqrt(v);
}
function zscore(val, arr) {
  const s = std(arr), m = mean(arr);
  return s === 0 ? 0 : (val - m) / s;
}

async function load() {
  const res = await fetch("data/rankings.json", { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to load rankings.json");
  return await res.json();
}

function colorForTeam(team) {
  return TEAM_COLORS[team] || "#122033";
}

function renderMeta(meta, season, lastBuild) {
  meta.innerHTML = `
    <div>Season: <strong>${season}</strong></div>
    <div>Last build (UTC): <strong>${lastBuild}</strong></div>
    <div>Click a team to expand stats.</div>
  `;
}

function buildZFields(teams) {
  // compute z across the currently loaded set (Top 25)
  const offPPA = teams.map(t => t.off_ppa ?? 0);
  const defPPA = teams.map(t => t.def_ppa ?? 0);
  const offSR  = teams.map(t => t.off_sr  ?? 0);
  const defSR  = teams.map(t => t.def_sr  ?? 0);

  return teams.map(t => ({
    team: t.team,
    z: {
      off_ppa: zscore(t.off_ppa ?? 0, offPPA),
      def_ppa: zscore(t.def_ppa ?? 0, defPPA),
      off_sr:  zscore(t.off_sr  ?? 0, offSR),
      def_sr:  zscore(t.def_sr  ?? 0, defSR)
    }
  })).reduce((acc, row) => (acc[row.team] = row.z, acc), {});
}

function rowTemplate(t, zmap) {
  const color = colorForTeam(t.team);
  const z = zmap[t.team] || { off_ppa: 0, def_ppa: 0, off_sr: 0, def_sr: 0 };

  return `
    <li class="rank-item" data-team="${t.team}">
      <div class="rank-head">
        <div class="rank-num">${t.rank}</div>
        <button class="team-pill" style="background: ${color}22; border-color: ${color}66;">
          <span class="team-name" style="color:${color}">${t.team}</span>
          <span class="record">(${t.wins}-${t.losses})</span>
        </button>
        <div class="score">${fmt(t.score, 3)}</div>
      </div>
      <div class="details">
        <div class="grid">
          <div class="card">
            <h4>Résumé</h4>
            <div class="kv"><div class="k">Strength of Schedule</div><div class="v">${fmt(t.sos,3)}</div></div>
            <div class="kv"><div class="k">Average Scoring Margin</div><div class="v">${fmt(t.avg_margin,1)}</div></div>
            <div class="kv"><div class="k">Quality Wins (FBS)</div><div class="v">${t.qual_wins}</div></div>
            <div class="kv"><div class="k">Bad Losses (FBS)</div><div class="v">${t.bad_losses}</div></div>
          </div>
          <div class="card">
            <h4>Record</h4>
            <div class="kv"><div class="k">Overall</div><div class="v">${t.wins}-${t.losses}</div></div>
            <div class="kv"><div class="k">Points For</div><div class="v">${t.points_for}</div></div>
            <div class="kv"><div class="k">Points Against</div><div class="v">${t.points_against}</div></div>
            <div class="kv"><div class="k">FBS Wins / Losses</div><div class="v">${t.fbs_wins}-${t.fbs_losses}</div></div>
          </div>
          <div class="card">
            <h4>Advanced (z-scores within this list)</h4>
            <div class="kv"><div class="k">Offense: Predicted Points Added</div><div class="v">${fmt(z.off_ppa,2)}</div></div>
            <div class="kv"><div class="k">Defense: Predicted Points Added</div><div class="v">${fmt(z.def_ppa,2)}</div></div>
            <div class="kv"><div class="k">Offense: Success Rate</div><div class="v">${fmt(z.off_sr,2)}</div></div>
            <div class="kv"><div class="k">Defense: Success Rate</div><div class="v">${fmt(z.def_sr,2)}</div></div>
          </div>
        </div>
      </div>
    </li>
  `;
}

function attachInteractivity(root) {
  root.querySelectorAll(".team-pill").forEach(btn => {
    const li = btn.closest(".rank-item");
    const details = li.querySelector(".details");
    btn.addEventListener("click", () => {
      details.style.display = (details.style.display === "block") ? "none" : "block";
    });
  });
}

async function main() {
  try {
    const data = await load();
    const metaEl = document.getElementById("meta");
    const listEl = document.getElementById("rank-list");
    renderMeta(metaEl, data.season, data.last_build_utc);

    const teams = (data.top25 || []);
    const zmap = buildZFields(teams);

    listEl.innerHTML = teams.map(t => rowTemplate(t, zmap)).join("");
    attachInteractivity(listEl);
  } catch (e) {
    console.error(e);
    const listEl = document.getElementById("rank-list");
    listEl.innerHTML = `<li class="rank-item"><div class="rank-head">Failed to load rankings.</div></li>`;
  }

  // Explainer modal
  const dlg = document.getElementById("explainer");
  document.getElementById("open-explainer").addEventListener("click", () => dlg.showModal());
  document.getElementById("close-explainer").addEventListener("click", () => dlg.close());
}

main();
