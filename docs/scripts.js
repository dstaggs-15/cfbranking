/*
 * CFB COMPUTER RANKINGS FRONT END
 *
 * Backend source:
 *   docs/data/rankings.json
 *
 * Logo source:
 *   sai'emgilani's college football logo gist
 *
 * This frontend intentionally follows the CURRENT backend schema.
 */


// =========================================================
// CONFIG
// =========================================================

const DATA_URL = "./data/rankings.json?v=" + Date.now();

const LOGO_DATA_URL =
  "https://gist.githubusercontent.com/saiemgilani/c6596f0e1c8b148daabc2b7f1e6f6add/raw/logos";


// =========================================================
// DOM
// =========================================================

const metaEl = document.getElementById("meta");
const listEl = document.getElementById("rank-list");
const errorEl = document.getElementById("error");

const explainer = document.getElementById("explainer");
const openExplainer = document.getElementById("open-explainer");
const closeExplainer = document.getElementById("close-explainer");


// =========================================================
// TEAM COLORS
// =========================================================

const TEAM_COLORS = {
  "Alabama": "#9E1B32",
  "Arizona": "#CC0033",
  "Arizona State": "#8C1D40",
  "Arkansas": "#9D2235",
  "Arkansas State": "#CC092F",
  "Army": "#CE9C00",
  "Auburn": "#03244D",

  "Baylor": "#004834",
  "Boise State": "#09347A",
  "Boston College": "#8B1A1A",
  "BYU": "#002E5D",

  "California": "#003262",
  "Cincinnati": "#E00122",
  "Clemson": "#F56600",
  "Coastal Carolina": "#006F71",
  "Colorado": "#CFB87C",
  "Connecticut": "#000E2F",

  "Duke": "#003087",

  "East Carolina": "#592A8A",

  "Florida": "#0021A5",
  "Florida State": "#782F40",
  "Fresno State": "#DB0032",

  "Georgia": "#BA0C2F",
  "Georgia Southern": "#041E42",
  "Georgia State": "#0039A6",

  "Houston": "#C8102E",

  "Illinois": "#13294B",
  "Indiana": "#990000",
  "Iowa": "#000000",
  "Iowa State": "#C8102E",

  "Kansas": "#0051BA",
  "Kansas State": "#512888",
  "Kent State": "#002664",
  "Kentucky": "#0033A0",

  "Liberty": "#AA0000",
  "Louisiana": "#CE181E",
  "Louisiana Tech": "#002D62",
  "Louisiana Monroe": "#004B91",
  "Louisville": "#AD0000",

  "Maryland": "#E03A3E",
  "Memphis": "#003087",
  "Miami": "#F47321",
  "Miami (FL)": "#F47321",
  "Miami (Ohio)": "#B61E2E",
  "Michigan": "#00274C",
  "Michigan State": "#18453B",
  "Minnesota": "#7A0019",
  "Mississippi State": "#5D1725",
  "Missouri": "#F1B82D",

  "Nebraska": "#E41C38",
  "Nevada": "#003366",
  "North Carolina": "#7BAFD4",
  "North Carolina State": "#CC0000",
  "NC State": "#CC0000",
  "North Texas": "#00853F",
  "Northwestern": "#4E2A84",
  "Notre Dame": "#0C2340",

  "Ohio": "#00694E",
  "Ohio State": "#BB0000",
  "Oklahoma": "#841617",
  "Oklahoma State": "#FF7300",
  "Ole Miss": "#CE1126",
  "Oregon": "#154733",
  "Oregon State": "#DC4405",

  "Penn State": "#041E42",
  "Pittsburgh": "#003594",
  "Purdue": "#CEB888",

  "Rice": "#00205B",
  "Rutgers": "#CC0033",

  "SMU": "#CC0000",
  "Sam Houston": "#F47B20",
  "San Diego State": "#A6192E",
  "South Alabama": "#003E7E",
  "South Carolina": "#73000A",
  "Southern Miss": "#FFAA00",
  "Stanford": "#8C1515",
  "Syracuse": "#F76900",

  "TCU": "#4D1979",
  "Temple": "#9D2235",
  "Tennessee": "#FF8200",
  "Texas": "#BF5700",
  "Texas A&M": "#500000",
  "Texas State": "#501214",
  "Texas Tech": "#CC0000",
  "Toledo": "#003E7E",
  "Troy": "#B01E24",
  "Tulane": "#006747",
  "Tulsa": "#003D79",

  "UCF": "#BA9B37",
  "UCLA": "#2D68C4",
  "UNLV": "#CF0A2C",
  "USC": "#990000",
  "UTEP": "#FF8200",
  "UTSA": "#0C2340",
  "Utah": "#CC0000",

  "Vanderbilt": "#000000",
  "Virginia": "#232D4B",
  "Virginia Tech": "#630031",

  "Wake Forest": "#9E7E38",
  "Washington": "#4B2E83",
  "Washington State": "#981E32",
  "West Virginia": "#002855",
  "Western Michigan": "#6C4023",
  "Wisconsin": "#C5050C"
};


// =========================================================
// HELPERS
// =========================================================

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}


function number(value, digits = 3) {
  if (
    value === null ||
    value === undefined ||
    value === "" ||
    Number.isNaN(Number(value))
  ) {
    return "—";
  }

  return Number(value).toFixed(digits);
}


function integer(value) {
  if (
    value === null ||
    value === undefined ||
    Number.isNaN(Number(value))
  ) {
    return "—";
  }

  return Math.round(Number(value)).toLocaleString();
}


function percent(value, digits = 1) {
  if (
    value === null ||
    value === undefined ||
    Number.isNaN(Number(value))
  ) {
    return "—";
  }

  return (Number(value) * 100).toFixed(digits) + "%";
}


function winPercentage(team) {
  const games = Number(team.games ?? 0);

  if (!games) {
    return null;
  }

  return Number(team.wins ?? 0) / games;
}


function colorForTeam(team) {
  return TEAM_COLORS[team] || "#52525b";
}


function hexToRgba(hex, alpha) {
  let clean = String(hex).replace("#", "");

  if (clean.length === 3) {
    clean = clean
      .split("")
      .map(char => char + char)
      .join("");
  }

  const value = Number.parseInt(clean, 16);

  if (Number.isNaN(value)) {
    return `rgba(82,82,91,${alpha})`;
  }

  const r = (value >> 16) & 255;
  const g = (value >> 8) & 255;
  const b = value & 255;

  return `rgba(${r},${g},${b},${alpha})`;
}


function initials(team) {
  return String(team || "??")
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map(part => part[0])
    .join("")
    .toUpperCase();
}


// =========================================================
// LOGOS
// =========================================================

let logoMap = new Map();


function normalizeTeamName(name) {
  return String(name || "")
    .toLowerCase()
    .replace(/\s+/g, " ")
    .trim();
}


function addLogoAlias(map, name, logo) {
  if (!name || !logo) {
    return;
  }

  map.set(normalizeTeamName(name), logo);
}


function parseLogoData(text) {

  const map = new Map();

  const lines = String(text)
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(Boolean);

  /*
   * The gist is a pipe-delimited markdown-style table.
   *
   * We also support normal CSV just in case the gist changes format.
   */

  for (const line of lines) {

    if (
      line.startsWith("|") &&
      line.includes("school") &&
      line.includes("logo")
    ) {
      continue;
    }

    let columns;

    if (line.startsWith("|")) {

      columns = line
        .split("|")
        .slice(1, -1)
        .map(value => value.trim());

    } else {

      columns = line
        .split(",")
        .map(value => value.trim());
    }

    if (columns.length < 12) {
      continue;
    }

    const school = columns[1];
    const altName1 = columns[4];
    const altName2 = columns[5];
    const altName3 = columns[6];

    const logo = columns[11];

    if (!school || !logo) {
      continue;
    }

    const secureLogo = logo.replace(
      /^http:\/\//i,
      "https://"
    );

    addLogoAlias(map, school, secureLogo);
    addLogoAlias(map, altName1, secureLogo);
    addLogoAlias(map, altName2, secureLogo);
    addLogoAlias(map, altName3, secureLogo);
  }

  return map;
}


async function loadLogos() {

  try {

    const response = await fetch(LOGO_DATA_URL, {
      cache: "force-cache"
    });

    if (!response.ok) {
      throw new Error("Logo list request failed");
    }

    const text = await response.text();

    logoMap = parseLogoData(text);

  } catch (error) {

    console.warn(
      "Could not load college football logos.",
      error
    );

    logoMap = new Map();
  }
}


function logoForTeam(team) {

  const normalized = normalizeTeamName(team);

  /*
   * Common backend names that can differ from the gist.
   */

  const aliases = {
    "usc": "usc",
    "miami": "miami",
    "miami (fl)": "miami",
    "miami fl": "miami",
    "nc state": "north carolina state",
    "virginia tech": "virginia tech",
    "louisiana tech": "louisiana tech",
    "louisiana": "louisiana",
    "ulm": "louisiana monroe",
    "ul monroe": "louisiana monroe"
  };

  const lookupName =
    aliases[normalized] || normalized;

  return logoMap.get(lookupName) || null;
}


function teamLogo(team) {

  const logo = logoForTeam(team);

  if (logo) {

    return `
      <img
        class="team-logo"
        src="${esc(logo)}"
        alt=""
        loading="lazy"
        onerror="this.style.display='none'; this.nextElementSibling.style.display='grid';"
      >
      <span
        class="team-logo-fallback"
        aria-hidden="true"
        style="display:none"
      >${esc(initials(team))}</span>
    `;

  }

  return `
    <span
      class="team-logo-fallback"
      aria-hidden="true"
    >${esc(initials(team))}</span>
  `;
}


// =========================================================
// TEAM COLORS
// =========================================================

function teamStyle(team) {

  const color = colorForTeam(team);

  return [
    `background:${hexToRgba(color, 0.14)}`,
    `border-color:${hexToRgba(color, 0.48)}`
  ].join(";");
}


// =========================================================
// META
// =========================================================

function renderMeta(data) {

  const weights = data?.notes?.weights || {};

  const sosWeight =
    Number(weights.sos1 ?? 0) +
    Number(weights.sos2 ?? 0);

  metaEl.innerHTML = `
    <div class="meta-chip">
      <span>Season</span>
      <strong>${esc(data.season)}</strong>
    </div>

    <div class="meta-chip">
      <span>Last build</span>
      <strong>${esc(formatBuildDate(data.last_build_utc))}</strong>
    </div>

    <div class="meta-chip">
      <span>SOS weighting</span>
      <strong>${number(sosWeight, 2)}</strong>
    </div>

    <div class="meta-chip">
      <span>Teams ranked</span>
      <strong>${(data.top25 || []).length}</strong>
    </div>
  `;
}


function formatBuildDate(value) {

  if (!value) {
    return "—";
  }

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return String(value);
  }

  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit"
  });
}


// =========================================================
// ADVANCED METRIC NORMALIZATION
// =========================================================

function mean(values) {

  const valid = values
    .map(Number)
    .filter(value => Number.isFinite(value));

  if (!valid.length) {
    return 0;
  }

  return valid.reduce(
    (sum, value) => sum + value,
    0
  ) / valid.length;
}


function std(values) {

  const valid = values
    .map(Number)
    .filter(value => Number.isFinite(value));

  if (valid.length < 2) {
    return 1;
  }

  const average = mean(valid);

  const variance =
    valid.reduce(
      (sum, value) =>
        sum + Math.pow(value - average, 2),
      0
    ) / valid.length;

  return Math.sqrt(variance) || 1;
}


function zScore(value, values) {

  if (
    value === null ||
    value === undefined ||
    !Number.isFinite(Number(value))
  ) {
    return 0;
  }

  return (
    Number(value) - mean(values)
  ) / std(values);
}


function buildAdvancedMap(teams) {

  const fields = [
    "off_ppa",
    "def_ppa",
    "off_sr",
    "def_sr"
  ];

  const values = {};

  for (const field of fields) {
    values[field] = teams.map(
      team => Number(team[field] ?? 0)
    );
  }

  const map = new Map();

  for (const team of teams) {

    map.set(team.team, {

      off_ppa:
        zScore(
          team.off_ppa,
          values.off_ppa
        ),

      def_ppa:
        zScore(
          team.def_ppa,
          values.def_ppa
        ),

      off_sr:
        zScore(
          team.off_sr,
          values.off_sr
        ),

      def_sr:
        zScore(
          team.def_sr,
          values.def_sr
        )
    });
  }

  return map;
}


// =========================================================
// ADVANCED METRIC UI
// =========================================================

function metricStatus(z, inverse = false) {

  let score = Number(z) || 0;

  if (inverse) {
    score *= -1;
  }

  if (score >= 1.0) {
    return ["Elite", "badge-good"];
  }

  if (score >= 0.25) {
    return ["Above average", "badge-good"];
  }

  if (score <= -1.0) {
    return ["Poor", "badge-bad"];
  }

  if (score <= -0.25) {
    return ["Below average", "badge-bad"];
  }

  return ["Average", "badge-neutral"];
}


function metricBar(z, inverse = false) {

  let score = Number(z) || 0;

  if (inverse) {
    score *= -1;
  }

  /*
   * Map approximately -2.5 to +2.5
   * into 5% to 95%.
   */

  const width =
    Math.max(
      5,
      Math.min(
        95,
        50 + score * 17
      )
    );

  return width;
}


function advancedMetricHTML(
  name,
  value,
  z,
  inverse = false,
  suffix = ""
) {

  const [label, badgeClass] =
    metricStatus(z, inverse);

  const width =
    metricBar(z, inverse);

  return `
    <div class="metric">

      <div class="metric-top">
        <span class="metric-name">
          ${esc(name)}
        </span>

        <span class="metric-value">
          ${esc(number(value, 2))}${esc(suffix)}
          <span class="badge ${badgeClass}">
            ${esc(label)}
          </span>
        </span>
      </div>

      <div class="metric-bar">
        <div
          class="metric-fill"
          style="width:${width}%"
        ></div>
      </div>

      <div class="metric-caption">
        <span>Below average</span>
        <span>Above average</span>
      </div>

    </div>
  `;
}


// =========================================================
// STAT ROW
// =========================================================

function statRow(label, value) {

  return `
    <div class="stat-row">
      <span class="stat-label">
        ${esc(label)}
      </span>

      <span class="stat-value">
        ${esc(value)}
      </span>
    </div>
  `;
}


// =========================================================
// TEAM BUBBLE
// =========================================================

function teamBubble(team) {

  const color = colorForTeam(team.team);

  return `
    <button
      class="team-pill"
      type="button"
      data-team="${esc(team.team)}"
      style="${teamStyle(team.team)}"
      aria-expanded="false"
    >

      ${teamLogo(team.team)}

      <span class="team-copy">

        <span
          class="team-name"
          style="color:${esc(color)}"
        >
          ${esc(team.team)}
        </span>

        <span class="record">
          ${esc(team.wins)}-${esc(team.losses)}
        </span>

      </span>

    </button>
  `;
}


// =========================================================
// DETAIL DRAWER
// =========================================================

function detailsHTML(team, z) {

  const winPct = winPercentage(team);

  const location =
    Number(team.location_adj ?? 0);

  return `
    <div class="details-inner">

      <!-- RESUME -->

      <section class="stat-card">

        <div class="stat-title">
          Résumé
        </div>

        ${statRow(
          "Strength of Schedule (SOS1)",
          number(team.sos, 3)
        )}

        ${statRow(
          "Opponents' Opponent (SOS2)",
          number(team.sos2, 3)
        )}

        ${statRow(
          "Average Scoring Margin",
          number(team.avg_margin, 1)
        )}

        ${statRow(
          "Quality Wins",
          integer(team.qual_wins)
        )}

        ${statRow(
          "Bad Losses",
          integer(team.bad_losses)
        )}

        ${statRow(
          "Location Adjustment",
          location >= 0
            ? "+" + number(location, 3)
            : number(location, 3)
        )}

        ${statRow(
          "Conference Champion",
          team.conf_champ
            ? "Yes"
            : "No"
        )}

      </section>


      <!-- RECORD -->

      <section class="stat-card">

        <div class="stat-title">
          Record
        </div>

        ${statRow(
          "Overall",
          `${team.wins}-${team.losses}`
        )}

        ${statRow(
          "Win Percentage",
          winPct === null
            ? "—"
            : (winPct * 100).toFixed(1) + "%"
        )}

        ${statRow(
          "FBS Wins",
          integer(team.fbs_wins)
        )}

        ${statRow(
          "FBS Losses",
          integer(team.fbs_losses)
        )}

        ${statRow(
          "Points For",
          integer(team.points_for)
        )}

        ${statRow(
          "Points Against",
          integer(team.points_against)
        )}

      </section>


      <!-- ADVANCED -->

      <section class="stat-card">

        <div class="stat-title">
          Advanced
        </div>

        ${advancedMetricHTML(
          "Offensive PPA",
          team.off_ppa,
          z?.off_ppa
        )}

        ${advancedMetricHTML(
          "Defensive PPA",
          team.def_ppa,
          z?.def_ppa,
          true
        )}

        ${advancedMetricHTML(
          "Offensive Success Rate",
          team.off_sr,
          z?.off_sr
        )}

        ${advancedMetricHTML(
          "Defensive Success Rate",
          team.def_sr,
          z?.def_sr,
          true
        )}

      </section>


      <!-- WHY -->

      <section class="stat-card wide">

        <div class="stat-title">
          Why this team is ranked here
        </div>

        <ul class="why-list">

          ${
            Array.isArray(team.why) &&
            team.why.length

              ? team.why
                  .map(
                    reason =>
                      `<li>${esc(reason)}</li>`
                  )
                  .join("")

              : `<li>
                  The backend did not provide an explanation
                  for this team.
                </li>`
          }

        </ul>

      </section>

    </div>
  `;
}


// =========================================================
// RANKING ROW
// =========================================================

function rowTemplate(team, advancedMap) {

  const z =
    advancedMap.get(team.team) || {
      off_ppa: 0,
      def_ppa: 0,
      off_sr: 0,
      def_sr: 0
    };

  return `
    <li
      class="rank-item"
      data-rank="${esc(team.rank)}"
    >

      <div class="rank-head">

        <div class="rank-num">
          ${esc(team.rank)}
        </div>

        ${teamBubble(team)}

        <div class="score">

          ${number(team.score, 3)}

          <span class="score-label">
            Power Rating
          </span>

        </div>

      </div>

      <div class="details">
        ${detailsHTML(team, z)}
      </div>

    </li>
  `;
}


// =========================================================
// INTERACTIVITY
// =========================================================

function attachRowEvents() {

  document
    .querySelectorAll(".team-pill")
    .forEach(button => {

      button.addEventListener(
        "click",
        () => {

          const item =
            button.closest(".rank-item");

          const details =
            item.querySelector(".details");

          const currentlyOpen =
            details.classList.contains("open");

          /*
           * Close every other team first.
           */

          document
            .querySelectorAll(".details.open")
            .forEach(other => {

              other.classList.remove("open");

              const otherButton =
                other
                  .closest(".rank-item")
                  ?.querySelector(".team-pill");

              if (otherButton) {
                otherButton.setAttribute(
                  "aria-expanded",
                  "false"
                );
              }
            });

          /*
           * Toggle clicked team.
           */

          if (!currentlyOpen) {

            details.classList.add("open");

            button.setAttribute(
              "aria-expanded",
              "true"
            );

            /*
             * Small scroll adjustment so the opened
             * team stays visible on mobile.
             */

            setTimeout(() => {

              const rect =
                item.getBoundingClientRect();

              const headerOffset = 20;

              if (rect.bottom > window.innerHeight) {

                window.scrollBy({
                  top:
                    rect.bottom -
                    window.innerHeight +
                    headerOffset,

                  behavior: "smooth"
                });
              }

            }, 80);

          }

        }
      );

    });
}


// =========================================================
// LOAD RANKINGS
// =========================================================

async function loadRankings() {

  const response =
    await fetch(DATA_URL, {
      cache: "no-store"
    });

  if (!response.ok) {
    throw new Error(
      `rankings.json returned HTTP ${response.status}`
    );
  }

  return response.json();
}


// =========================================================
// RENDER
// =========================================================

function render(data) {

  if (
    !data ||
    !Array.isArray(data.top25)
  ) {
    throw new Error(
      "The backend JSON does not contain a top25 array."
    );
  }

  renderMeta(data);

  const teams = data.top25;

  const advancedMap =
    buildAdvancedMap(teams);

  listEl.innerHTML =
    teams
      .map(
        team =>
          rowTemplate(
            team,
            advancedMap
          )
      )
      .join("");

  attachRowEvents();
}


// =========================================================
// ERROR
// =========================================================

function showError(error) {

  console.error(error);

  errorEl.hidden = false;

  errorEl.innerHTML = `
    <strong>Unable to load the rankings.</strong>
    <br>
    ${esc(error.message || error)}
  `;

  listEl.innerHTML = "";
}


// =========================================================
// EXPLAINER MODAL
// =========================================================

openExplainer.addEventListener(
  "click",
  () => {
    explainer.showModal();
  }
);


closeExplainer.addEventListener(
  "click",
  () => {
    explainer.close();
  }
);


explainer.addEventListener(
  "click",
  event => {

    if (event.target === explainer) {
      explainer.close();
    }

  }
);


document.addEventListener(
  "keydown",
  event => {

    if (
      event.key === "Escape" &&
      explainer.open
    ) {
      explainer.close();
    }

  }
);


// =========================================================
// START
// =========================================================

async function main() {

  listEl.innerHTML = `
    <li class="loading">
      Loading computer rankings…
    </li>
  `;

  try {

    /*
     * Load rankings and logos independently.
     * If logos fail, the rankings still work.
     */

    const [data] =
      await Promise.all([
        loadRankings(),
        loadLogos()
      ]);

    render(data);

  } catch (error) {

    showError(error);

  }

}


main();
