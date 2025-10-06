async function loadJSON(url) {
  const res = await fetch(url, {cache: "no-store"});
  if (!res.ok) throw new Error(`Failed to fetch ${url}`);
  return res.json();
}

function imgWithFallback(src, alt) {
  const img = document.createElement("img");
  img.src = src;
  img.alt = alt;
  img.onerror = () => { img.style.display = "none"; };
  return img;
}

function renderTable(data) {
  const tbody = document.querySelector("#rankings tbody");
  tbody.innerHTML = "";
  data.top25.forEach(row => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.rank}</td>
      <td class="team"></td>
      <td>${row.record}</td>
      <td class="score">${row.final_score.toFixed(2)}</td>
      <td class="score">${row.performance.toFixed(2)}</td>
      <td class="score">${row.sos.toFixed(2)}</td>
      <td class="score">${row.quality_wins.toFixed(2)}</td>
    `;
    const teamCell = tr.querySelector(".team");
    const img = imgWithFallback(row.logo, row.team);
    teamCell.appendChild(img);
    const name = document.createElement("span");
    name.textContent = row.team;
    teamCell.appendChild(name);
    tbody.appendChild(tr);
  });
  const meta = document.getElementById("meta");
  meta.textContent = `Season ${data.season} • Last build: ${new Date(data.last_build_utc).toLocaleString()}`;
}

(async () => {
  try {
    const data = await loadJSON("data/rankings.json");
    renderTable(data);
  } catch (e) {
    console.error(e);
    document.getElementById("meta").textContent = "Error loading rankings.";
  }
})();
