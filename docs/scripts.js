async function loadRankings() {
  const res = await fetch("data/rankings.json");
  const data = await res.json();

  const container = document.getElementById("rankings");
  const updated = document.getElementById("updated");

  updated.textContent = `Last updated: ${new Date(data.last_build_utc).toLocaleString()} • Season ${data.season}`;

  if (!data.top25 || data.top25.length === 0) {
    container.innerHTML = "<p>No rankings available yet. Waiting for new games to be completed.</p>";
    return;
  }

  container.innerHTML = data.top25
    .map(
      (t, i) => `
      <div class="card" data-team="${t.team}">
        <div class="rank">#${i + 1}</div>
        <div class="team">${t.team}</div>
        <div class="score">Score: ${t.score.toFixed(3)}</div>
        <div class="details hidden">
          <div>Record: ${t.wins}-${t.losses}</div>
          <div>Points For: ${t.points_for}</div>
          <div>Points Against: ${t.points_against}</div>
          <div>Average Margin: ${t.avg_margin}</div>
          <div>Strength of Schedule: ${(t.sos * 100).toFixed(1)}%</div>
        </div>
      </div>`
    )
    .join("");

  // Click-to-expand team details
  document.querySelectorAll(".card").forEach(card => {
    card.addEventListener("click", () => {
      const details = card.querySelector(".details");
      details.classList.toggle("hidden");
    });
  });

  // About section toggle
  const aboutToggle = document.getElementById("about-to
