// If JSON fields change later, adjust renderCard() only.
function renderCard(entry, index){
  const sosPct = (entry.sos ?? 0) * 100;
  const avgMargin = (entry.avg_margin ?? entry.avgMargin ?? 0);
  return `
    <article class="card" tabindex="0" aria-expanded="false">
      <div class="row1">
        <div class="rank-badge">#${index + 1}</div>
        <div>
          <div class="team">${entry.team}</div>
          <div class="score">Score: ${Number(entry.score).toFixed(3)}</div>
        </div>
      </div>
      <section class="details hidden">
        <div class="kv">
          <div>Record</div><div>${entry.wins}-${entry.losses}</div>
          <div>Avg margin</div><div>${avgMargin}</div>
          <div>Strength of schedule</div><div>${sosPct.toFixed(1)}%</div>
          <div>Points for</div><div>${entry.points_for}</div>
          <div>Points against</div><div>${entry.points_against}</div>
        </div>
      </section>
    </article>
  `;
}

async function main(){
  const updated = document.getElementById("updated");
  const grid = document.getElementById("rankings");
  const empty = document.getElementById("empty");
  const aboutBtn = document.getElementById("about-toggle");
  const about = document.getElementById("about");

  aboutBtn.addEventListener("click", () => {
    about.classList.toggle("hidden");
  });

  // cache-bust fetch to avoid Pages caching old JSON
  const url = `./data/rankings.json?v=${Date.now()}`;
  let data;
  try{
    const res = await fetch(url, {cache: "no-store"});
    if(!res.ok){
      throw new Error(`HTTP ${res.status}`);
    }
    data = await res.json();
  }catch(err){
    empty.classList.remove("hidden");
    empty.textContent = `Could not load rankings.json (${err.message}).`;
    return;
  }

  updated.textContent = `Last updated ${new Date(data.last_build_utc).toLocaleString()} • Season ${data.season}`;

  const list = Array.isArray(data.top25) ? data.top25 : [];
  if(list.length === 0){
    empty.classList.remove("hidden");
    return;
  }

  grid.innerHTML = list.map(renderCard).join("");

  // click/keyboard expand for each card
  grid.querySelectorAll(".card").forEach(card => {
    const details = card.querySelector(".details");
    const toggle = () => {
      details.classList.toggle("hidden");
      card.setAttribute("aria-expanded", details.classList.contains("hidden") ? "false" : "true");
    };
    card.addEventListener("click", toggle);
    card.addEventListener("keypress", (e) => {
      if(e.key === "Enter" || e.key === " "){ e.preventDefault(); toggle(); }
    });
  });
}

main();
