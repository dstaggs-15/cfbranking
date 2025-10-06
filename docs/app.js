// ---- helpers: colors ----
function hashColorHSL(str){
  let h=0; for(let i=0;i<str.length;i++) h = (h*31 + str.charCodeAt(i))>>>0;
  const hue = h % 360, sat = 68, light = 46;
  return `hsl(${hue} ${sat}% ${light}%)`;
}
function maybeLighter(hsl, pct=12){
  const m = hsl.match(/hsl\((\d+)\s+(\d+)%\s+(\d+)%\)/);
  if(!m) return hsl;
  const H=+m[1], S=+m[2], L=Math.min(95, +m[3]+pct);
  return `hsl(${H} ${S}% ${L}%)`;
}

async function loadTeamColors(){
  try{
    const res = await fetch('data/team_colors.json', {cache:'no-store'});
    if(!res.ok) return {};
    const arr = await res.json();
    const map = {};
    for(const rec of arr){
      const p = rec.primary && rec.primary.startsWith("#") ? rec.primary : null;
      const s = rec.secondary && rec.secondary.startsWith("#") ? rec.secondary : null;
      map[rec.team] = { primary:p, secondary:s };
    }
    return map;
  }catch{ return {}; }
}
function teamPalette(team, colorsMap){
  const c = colorsMap[team];
  if(c && c.primary){
    const p = c.primary;
    const s = c.secondary || maybeLighter(`hsl(0 0% 50%)`, 18);
    return { primary:p, secondary:s };
  }
  const p = hashColorHSL(team);
  const s = maybeLighter(p, 18);
  return { primary:p, secondary:s };
}

// ---- UI ----
function bubble(label, val){
  const el = document.createElement('div');
  el.className = 'bubble';
  const lab = document.createElement('span'); lab.className='label'; lab.textContent = label;
  const v = document.createElement('span'); v.className='val'; v.textContent = val;
  el.append(lab, v);
  return el;
}
function card(team, palette){
  const el = document.createElement('div');
  el.className = 'card';
  el.style.setProperty('--ring', palette.primary);
  el.style.setProperty('--ring2', palette.secondary);

  const rank = document.createElement('div'); rank.className='rank-badge'; rank.textContent = `#${team.rank}`;
  const name = document.createElement('div'); name.className='team-name'; name.textContent = team.team;
  const rc = document.createElement('div'); rc.className='rec-conf'; rc.textContent = `${team.record} • ${team.conference}`;

  const bubbles = document.createElement('div'); bubbles.className='bubbles';
  bubbles.append(
    bubble('Final', team.final_score.toFixed(2)),
    bubble('Perf', team.performance.toFixed(2)),
    bubble('SOS', team.sos.toFixed(2)),
    bubble('Q-Wins', team.quality_wins.toFixed(2))
  );

  el.append(rank, name, rc, bubbles);
  el.addEventListener('click', () => openModal(team, palette));
  return el;
}
function kv(key, val){
  const wrap = document.createElement('div'); wrap.className='kv';
  const k = document.createElement('div'); k.className='key'; k.textContent = key;
  const v = document.createElement('div'); v.className='val'; v.textContent = val;
  wrap.append(k,v); return wrap;
}
function openModal(team, palette){
  const modal = document.getElementById('modal');
  const content = document.getElementById('modal-content');
  content.innerHTML = '';

  const h2 = document.createElement('h2');
  h2.textContent = `${team.team} — #${team.rank}`;
  content.appendChild(h2);

  const row = document.createElement('div'); row.className='row';
  const chips = [
    `Record ${team.record}`,
    `Perf #${team.performance_rank}`,
    `SOS #${team.sos_rank}`,
    `Games ${team.games}`,
    `Win% ${(team.win_pct*100).toFixed(1)}%`
  ];
  for(const c of chips){ const chip = document.createElement('div'); chip.className='chip'; chip.textContent=c; row.appendChild(chip); }
  content.appendChild(row);

  const k1 = kv('Final Score', team.final_score.toFixed(3));
  const k2 = kv('Performance', team.performance.toFixed(2));
  const k3 = kv('Strength of Schedule', team.sos.toFixed(2));
  const k4 = kv('Quality Wins (scaled)', team.quality_wins.toFixed(3));
  content.append(k1,k2,k3,k4);

  const sep = document.createElement('div'); sep.style.height='8px'; content.appendChild(sep);

  const oTitle = document.createElement('h3'); oTitle.textContent = 'Offense (season advanced)'; content.appendChild(oTitle);
  content.append(
    kv('PPA', team.off_ppa.toFixed(4)),
    kv('Success Rate', team.off_success_rate.toFixed(4)),
    kv('Explosiveness', team.off_explosiveness.toFixed(4)),
    kv('Pts / Opp', team.off_pts_per_opp.toFixed(4)),
  );

  const dTitle = document.createElement('h3'); dTitle.textContent = 'Defense (season advanced)'; content.appendChild(dTitle);
  content.append(
    kv('PPA (lower is better)', team.def_ppa.toFixed(4)),
    kv('Success Rate', team.def_success_rate.toFixed(4)),
    kv('Explosiveness', team.def_explosiveness.toFixed(4)),
    kv('Pts / Opp', team.def_pts_per_opp.toFixed(4)),
  );

  const modalCard = document.querySelector('.modal-card');
  modalCard.style.outline = `2px solid ${palette.primary}`;
  modalCard.style.boxShadow = `0 12px 28px rgba(0,0,0,.5), 0 0 0 3px ${palette.secondary} inset`;

  modal.classList.remove('hidden');
  modal.setAttribute('aria-hidden','false');
}
function closeModal(){ const m=document.getElementById('modal'); m.classList.add('hidden'); m.setAttribute('aria-hidden','true'); }
document.getElementById('close-modal').addEventListener('click', closeModal);
document.querySelector('.modal-backdrop').addEventListener('click', closeModal);
document.addEventListener('keydown', (e) => { if(e.key==='Escape') closeModal(); });

function bindTabs(){
  const tabs = document.querySelectorAll('.tab');
  tabs.forEach(t=>{
    t.addEventListener('click', ()=>{
      tabs.forEach(x=>x.classList.remove('active'));
      t.classList.add('active');
      const which = t.dataset.tab;
      document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
      document.getElementById(`panel-${which}`).classList.add('active');
    });
  });
}

async function main(){
  bindTabs();
  const meta = document.getElementById('meta');
  const grid = document.getElementById('bubble-grid');
  try{
    const [data, colorsMap] = await Promise.all([
      fetch('data/rankings.json', {cache:'no-store'}).then(r=>r.json()),
      loadTeamColors()
    ]);

    meta.textContent = `Season ${data.season} • Last build: ${new Date(data.last_build_utc).toLocaleString()}`;

    grid.innerHTML = '';
    data.top25.forEach(team => {
      const palette = teamPalette(team.team, colorsMap);
      const c = card(team, palette);
      grid.appendChild(c);
    });
  }catch(err){
    console.error(err);
    meta.textContent = "Error loading rankings.";
  }
}
main();
