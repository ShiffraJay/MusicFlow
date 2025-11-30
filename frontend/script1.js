async function getRecommendations(userId = 'guest') {
    const grid = document.getElementById("recsGrid");
    grid.innerHTML = "";
    try {
        const response = await fetch("http://127.0.0.1:5000/recommend/hybrid", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ user_id: userId, top_n: 12 })
        });
        if (response.ok) {
            const data = await response.json();
            if (Array.isArray(data) && data.length) {
                data.forEach(item => grid.appendChild(card(item.song_id, item.artist, item.title, item.image_url, item.audio_url)));
                return;
            }
        }
    } catch (e) { console.warn('Hybrid error', e); }
    // Fallback to catalog
    const res = await fetch('http://127.0.0.1:5000/songs?limit=12');
    const items = await res.json();
    items.forEach(item => grid.appendChild(card(item.song_id, item.artist, item.title, item.image_url, item.audio_url)));
}

async function loadPopular() {
    const grid = document.getElementById('popularGrid');
    grid.innerHTML = '';
    try {
        const res = await fetch('http://127.0.0.1:5000/popular?top_n=12');
        if (res.ok) {
            const data = await res.json();
            if (Array.isArray(data) && data.length) {
                data.forEach(s => grid.appendChild(card(s.song_id, s.artist, s.title, s.image_url, s.audio_url)));
                return;
            }
        }
    } catch (e) { console.warn('Popular error', e); }
    const res2 = await fetch('http://127.0.0.1:5000/songs?limit=12');
    const items = await res2.json();
    items.forEach(s => grid.appendChild(card(s.song_id, s.artist, s.title, s.image_url, s.audio_url)));
}

async function loadTrending(userId = 'guest') {
    const grid = document.getElementById("trendingGrid");
    grid.innerHTML = "";
    try {
        const res = await fetch(`http://127.0.0.1:5000/trending?top_n=8&user_id=${userId}`);
        if (res.ok) {
            const data = await res.json();
            if (Array.isArray(data) && data.length) {
                data.forEach(s => grid.appendChild(card(s.song_id, s.artist, s.title, s.image_url, s.audio_url)));
                return;
            }
        }
    } catch (e) { console.warn('Trending error', e); }
    const res2 = await fetch('http://127.0.0.1:5000/songs?limit=8');
    const items = await res2.json();
    items.forEach(s => grid.appendChild(card(s.song_id, s.artist, s.title, s.image_url, s.audio_url)));
}

async function loadCatalogFallback() {
  // If any grid is still empty, seed it from /songs
  const catalogs = await fetch('http://127.0.0.1:5000/songs?limit=12').then(r=>r.json()).catch(()=>[]);
  const grids = [
    document.getElementById('trendingGrid'),
    document.getElementById('popularGrid'),
    document.getElementById('recsGrid')
  ];
  grids.forEach(g => {
    if (g && g.children.length === 0) {
      catalogs.forEach(s => g.appendChild(card(s.song_id, s.artist, s.title, s.image_url, s.audio_url)));
    }
  });
}


async function sendInteraction(userId, songId, type) {
    await fetch("http://127.0.0.1:5000/interaction", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: userId, song_id: songId, type })
    });
}

function card(id, sub, title, imageUrl, audioUrl){
  const el = document.createElement('div');
  el.className = 'card';
  el.innerHTML = `
    <div class="thumb" style="background-image:url('${imageUrl || ''}');background-size:cover;background-position:center">${imageUrl ? '' : '🎵'}</div>
    <div class="title">${title || id}</div>
    <div class="sub">${sub || ''}</div>
  `;
  el.addEventListener('click',()=>{
    const userId = document.getElementById('userId').value || 'guest';
    sendInteraction(userId, id, 'play');
    document.getElementById('nowTitle').textContent = title || id;
    document.getElementById('nowArtist').textContent = sub || '';
    playAudio(audioUrl);
  });
  return el;
}

const audio = document.getElementById('audio');
function playAudio(url){
  if(url){ audio.src = url; }
  if(!audio.src){ audio.src = 'data:audio/mp3;base64,//uQZAAAAAAAAAAAAAAAAAAAA'; }
  audio.play().catch(()=>{});
}
document.getElementById('playBtn').addEventListener('click',()=>{
  if(audio.paused){audio.play().catch(()=>{});} else {audio.pause();}
});
document.getElementById('vol').addEventListener('input',e=>{audio.volume = e.target.value/100});
async function setupSearch(){
  const input = document.getElementById('searchInput');
  let t;
  input.addEventListener('input',()=>{
    clearTimeout(t);
    t = setTimeout(async ()=>{
      const q = input.value.trim();
      if(!q){ return; }
      const uid = document.getElementById('userId').value || 'guest';
      const res = await fetch(`http://127.0.0.1:5000/songs?q=${encodeURIComponent(q)}&limit=24`);
      const data = await res.json();
      const grid = document.getElementById('trendingGrid');
      grid.innerHTML = '';
      data.forEach(s=> grid.appendChild(card(s.song_id, s.artist, s.title, s.image_url, s.audio_url)));
    }, 250);
  });
}


// === INITIAL LOAD ===
window.addEventListener('DOMContentLoaded', () => {
    const uid = document.getElementById('userId').value || 'guest';
    loadTrending(uid);   // pass userId
    loadPopular();       // Popular is fine
    getRecommendations(uid); // For You section
    setupSearch();
    // Final safety fallback after 1s if grids are empty
    setTimeout(loadCatalogFallback, 1000);
});
