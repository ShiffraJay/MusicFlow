const API = "http://127.0.0.1:5000";

async function createUser(userId) {
  await fetch(`${API}/user`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId })
  });
}

async function getRecommendations(userId) {
  const res = await fetch(`${API}/recommend/hybrid`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, top_n: 8 })
  });

  if (!res.ok) {
    alert("Error fetching recommendations 😢");
    return;
  }

  const data = await res.json();
  renderGrid(data.recommendations);
}

function renderGrid(songs) {
  const grid = document.getElementById("recsGrid");
  grid.innerHTML = songs.map((s, index) => `
    <div class="card" data-song-id="${s.song_id || index}">
      <img src="http://127.0.0.1:5000/static/images/default_playlist.png" alt="${s.title}">
      <h4>${s.title}</h4>
      <p>${s.artist}</p>
      <div>
        <button onclick="sendFeedback('${s.title}', true)">👍 Like</button>
        <button onclick="sendFeedback('${s.title}', false)">👎 Dislike</button>
      </div>
    </div>
  `).join("");
}

async function sendFeedback(song, liked) {
  const userId = document.getElementById("userId").value || "guest";
  await fetch(`${API}/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, song, liked })
  });
  alert(`${liked ? "❤️ You liked" : "💔 You disliked"} ${song}`);
}

document.getElementById("startBtn").addEventListener("click", async () => {
  const userId = document.getElementById("userId").value || "guest";
  await createUser(userId);
  await getRecommendations(userId);
});

document.getElementById("showMetrics").addEventListener("click", async () => {
  const res = await fetch(`${API}/metrics`);
  const data = await res.json();
  
  // Create a modal for better display
  const modal = document.createElement('div');
  modal.className = 'metrics-modal';
  modal.innerHTML = `
    <div class="metrics-content">
      <h2>📊 Recommendation System Metrics</h2>
      <button class="close-btn" onclick="this.closest('.metrics-modal').remove()">×</button>
      
      <div class="metrics-section">
        <h3>🎯 Accuracy Metrics</h3>
        <div class="metric-item">
          <strong>Precision@K: ${data["Precision@K"] || 0}</strong>
        </div>
        <div class="metric-item">
          <strong>Recall@K: ${data["Recall@K"] || 0}</strong>
        </div>
        <div class="metric-item">
          <strong>NDCG@K: ${data["NDCG@K"] || 0}</strong>
        </div>
      </div>

      <div class="metrics-section">
        <h3>🎵 Quality Metrics</h3>
        <div class="metric-item">
          <strong>Diversity: ${data["Diversity"] || 0}</strong>
        </div>
        <div class="metric-item">
          <strong>Coverage: ${(data["Coverage"] * 100 || 0).toFixed(2)}%</strong>
        </div>
        <div class="metric-item">
          <strong>RMSE: ${data["RMSE"] || "N/A"}</strong>
        </div>
      </div>

      <div class="metrics-section">
        <h3>📈 Feedback Statistics</h3>
        <div class="metric-item">
          <strong>Total Feedback: ${data["Total_Feedback"] || 0}</strong>
          <p>Positive: ${data["Positive_Feedback"] || 0} | Negative: ${data["Negative_Feedback"] || 0}</p>
        </div>
      </div>
    </div>
  `;
  
  document.body.appendChild(modal);
  modal.addEventListener('click', (e) => {
    if (e.target === modal) modal.remove();
  });
});
