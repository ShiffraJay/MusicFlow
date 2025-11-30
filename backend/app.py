from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, pandas as pd, numpy as np, os, json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from math import sqrt
from sklearn.metrics import mean_squared_error
from collections import defaultdict

app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)

# -----------------------------
# 🔹 Load model and data
# -----------------------------
MODEL_PATH = "music_model.pkl"
CSV_PATH = "Music.csv"

if not os.path.exists(MODEL_PATH) or not os.path.exists(CSV_PATH):
    raise FileNotFoundError("❌ Missing files: ensure music_model.pkl and Music.csv are in the same folder.")

with open(MODEL_PATH, "rb") as f:
    model_data = pickle.load(f)

df = model_data["df"]
scaler = model_data["scaler"]
user_song_matrix = model_data.get("user_song_matrix")
svd = model_data.get("svd")
scaled = model_data.get("scaled")

# Optional: precompute cosine similarity matrix for fast hybrid recs
content_similarity = cosine_similarity(scaled)

# Detect the correct title column
title_col = "name" if "name" in df.columns else "title"

# Create genre clusters using KMeans on audio features (acts as genre proxy)
print("🎵 Creating genre clusters from audio features...")
n_clusters = min(50, len(df) // 100)  # Adaptive number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['genre_cluster'] = kmeans.fit_predict(scaled)
print(f"✅ Created {n_clusters} genre clusters")

# Precompute artist-based similarity (artists often share genres)
artist_songs = df.groupby('artist')[title_col].apply(list).to_dict()

# In-memory user feedback
users = {}
feedback_log = []
# Track recommendations shown to each user
user_recommendations = defaultdict(list)  # user_id -> list of recommended songs

# -----------------------------
# 🔹 Recommendation Functions
# -----------------------------
def recommend_content(song, top_k=5):
    if song.lower() not in df[title_col].str.lower().values:
        return []

    idx = df[df[title_col].str.lower() == song.lower()].index[0]
    sims = cosine_similarity(scaled[idx].reshape(1, -1), scaled).flatten()
    indices = sims.argsort()[::-1][1:top_k + 1]
    return df.iloc[indices][[title_col, "artist"]].to_dict(orient="records")


def recommend_cf(song, top_k=5):
    if song not in user_song_matrix.columns:
        return []

    song_vec = user_song_matrix[song].values.reshape(1, -1)
    sims = cosine_similarity(song_vec, user_song_matrix.T).flatten()
    indices = sims.argsort()[::-1][1:top_k + 1]
    return [{"name": user_song_matrix.columns[i]} for i in indices]


def recommend_hybrid(song, top_k=5):
    c1 = recommend_content(song, top_k)
    c2 = recommend_cf(song, top_k)
    combined = pd.DataFrame(c1 + c2).drop_duplicates(subset=title_col)
    return combined.head(top_k).to_dict(orient="records")


def recommend_by_genre_cluster(liked_songs, top_k=10):
    """Recommend songs from the same genre clusters as liked songs"""
    if not liked_songs:
        return []
    
    liked_indices = df[df[title_col].isin(liked_songs)].index
    if len(liked_indices) == 0:
        return []
    
    # Get genre clusters of liked songs
    liked_clusters = df.loc[liked_indices, 'genre_cluster'].unique()
    
    # Find songs in same clusters (excluding already liked songs)
    genre_recs = df[
        (df['genre_cluster'].isin(liked_clusters)) & 
        (~df[title_col].isin(liked_songs))
    ]
    
    if len(genre_recs) == 0:
        return []
    
    # Score by cluster frequency and audio similarity
    cluster_counts = df.loc[liked_indices, 'genre_cluster'].value_counts()
    genre_recs = genre_recs.copy()
    genre_recs['cluster_score'] = genre_recs['genre_cluster'].map(cluster_counts).fillna(0)
    
    # Sort by cluster score and take top_k
    genre_recs = genre_recs.sort_values('cluster_score', ascending=False).head(top_k)
    
    return genre_recs[[title_col, "artist"]].to_dict(orient="records")


def recommend_by_artist(liked_songs, top_k=10):
    """Recommend songs from artists of liked songs (genre proxy)"""
    if not liked_songs:
        return []
    
    liked_indices = df[df[title_col].isin(liked_songs)].index
    if len(liked_indices) == 0:
        return []
    
    # Get artists of liked songs
    liked_artists = df.loc[liked_indices, 'artist'].unique()
    
    # Find songs by same artists (excluding already liked songs)
    artist_recs = df[
        (df['artist'].isin(liked_artists)) & 
        (~df[title_col].isin(liked_songs))
    ]
    
    if len(artist_recs) == 0:
        return []
    
    # Score by artist frequency
    artist_counts = df.loc[liked_indices, 'artist'].value_counts()
    artist_recs = artist_recs.copy()
    artist_recs['artist_score'] = artist_recs['artist'].map(artist_counts).fillna(0)
    
    # Sort and take top_k
    artist_recs = artist_recs.sort_values('artist_score', ascending=False).head(top_k)
    
    return artist_recs[[title_col, "artist"]].to_dict(orient="records")

# -----------------------------
# 🔹 User System
# -----------------------------
@app.route("/user", methods=["POST"])
def create_user():
    data = request.get_json()
    user_id = data.get("user_id", "guest")
    if user_id not in users:
        users[user_id] = {"likes": [], "dislikes": []}
        print(f"🧑‍💻 Created new user: {user_id}")
    return jsonify({"message": f"User {user_id} active"})

# -----------------------------
# 🔹 Feedback
# -----------------------------
@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    user_id = data.get("user_id", "guest")
    song = data.get("song")
    liked = bool(data.get("liked"))

    if user_id not in users:
        users[user_id] = {"likes": [], "dislikes": []}

    if liked:
        users[user_id]["likes"].append(song)
    else:
        users[user_id]["dislikes"].append(song)

    feedback_log.append({"user_id": user_id, "song": song, "liked": liked})
    print(f"🎧 Feedback from {user_id}: {'❤️' if liked else '💔'} {song}")
    return jsonify({"status": "feedback logged"})

# -----------------------------
# 🔹 Hybrid Personalized Recs
# -----------------------------
@app.route("/recommend/hybrid", methods=["POST"])
def hybrid_recommend():
    data = request.get_json(force=True)
    user_id = data.get("user_id", "guest")
    top_n = int(data.get("top_n", 12))

    if user_id not in users:
        users[user_id] = {"likes": [], "dislikes": []}

    liked_songs = users[user_id]["likes"]
    disliked_songs = users[user_id]["dislikes"]

    all_recs = []
    
    if liked_songs:
        liked_indices = df[df[title_col].isin(liked_songs)].index
        if len(liked_indices) > 0:
            # 1. Content-based: Similar audio features (40% weight)
            avg_vector = content_similarity[liked_indices].mean(axis=0)
            scores = list(enumerate(avg_vector))
            scores = sorted(scores, key=lambda x: x[1], reverse=True)
            content_recs = [i for i, _ in scores if df.iloc[i][title_col] not in liked_songs][:top_n]
            
            # 2. Genre-based: Same genre clusters (40% weight)
            genre_recs_list = recommend_by_genre_cluster(liked_songs, top_n)
            genre_recs = df[df[title_col].isin([r[title_col] for r in genre_recs_list])].index.tolist()
            
            # 3. Artist-based: Same artists (20% weight)
            artist_recs_list = recommend_by_artist(liked_songs, top_n // 2)
            artist_recs = df[df[title_col].isin([r[title_col] for r in artist_recs_list])].index.tolist()
            
            # Combine with weights
            rec_indices = []
            seen_titles = set(liked_songs)
            
            # Add genre-based recs (prioritize genre diversity)
            for idx in genre_recs[:int(top_n * 0.4)]:
                if df.iloc[idx][title_col] not in seen_titles:
                    rec_indices.append(idx)
                    seen_titles.add(df.iloc[idx][title_col])
            
            # Add content-based recs
            for idx in content_recs[:int(top_n * 0.4)]:
                if df.iloc[idx][title_col] not in seen_titles:
                    rec_indices.append(idx)
                    seen_titles.add(df.iloc[idx][title_col])
            
            # Add artist-based recs
            for idx in artist_recs[:int(top_n * 0.2)]:
                if df.iloc[idx][title_col] not in seen_titles:
                    rec_indices.append(idx)
                    seen_titles.add(df.iloc[idx][title_col])
            
            # Fill remaining slots with content-based
            for idx in content_recs:
                if len(rec_indices) >= top_n:
                    break
                if df.iloc[idx][title_col] not in seen_titles:
                    rec_indices.append(idx)
                    seen_titles.add(df.iloc[idx][title_col])
            
            rec_df = df.iloc[rec_indices] if rec_indices else df.sample(top_n)
        else:
            rec_df = df.sample(top_n)
    else:
        # No liked songs: return diverse recommendations
        rec_df = df.sample(top_n)

    if disliked_songs:
        rec_df = rec_df[~rec_df[title_col].isin(disliked_songs)]

    recs = []
    for _, row in rec_df.iterrows():
        recs.append({
            "song_id": row.get("spotify_id", ""),
            "title": row.get(title_col, ""),
            "artist": row.get("artist", ""),
            "image_url": row.get("img") or "https://placehold.co/200x200?text=🎵",
            "audio_url": row.get("preview") or "",
        })

    # Track which songs were recommended to this user
    recommended_titles = [r["title"] for r in recs]
    user_recommendations[user_id].extend(recommended_titles)
    
    print(f"🎯 Generated {len(recs)} recs for {user_id} (liked: {len(liked_songs)})")
    return jsonify({"source": "hybrid", "user": user_id, "recommendations": recs})

# -----------------------------
# 🔹 Evaluation Endpoint
# -----------------------------
@app.route("/metrics", methods=["GET"])
def get_metrics():
    """Improved evaluation metrics for recommendation system"""
    if not feedback_log:
        return jsonify({
            "Precision@K": 0, 
            "Recall@K": 0, 
            "NDCG@K": 0,
            "Diversity": 0,
            "Coverage": 0,
            "RMSE": None,
            "Total_Feedback": 0
        })
    
    # Track user feedback
    total_feedback = len(feedback_log)
    positive_feedback = sum(1 for f in feedback_log if f["liked"])
    negative_feedback = total_feedback - positive_feedback
    
    # Build user feedback maps
    user_likes = defaultdict(set)
    user_dislikes = defaultdict(set)
    
    for f in feedback_log:
        user_id = f["user_id"]
        if f["liked"]:
            user_likes[user_id].add(f["song"])
        else:
            user_dislikes[user_id].add(f["song"])
    
    # PRECISION@K: Of all songs we RECOMMENDED, how many did users LIKE?
    # This measures recommendation quality!
    precision_scores = []
    for user_id in users:
        recommended_songs = user_recommendations.get(user_id, [])
        if len(recommended_songs) == 0:
            continue
        
        # Count how many recommended songs were liked
        liked_from_recs = sum(1 for song in recommended_songs if song in user_likes[user_id])
        total_recommended = len(recommended_songs)
        
        if total_recommended > 0:
            user_precision = liked_from_recs / total_recommended
            precision_scores.append(user_precision)
    
    precision = round(np.mean(precision_scores), 4) if precision_scores else 0
    
    # RECALL@K: Of all songs users LIKED, how many did we RECOMMEND?
    recall_scores = []
    for user_id in users:
        user_liked_songs = user_likes[user_id]
        if len(user_liked_songs) == 0:
            continue
        
        recommended_songs = set(user_recommendations.get(user_id, []))
        # Count how many liked songs were in our recommendations
        recommended_and_liked = len(user_liked_songs & recommended_songs)
        
        if len(user_liked_songs) > 0:
            user_recall = recommended_and_liked / len(user_liked_songs)
            recall_scores.append(user_recall)
    
    recall = round(np.mean(recall_scores), 4) if recall_scores else 0
    
    # NDCG@K: Ranking quality - are liked songs at the top of recommendations?
    ndcg_scores = []
    for user_id in users:
        recommended_songs = user_recommendations.get(user_id, [])
        user_liked_set = user_likes[user_id]
        
        if len(recommended_songs) == 0 or len(user_liked_set) == 0:
            continue
        
        # Calculate DCG: liked songs at top positions get higher scores
        dcg = 0.0
        for i, song in enumerate(recommended_songs):
            if song in user_liked_set:
                # Position i (0-indexed) -> relevance score 1
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG: ideal ranking (all liked songs at top)
        num_liked_in_recs = len(user_liked_set & set(recommended_songs))
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(num_liked_in_recs, len(recommended_songs))))
        
        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
    
    ndcg = round(np.mean(ndcg_scores), 4) if ndcg_scores else 0
    
    # Diversity: Measure how diverse recommendations are (genre cluster diversity)
    all_recommended_songs = set()
    for user_id in users:
        all_recommended_songs.update(user_recommendations.get(user_id, []))
    
    if len(all_recommended_songs) > 0:
        recommended_indices = df[df[title_col].isin(all_recommended_songs)].index
        unique_clusters = len(df.loc[recommended_indices, 'genre_cluster'].unique())
        total_clusters = len(df['genre_cluster'].unique())
        diversity = round(unique_clusters / total_clusters, 4) if total_clusters > 0 else 0
    else:
        diversity = 0
    
    # Coverage: Percentage of catalog recommended
    coverage = round(len(all_recommended_songs) / len(df), 4) if len(df) > 0 else 0
    
    # RMSE per user: Calculate prediction error based on actual user feedback
    # Compare predicted ratings vs actual feedback (liked=1, disliked=0, not rated=neutral)
    user_rmses = []
    
    for user_id in users:
        # Get all songs recommended to this user
        recommended_songs = user_recommendations.get(user_id, [])
        if len(recommended_songs) == 0:
            continue
            
        actual_ratings = []
        predicted_ratings = []
        
        # Get user's liked/disliked songs
        user_liked_set = set(users[user_id]["likes"])
        user_disliked_set = set(users[user_id]["dislikes"])
        
        # For each recommended song, calculate prediction vs actual
        for song in recommended_songs:
            # Actual rating: 1 if liked, 0 if disliked, 0.5 if not rated yet
            if song in user_liked_set:
                actual = 1.0
            elif song in user_disliked_set:
                actual = 0.0
            else:
                # Song was recommended but user hasn't rated it - skip for RMSE calculation
                continue
            
            actual_ratings.append(actual)
            
            # Predicted: Use content similarity to user's liked songs
            if song in df[title_col].values:
                song_idx = df[df[title_col] == song].index[0]
                if len(users[user_id]["likes"]) > 0:
                    # Calculate similarity to user's liked songs
                    liked_indices = df[df[title_col].isin(users[user_id]["likes"])].index
                    if len(liked_indices) > 0:
                        similarities = content_similarity[song_idx, liked_indices]
                        predicted = float(np.mean(similarities))
                    else:
                        predicted = 0.5
                else:
                    # No preferences yet - neutral prediction
                    predicted = 0.5
            else:
                predicted = 0.5
            
            predicted_ratings.append(max(0.0, min(1.0, predicted)))
        
        # Calculate RMSE for this user if we have ratings
        if len(actual_ratings) > 0 and len(predicted_ratings) > 0:
            user_rmse = sqrt(mean_squared_error(actual_ratings, predicted_ratings))
            user_rmses.append(user_rmse)
    
    # Average RMSE across all users with feedback
    if user_rmses:
        rmse = round(np.mean(user_rmses), 4)
    else:
        # No user feedback yet - return None instead of constant model RMSE
        rmse = None

    return jsonify({
        "Precision@K": precision,
        "Recall@K": recall,
        "NDCG@K": ndcg,
        "Diversity": diversity,
        "Coverage": coverage,
        "RMSE": rmse,
        "Total_Feedback": total_feedback,
        "Positive_Feedback": positive_feedback,
        "Negative_Feedback": negative_feedback
    })

# -----------------------------
# 🔹 Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)