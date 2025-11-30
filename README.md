# 🎧 MusicFlow – Hybrid Music Recommendation System

MusicFlow is an AI-powered hybrid music recommendation system that generates personalized music suggestions using audio feature analysis, clustering, and collaborative filtering. It simulates the core logic used by platforms like Spotify through machine learning techniques such as Truncated SVD, KMeans clustering, and cosine similarity. The system includes a Flask backend that loads a trained model (`music_model.pkl`) and provides recommendations to a lightweight HTML/CSS/JS frontend.

---

## 🚀 Features
- Hybrid recommendation engine  
- Content-based filtering using cosine similarity  
- Genre clustering using KMeans  
- Dimensionality reduction with Truncated SVD  
- In-memory user feedback (likes/dislikes)  
- Evaluation metrics (Precision, Recall, NDCG, RMSE)  
- Flask backend with REST APIs  
- Simple and clean web interface  

---

## 🧠 How It Works
1. **Training Notebook (`training_model.ipynb`)**  
   - Loads dataset (`Music.csv`)  
   - Scales audio features  
   - Extracts latent factors using Truncated SVD  
   - Creates reduced feature vectors  
   - (Optional) Builds user-song matrix  
   - Saves all trained components into `music_model.pkl`  

2. **Backend (Flask)**  
   - Loads the trained `.pkl` model  
   - Applies KMeans clustering on startup  
   - Generates hybrid recommendations  
   - Logs user feedback and computes metrics  

3. **Frontend**  
   - User inputs ID  
   - Sends request to backend  
   - UI displays recommended songs, album images, and previews  

---

## 📁 Project Structure
