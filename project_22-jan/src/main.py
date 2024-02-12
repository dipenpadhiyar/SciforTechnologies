import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from streamlit_star_rating import st_star_rating

# Initialize a session state variable
if 'page' not in st.session_state:
    st.session_state.page = 'home'

def clean_title(title):
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    return title

def content_based_search(query, vectorizer, tfidf, movies):
    query_vec = vectorizer.transform([re.sub("[^a-zA-Z0-9 ]", "", query)])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices].iloc[::-1]
    return results

def collaborative_based_search(query, movies, ratings):
    movie_id = movies[movies["title"].str.contains(query, case=False)].iloc[0]["movieId"]

    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

    similar_user_recs = similar_user_recs[similar_user_recs > .10]
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())

    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)

    return rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]

def store_rating(value, search_method, query, user_ratings, ratings_file):
    print(value,"data :)")
    if value is not None and value > 0:
        new_rating = pd.DataFrame({"method": [search_method], "query": [query], "rating": [value]})
        user_ratings = pd.concat([user_ratings, new_rating], ignore_index=True)
        user_ratings.to_csv(ratings_file, index=False)
    return user_ratings

def display_results(results):
    if not results.empty:
        for index, row in results.iterrows():
            st.write(f"{row['title']} - {row['genres']}")


def star_rating(search_method, query, ratings_file):
    # Use session_state to store and retrieve values
    session_state = st.session_state

    # Use session_state to store and retrieve values
    stars_key = f"{search_method}_stars"
    query_key = f"{search_method}_query"

    # Clear session state for the specific search method when switching
    other_method = "Collaborative-Based" if search_method == "Content-Based" else "Content-Based"
    other_stars_key = f"{other_method}_stars"
    other_query_key = f"{other_method}_query"
    session_state[other_stars_key] = 0
    session_state[other_query_key] = ""

    # Get previous values from session_state
    prev_stars = session_state.get(stars_key, 0)
    prev_query = session_state.get(query_key, "")

    # Display star rating
    stars = st_star_rating("Please rate your experience", maxValue=5, defaultValue=prev_stars, key="rating")

    # Manually add a reset button
    reset_button = st.button("Reset Rating")

    # Reset the rating if the button is clicked
    if reset_button:
        stars = 0

    # Store current values in session_state
    session_state[stars_key] = stars
    session_state[query_key] = query

    # Store rating only if it's changed
    if stars > 0 and (prev_stars != stars or prev_query != query):
        print(stars, query, "data :)")
        store_rating(stars, search_method, query, user_ratings, ratings_file)

    # Remove the rating and reset button components after review
    st.empty()



# def main():

movies = pd.read_csv("../Dataset/Movie_lens/ml-25m/movies.csv")
# movies = pd.read_csv("project_22-jan/Dataset/Movie_lens/ml-25m/movies.csv")
movies["clean_title"] = movies["title"].apply(clean_title)

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf = vectorizer.fit_transform(movies["clean_title"])

ratings = pd.read_csv('../Dataset/Movie_lens/ml-25m/ratings.csv')
# movies = pd.read_csv("project_22-jan/Dataset/Movie_lens/ml-25m/ratings.csv")
ratings_file = "ratings.csv"
try:
    user_ratings = pd.read_csv(ratings_file)
except FileNotFoundError:
    user_ratings = pd.DataFrame(columns=["method", "query", "rating"])

st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #3498DB;">
    <a class="navbar-brand" target="_blank">Movie Recommendation</a>
    <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav"
            aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
        <span class="navbar-toggler-icon"></span>
    </button>
    <div class="collapse navbar-collapse" id="navbarNav">
        <ul class="navbar-nav">
            <li class="nav-item active">
                <a class="nav-link disabled" href="#" onclick="set_page('home')">Home <span
                        class="sr-only">(current)</span></a>
            </li>
            <li class="nav-item">
                <a class="nav-link" href="http://localhost:8501/statistics" onclick="set_page('statistics')">Statistics</a>
            </li>
                        <li class="nav-item">
                <a class="nav-link" href="http://localhost:8501/user_stat" onclick="set_page('user_stat')">User Review</a>
            </li>
        </ul>
    </div>
</nav>
<script>
    function set_page(page) {
        if (page === 'home') {
            // If 'home' is clicked, reload the page
            window.location.reload();
        } else if (page === 'statistics') {
            // If 'statistics' is clicked, redirect to the Statistics page
            window.location.href = "http://localhost:8501/statistics";
        } else if (page === 'user_stat') {
            // If 'statistics' is clicked, redirect to the Statistics page
            window.location.href = "http://localhost:8501/user_stat";
        }
    }
</script>
""", unsafe_allow_html=True)


# Use Markdown to inject custom CSS
st.markdown("""
<style>
    header.st-emotion-cache-uc1cuc,
    div.st-emotion-cache-1nnh243 {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

st.header("Movie Search")
query = st.text_input("Enter a movie title", "Toy Story")
search_method = st.radio("Select Search Method", ["Content-Based", "Collaborative-Based"])

if st.button("Search", use_container_width=True):
    if search_method == "Content-Based":
        content_results = content_based_search(query, vectorizer, tfidf, movies)
        st.subheader("Content-Based Results:")
        display_results(content_results)
    elif search_method == "Collaborative-Based":
        collaborative_results = collaborative_based_search(query, movies, ratings)
        st.subheader("Collaborative-Based Results:")
        display_results(collaborative_results)
        
        # Add a slider for rating with on_change callback
star_rating(search_method,query,ratings_file)
data = 0
if data != 0:
    st.switch_page("pages/statistics.py")
    st.switch_page("pages/user_stat.py")
