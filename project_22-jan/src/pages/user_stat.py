import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px

# Initialize a session state variable
if 'page' not in st.session_state:
    st.session_state.page = 'user_stat'
    
# Load MovieLens dataset
movies = pd.read_csv("../Dataset/Movie_lens/ml-25m/movies.csv")
ratings = pd.read_csv('../Dataset/Movie_lens/ml-25m/ratings.csv')

# movies = pd.read_csv("project_22-jan/Dataset/Movie_lens/ml-25m/movies.csv")
# ratings = pd.read_csv('project_22-jan/Dataset/Movie_lens/ml-25m/ratings.csv')

st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)
st.markdown("""
            <nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #3498DB;"> 
<a class="navbar-brand" target="_blank">Movie Recommandation</a> 
<button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false"
aria-label="Toggle navigation"> <span class="navbar-toggler-icon"></span> </button>
<div class="collapse navbar-collapse" id="navbarNav">
<ul class="navbar-nav">
    <li class="nav-item active"> <a class="nav-link disabled" href="http://localhost:8501">Home <span class="sr-only">(current)</span></a>
    <li class="nav-item"> <a data-testid="stSidebarNavLink" href="http://localhost:8501" class="nav-link">Statistics</a> </li>
    <li class="nav-item"> <a data-testid="stSidebarNavLink" href="#" class="nav-link">User Ratings</a> </li>
    </li>  
</ul>
</div>
</nav>
            """, unsafe_allow_html=True)

centered_heading = """
<div data-stale="false" width="688" class="element-container st-emotion-cache-1v0ltu e1f1d6gn4" data-testid="element-container">
    <div class="stHeadingContainer" data-testid="stHeading">
        <div class="stMarkdown" style="width: 688px; text-align: center;">
            <div data-testid="stMarkdownContainer" class="st-emotion-cache-17b17hr e1nzilvr5" style="width: 688px;">
                <div class="st-emotion-cache-1629p8f e1nzilvr2">
                    <h1 id="movie-recommendation-app">
                        <div data-testid="StyledLinkIconContainer" class="st-emotion-cache-zt5igj e1nzilvr4">
                            <a href="#movie-recommendation-app" class="st-emotion-cache-1dgmtll e1nzilvr3">
                                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                    <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 a5 0 0 0-7.07-7.07l-1.72 1.71"></path>
                                    <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path>
                                </svg>
                            </a>
                            <span class="st-emotion-cache-10trblm e1nzilvr1">User Ratings</span>
                        </div>
                    </h1>
                </div>
            </div>
        </div>
    </div>
</div>
"""

st.markdown(centered_heading, unsafe_allow_html=True)

# Use Markdown to inject custom CSS
st.markdown("""
<style>
    header.st-emotion-cache-uc1cuc,
    div.st-emotion-cache-1nnh243 {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

user_ratings = pd.read_csv("ratings.csv")

st.dataframe(user_ratings)

# Histogram
fig_histogram = px.histogram(user_ratings, x='rating', color='method', nbins=5, title='Distribution of Ratings by Method')
st.plotly_chart(fig_histogram)

# Box Plot
fig_boxplot = px.box(user_ratings, x='method', y='rating', title='Box Plot of Ratings by Method')
st.plotly_chart(fig_boxplot)

# Scatter Plot
fig_scatter = px.scatter(user_ratings, x='query', y='rating', color='method', title='Scatter Plot of Ratings by Query')
st.plotly_chart(fig_scatter)
