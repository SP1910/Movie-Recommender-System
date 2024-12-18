import numpy as np
import pandas as pd
import pickle
import streamlit as st
import requests

#####################################################################################################
st.title('Movie Recommender System')
movies_list = pickle.load(open('Movies.pkl', 'rb'))
movies_list = pd.DataFrame(movies_list)

def stem(txt):
    y=[]
    for i in txt.split():
        y.append(ps.stem(i))
    return ' '.join(y)

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
movies_list['tags']=movies_list['tags'].apply(lambda x: stem(x))

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
vectors = cv.fit_transform(movies_list['tags']).toarray()

from sklearn.metrics.pairwise import  cosine_similarity
similarity = cosine_similarity(vectors)
###################################################################################################
def recommend(movie_name):
    recommended_posters = []
    recommended_movies = []
    idx = movies_list.loc[movies_list['title'] == movie_name].index[0]
    dist = similarity[idx]
    mr_list = sorted(list(enumerate(dist)), reverse=True, key=lambda x: x[1])[1:6]
    for i in mr_list:
        recommended_movies.append(movies_list.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movies_list.iloc[i[0]].tmdbId))
    return recommended_movies,recommended_posters
#################################################################################################
# st.write(response.text)
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?language=en-US"
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJmNThmZDViODNmZjczYTVlYTZiNDBkY2E2MDg4ZDUwNiIsIm5iZiI6MTczNDQzODU2OC4zLCJzdWIiOiI2NzYxNmVhOGJiMTU5MDdjZDMxODllYWQiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.NbgIXy5euHe1USAvZ4xuT0tHDrpCD8tvx6lUKKk5DCs"
    }
    response = requests.get(url, headers=headers)
    datajsn = response.json()
    return 'https://image.tmdb.org/t/p/w500' + datajsn["poster_path"]
####################################################################################################
movie = st.selectbox('Select Movie', movies_list['title'])
btn = st.button('Recommend')
if btn:

    names, posters = recommend(movie)
    col1, col2, col3, col4, col5 = st.columns([2,2,2,2,2])
    with col1:
        st.text(names[0])
        st.image(posters[0])
    with col2:
        st.text(names[1])
        st.image(posters[1])
    with col3:
        st.text(names[2])
        st.image(posters[2])
    with col4:
        st.text(names[3])
        st.image(posters[3])
    with col5:
        st.text(names[4])
        st.image(posters[4])
