import pandas as pd
import clustering as cl
import knn as knn_alg

# Import datasetu
movies = pd.read_csv('data/movie.csv')
ratings = pd.read_csv('data/rating.csv')
# wybór użytkownika, dla którego będziemy polecać filmy, oraz ile filmów chcemy wyświetlić
user_id = 23
number_of_movies = 10

movie_recommender = knn_alg.MovieRecommender(movies, ratings)
movie_ids_list = movie_recommender.recommend_movies(user_id, number_of_movies, False)

# Create an empty DataFrame to store the movie details
movie_details_df = pd.DataFrame(columns=['movieId', 'title', 'genres'])
rows_to_append = []
for arr in movie_ids_list:
    for movie_id in arr:
        movie_row = movies[movies['movieId'] == movie_id]
        if not movie_row.empty:
            rows_to_append.append(movie_row.iloc[0])

if rows_to_append:
    movie_details_df = pd.concat([movie_details_df, pd.DataFrame(rows_to_append)], ignore_index=True)

print(movie_details_df)