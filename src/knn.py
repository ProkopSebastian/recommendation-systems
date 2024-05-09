import numpy as np
import pandas as pd

class KNN:
    def __init__(self, movies, ratings):
        self.movies = movies
        self.ratings = ratings
        # Load the dataset (1/10 of the data due to huge size)
        df = pd.read_csv('data/rating.csv', nrows=int(0.1 * sum(1 for line in open('data/rating.csv'))))

        # Create a pivot table of users and their ratings for movies
        self.pivot_table = df.pivot_table(index='movieId', columns='userId', values='rating').fillna(0)
        self.normalize_pivot_table()
    
    def slice_data(self, df, scale):
        total_rows = len(df)
        n_rows_to_take = total_rows // scale  # Integer division
        return df.head(n_rows_to_take)

    def normalize_pivot_table(self):
        max_value = self.pivot_table.max().max()
        min_value = self.pivot_table.min().min()
        self.pivot_table = (self.pivot_table - min_value) / (max_value - min_value)
    
    def cosine_similarity(self, u, v):
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    
    # Function to find k-nearest neighbors using cosine similarity
    # target_point - a point (x,y) to which the neighbors are to be found
    def find_nearest_neighbors(self, target_point, k=1):
        similarities = []
        # Finding vector with given movieId that contains all users' ratings
        target_vec = self.pivot_table.loc[target_point[0]].values
        
        for index, row in self.pivot_table.iterrows():
            if index == target_point[0]:
                continue
            if self.pivot_table.loc[index, row.index.isin([target_point[1]])].any() != 0:
                continue
            # For given movie vector with all users' ratings we are looking for
            # the similar vector (based on other users' ratings) using cosine similarity
            point_vec = row.values
            similarity = self.cosine_similarity(target_vec, point_vec)
            similarities.append((index, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)  # Sort similarities in descending order
        neighbors = similarities[:k]  # Get top k neighbors
        return neighbors
    
class MovieRecommender:
    def __init__(self, movies, ratings):
        self.movies = movies
        self.ratings = ratings

    def get_movie_info(self, movieId):
        movie_info = self.movies[self.movies['movieId'] == movieId]
        if len(movie_info) > 0:
            movie_info = movie_info.iloc[0]
            return f"Movie ID: {movie_info['movieId']}, Title: {movie_info['title']}"
        else:
            return "Movie not found"
        
    # Function to find movies with the highest rating rated by a given userId
    def get_top_rated_movies_for_user(self, userId):
        user_ratings = self.ratings[self.ratings['userId'] == userId]
        max_rating = user_ratings['rating'].max()
        top_movies = user_ratings[user_ratings['rating'] == max_rating]['movieId'].tolist()
        return top_movies, max_rating
    
    # Function to recommend movies
    # userId - defines to which user movies should be recommended
    # k - number of neighbors
    # all - defines if user should get recommendations to all top rated movies
    def recommend_movies(self, userId, k = 1, all = False):
        knn = KNN(self.movies, self.ratings)
        top_movies, max_rating = self.get_top_rated_movies_for_user(userId)
        neighbors = []
        print("Maximum rating for user", userId, ":", max_rating)
        print("Movies with the maximum rating:", top_movies)
        if all is True:
            for movieId in top_movies:
                neighbors.append(self.find_recommended_movies(self, knn, movieId, userId, k))
        else:
            neighbors.append(self.find_recommended_movies(knn, top_movies[0], userId, k))
        return neighbors
    
    def find_recommended_movies(self, knn, movieId, userId, k):
        nearest_neighbors = knn.find_nearest_neighbors((movieId, userId), k)
        neighbor_ids = []
        for neighbor in nearest_neighbors:
            neighbor_ids.append(neighbor[0])
        return neighbor_ids
    

