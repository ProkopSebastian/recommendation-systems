import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
movies = pd.read_csv('data/movie.csv')
ratings = pd.read_csv('data/rating.csv')

class KMeansClustering:
    def __init__(self, X, num_clusters):
        self.K = num_clusters
        self.max_iterations = 100
        self.num_examples = X.shape[0]
        self.num_features = X.shape[1]

    def initialize_random_centroids(self, X):
        centroids = np.zeros((self.K, self.num_features))

        for k in range(self.K):
            centroid = X[np.random.choice(range(self.num_examples))]
            centroids[k] = centroid

        return centroids

    def create_clusters(self, X, centroids):
        # Will contain a list of the points that are associated with that specific cluster
        clusters = [[] for _ in range(self.K)]

        # Loop through each point and check which is the closest cluster
        for point_idx, point in enumerate(X):
            closest_centroid = np.argmin(
                np.sqrt(np.sum((point - centroids) ** 2, axis=1))
            )
            clusters[closest_centroid].append(point_idx)

        return clusters

    def calculate_new_centroids(self, clusters, X):
        centroids = np.zeros((self.K, self.num_features))
        for idx, cluster in enumerate(clusters):
            new_centroid = np.mean(X[cluster], axis=0)
            centroids[idx] = new_centroid

        return centroids

    def predict_cluster(self, clusters, X):
        y_pred = np.zeros(self.num_examples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                y_pred[sample_idx] = cluster_idx

        return y_pred

    def fit(self, X):
        centroids = self.initialize_random_centroids(X)

        for it in range(self.max_iterations):
            clusters = self.create_clusters(X, centroids)

            previous_centroids = centroids
            centroids = self.calculate_new_centroids(clusters, X)

            diff = centroids - previous_centroids

            if not diff.any():
                print("Termination criterion satisfied")
                break

        # Get label predictions
        y_pred = self.predict_cluster(clusters, X)

        return y_pred

def get_most_rated_movies(user_movie_ratings, max_number_of_movies):
    # 1- Count
    count_series = user_movie_ratings.count()
    count_df = pd.DataFrame([count_series.values], columns=count_series.index)
    user_movie_ratings_with_count = pd.concat([user_movie_ratings, count_df], ignore_index=True)
    
    # 2- sort
    user_movie_ratings_sorted = user_movie_ratings_with_count.sort_values(len(user_movie_ratings_with_count)-1, axis=1, ascending=False)
    user_movie_ratings_sorted = user_movie_ratings_sorted.drop(user_movie_ratings_sorted.tail(1).index)
    
    # 3- slice
    most_rated_movies = user_movie_ratings_sorted.iloc[:, :max_number_of_movies]
    
    return most_rated_movies

def get_users_who_rate_the_most(user_movie_ratings, max_number_of_users):
    # 1- Count
    user_movie_ratings['total_ratings'] = user_movie_ratings.count(axis=1)
    
    # 2- Sort
    user_movie_ratings_sorted = user_movie_ratings.sort_values(by='total_ratings', ascending=False)
    
    # 3- Slice
    users_who_rate_the_most = user_movie_ratings_sorted.iloc[:max_number_of_users, :-1]  # Exclude 'total_ratings' column
    
    return users_who_rate_the_most

# Define the sorting by rating function
def sort_by_rating_density(user_movie_ratings, n_movies, n_users):
    most_rated_movies = get_most_rated_movies(user_movie_ratings, n_movies)
    most_rated_movies = get_users_who_rate_the_most(most_rated_movies, n_users)
    return most_rated_movies

def find_movies(number_of_movies, user_id):
    # Przycinanie tabeli na podstawie popularności filmów
    # Obliczanie liczby ocen dla każdego filmu
    movie_counts = ratings['movieId'].value_counts()

    # Lista najpopularniejszych filmów (np. 1000 najpopularniejszych)
    top_movies = movie_counts.head(5000) #.index.tolist()

    # Przycinanie do najpopularniejszych filmów
    ratings_pruned = ratings[ratings['movieId'].isin(top_movies)]

    # Merge the two tables then pivot so we have Users X Movies dataframe
    ratings_title = pd.merge(ratings_pruned, movies[['movieId', 'title']], on='movieId' )
    user_movie_ratings = pd.pivot_table(ratings_title, index='userId', columns= 'title', values='rating')
    most_rated_movies_1k = get_most_rated_movies(user_movie_ratings, 1000)

    most_rated_movies_1k_small = get_users_who_rate_the_most(most_rated_movies_1k, 10000)
    imputer = SimpleImputer(strategy='mean')
    small_imputed = imputer.fit_transform(most_rated_movies_1k_small)
    # small = pd.DataFrame(small_imputed)
    small = pd.DataFrame(small_imputed, columns=most_rated_movies_1k_small.columns)
    selected_rows = small.index[:10000]
    selected_columns = small.columns[:1000]

    small = small.loc[selected_rows, selected_columns]

    num_clusters = 7
    X = small.values

    # Tutaj dzieje się cała magia
    Kmeans = KMeansClustering(X, num_clusters)
    y_pred = Kmeans.fit(X)

    # Dodaj nową kolumnę do small z przypisanymi grupami
    small.insert(0, 'userId', most_rated_movies_1k_small.index)
    small.insert(1, 'group', y_pred)

    # Teraz dla wybranego użytkownika trzeba określić jego grupę
    user_row = small.loc[small['userId'] == user_id]

    # Sprawdź wartość w kolumnie 'group' dla wybranego użytkownika
    user_group = user_row['group'].values[0]

    cluster = small[small.group == user_group].drop(['group'], axis=1)
    user_2_ratings  = most_rated_movies_1k_small.loc[user_id, :]

    # Which movies did they not rate? 
    user_2_unrated_movies =  user_2_ratings[user_2_ratings.isnull()]
    # What are the ratings of these movies the user did not rate?
    avg_ratings = pd.concat([user_2_unrated_movies, cluster.mean()], axis=1, join='inner').loc[:,0]
    # Let's sort by rating so the highest rated movies are presented first
    ans = avg_ratings.sort_values(ascending=False)[:number_of_movies]

    result_df = pd.DataFrame(ans.index, columns=['title'])
    
    return result_df