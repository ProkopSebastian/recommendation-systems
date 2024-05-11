import time
from surprise.model_selection import train_test_split
from surprise import Dataset, Reader, SVD, accuracy


class MatrixFactorization:
    def __init__(self, ratings, min_rating, max_rating, movies, n_epochs, n_factors):
        # Init variables
        self.n_epochs = n_epochs
        self.n_factors = n_factors
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.ratings = ratings
        self.movies = movies
        
        # Define a custom Reader object specifying the rating scale
        self.reader = Reader(rating_scale=(min_rating, max_rating))

        # Load data into Surprise Dataset
        self.data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], self.reader)

        self._get_svd_algorithm()

    def _get_svd_algorithm(self):        
        # Split the data into training and testing sets
        trainset, testset = train_test_split(self.data, test_size=0.2)
        
        # Initialize the SVD algorithm with current number of epochs
        algo = SVD(n_epochs=self.n_epochs)
        
        # Train the algorithm on the training set
        algo.fit(trainset)

        # Make predictions on the testing set
        self.predictions = algo.test(testset)
        
        # Retrieve prediction matrix
        self.prediction_matrix = algo.pu @ algo.qi.T

        # Scale predicted ratings
        max_value = self.min_rating
        min_value = self.max_rating

        for row in self.prediction_matrix:
            # Iterate over each element in the row
            for value in row:
                # Update max_value and min_value
                if value > max_value:
                    max_value = value
                if value < min_value:
                    min_value = value
      
        self.prediction_matrix = (self.prediction_matrix - min_value) / (max_value - min_value) * (self.max_rating - self.min_rating) + self.min_rating

    def get_recomendations(self, user_id, number_of_movies):
        # Get list of all user IDs
        user_ids = self.ratings['userId'].unique()

        # Sort the list of user IDs
        sorted_user_ids = sorted(user_ids)

        # Find the index of the given user ID
        index_of_given_user_id = sorted_user_ids.index(user_id)

        # Find already watched movies
        watched_movies = self.ratings[self.ratings['userId'] == user_id]['movieId'].tolist()

        # Extract predicted ratings for the specified user ID
        user_predicted_ratings = self.prediction_matrix[index_of_given_user_id - 1]

        # Filter predicted ratings for movies not watched already 
        user_predictions = [(movie_id, self.movies.loc[movie_id]['title'], rating)
                            for movie_id, rating in enumerate(user_predicted_ratings, start=1)
                            if movie_id not in watched_movies]

        # Sort predictions by rating in descending order
        user_predictions.sort(key=lambda x: x[2], reverse=True)

        # Get top n recommendations
        top_n_recommendations = user_predictions[:number_of_movies]

        return top_n_recommendations
    
    def get_rmse(self):
        # Calculate RMSE
        return accuracy.rmse(self.predictions, verbose=False)