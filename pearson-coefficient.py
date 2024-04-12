import math
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def main():
    while (True):
        t = input("Number of neighbours: ")
        if (t.isnumeric): break
    inp = str.split(t, sep=' ')

    n_neighbours = int(inp[0])

    with warnings.catch_warnings(action='ignore'):
        csv = pd.read_csv('./ratings.csv')
        csv.drop(columns=['timestamp'], inplace=True)
        user_sum_ratings = csv.groupby('userId')['rating'].sum()
        valid_users = user_sum_ratings[user_sum_ratings >= math.ceil(0.15 * csv['movieId'].nunique())].index ## 0.15: 15% of users
        reducedCsv = csv[csv['userId'].isin(valid_users)]

        ## Split csv to train/set proportions
        df = reducedCsv.pivot_table(index='userId', columns='movieId', values='rating')
        #dfCopy = df.copy()
        #dfCopy['averageRating'] = dfCopy.mean(axis=1, skipna=True)

        train_x, test_x = train_test_split(reducedCsv, test_size=0.2, random_state=8)

        for item, row in test_x.iterrows():
            userId = row['userId']
            movieId = row['movieId']

            if userId in df.index and movieId in df.columns:
                df.at[userId, movieId] = 'nan'

        print("Normalizing df...")
        df_numeric = df.apply(pd.to_numeric, errors='coerce')
        normalized_df = (df_numeric - df_numeric.mean(skipna=True)) / (df_numeric.max() - df_numeric.min())
        normalized_df.fillna(0, inplace=True)
        movie_similarity = normalized_df.corr(method='pearson')

        print("Calculating predictions...")
        hit = 0
        total = 0
        for item, row in test_x.iterrows():
            userId = row['userId']
            movieId = row['movieId']
            rating = row['rating']
            if (np.isnan(rating) or rating < 3): continue

            similarity_df = pd.DataFrame(data=movie_similarity.values, columns=normalized_df.columns, index=normalized_df.columns)
            nearest_movies = similarity_df.loc[movieId].sort_values(ascending=False, inplace=False)[1:n_neighbours+1]

            #pred = predictions_n_neighbours(test_x, nearest_movies, userId)
            #pred = predictions_n_neighbours_bias(test_x, nearest_movies, userId)
            pred = predictions_n_neighbours_custom_weight(test_x, nearest_movies, userId)
            #pred = predictions_n_neighbours_custom_weight_variance(test_x, nearest_movies, userId)

            if (pred == -1 or pred == 0): continue

            diff = abs(pred - rating)

            if (diff < 1): hit += 0.5
            elif (diff == 0): hit += 1
            
            total += 1
        accuracy = hit / total * 100
        print("Accuracy: {:.2f}%".format(accuracy))
        


def predictions_n_neighbours(df, similarities, userId):
    weightedSum = 0
    sumWeights = 0

    for movieId, score in similarities.items():
        if (np.isnan(score)): continue

        row = df.loc[(df['userId'] == userId) & (df['movieId'] == movieId), 'rating'] ## Find row in test dataframe

        if (row.empty or row.iloc[0] == 0.0): continue

        weightedSum += row.iloc[0] * score ## weighted_sum = rating * pearson_similarity
    

    if (weightedSum == 0): return -1

    sumWeights = np.sum(np.abs(similarities))
    pred = weightedSum / sumWeights

    return round(pred, 1)



def predictions_n_neighbours_bias(df, similarities, userId):
    weightedSum = 0
    sumWeights = 0
    avgUserRating = df.loc[df['userId'] == userId, 'rating'].mean() ## Calculate average rating of target user

    for movieId, score in similarities.items():
        if (np.isnan(score)): continue

        row = df.loc[(df['userId'] == userId) & (df['movieId'] == movieId), 'rating'] ## Find the row in the test dataframe

        if (row.empty or row.iloc[0] == 0.0): continue

        avgMovieRating = df.loc[df['movieId'] == movieId, 'rating'].mean() ## Calculate average rating a movie has received
        adjustedRating = row.iloc[0] - avgUserRating + avgMovieRating ## Calculate Neighbour bias

        weightedSum += adjustedRating * score
        sumWeights += np.abs(score)

    if (sumWeights == 0): return -1

    pred = weightedSum / sumWeights

    return round(pred, 1)



def predictions_n_neighbours_custom_weight(df, similarities, userId):
    weightedSum = 0
    sumWeights = 0
    userAvgRating = df.loc[df['userId'] == userId, 'rating'].mean()

    for movieId, score in similarities.items():
        if (np.isnan(score)): continue

        ## Get the common users between the current movie and the target user
        commonUsersOfMovie = df.loc[df['movieId'] == movieId, 'userId'].unique() ## Get the common users that rated the target movie
        moviesOfUser = df.loc[df['userId'] == userId, 'movieId'].unique() ## Get the common movies the target user has rated
        commonUsers = np.intersect1d(commonUsersOfMovie, moviesOfUser) ## Intersect the two

        weight = len(commonUsers) # Weight is equal to the number of common users that have rated a specific movie

        row = df.loc[(df['userId'] == userId) & (df['movieId'] == movieId), 'rating']

        if (row.empty or row.iloc[0] == 0.0): continue

        avgMovieRating = df.loc[df['movieId'] == movieId, 'rating'].mean()
        adjustedRating = row.iloc[0] - userAvgRating + avgMovieRating ## Neighbour bias, based on the above

        weightedSum += adjustedRating * score * weight
        sumWeights += np.abs(score) * weight

    if (sumWeights == 0): return -1

    pred = weightedSum / sumWeights

    return round(pred, 1)



def predictions_n_neighbours_custom_weight_variance(df, similarities, userId):
    weightedSum = 0
    sumWeights = 0
    userAvgRating = df.loc[df['userId'] == userId, 'rating'].mean()

    for movieId, score in similarities.items():
        if (np.isnan(score)): continue

        ## Get the variance of ratings for the current movie
        movieVariance = df.loc[df['movieId'] == movieId, 'rating'].var() ## Calculate the target movie variance, based on ratings

        row = df.loc[(df['userId'] == userId) & (df['movieId'] == movieId), 'rating']

        if (row.empty or row.iloc[0] == 0.0): continue

        avgMovieRating = df.loc[df['movieId'] == movieId, 'rating'].mean()
        adjustedRating = row.iloc[0] - userAvgRating + avgMovieRating ## Neighbour bias, based on the above

        weight = 1 / (1 + movieVariance)  ## Custom weight function based on variance

        weightedSum += adjustedRating * score * weight
        sumWeights += np.abs(score) * weight

    if (sumWeights == 0): return -1

    pred = weightedSum / sumWeights

    return round(pred, 1)



if __name__ == "__main__":
    main()