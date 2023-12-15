#Here we import the necessary libraries
from collections import defaultdict
from typing import List
import pandas as pd
import numpy as np


class DataHandler():
    """
    Class that handles the data cleaning and feature engineering of the raw anonymised movie watch history of opted-in Netflix users (UK)
    """
    #Here we define the threshold for the probability of a user following through with a movie (10 minutes)
    FOLLOW_THROUGH_THRESHOLD = 600

    def __init__(self, raw_dataset: pd.DataFrame):
        """
        Constructor of the DataHandler class

        Args:
            raw_dataset (pandas.DataFrame): Raw anonymised movie watch history of opted-in Netflix users (UK)
        """
        #Here we clean the raw dataset
        self.cleaned_data = self.__clean_data(raw_dataset)
        #Here we obtain the LSH user data
        self.lsh_user_data = self.__get_lsh_user_data()
        #Here we obtain the clustering data
        self.clustering_data = self.__get_clustering_data()

    def fill_nan_values(self) -> pd.DataFrame:
        """
        Function that fills the NaN values in each column with the mean/mode of the column of the clustering dataset

        Returns:
            clustering_data (pandas.DataFrame): DataFrame containing the clustering data of each user with the NaN values filled
        """
        #Here we make a copy of the clustering data in order to not modify it
        clustering_data = self.clustering_data.copy()
        #Here we fill the NaN values in the mean_duration_peak_hour column with the mean of the column
        clustering_data['mean_duration_peak_hour'] = clustering_data['mean_duration_peak_hour'].fillna(clustering_data['mean_duration_peak_hour'].mean())
        #Here we fill the NaN values in the genre_diversity column with the mean of the column
        clustering_data['genre_diversity'] = clustering_data['genre_diversity'].fillna(clustering_data['genre_diversity'].mean())
        #Here we fill the NaN values in the is_old_movie_lover column with the mode of the column
        clustering_data['is_old_movie_lover'] = clustering_data['is_old_movie_lover'].fillna(clustering_data['is_old_movie_lover'].mode()[0])
        #Here we fill the NaN values in the favourite_genre column with the mode of the column
        clustering_data['favourite_genre'] = clustering_data['favourite_genre'].fillna(clustering_data['favourite_genre'].mode()[0])

        #Here we return the clustering data
        return clustering_data
    
    
    def pivot_table(self, clustering_data: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """
        Function that creates a pivot table of the clustering data

        Args:
            clustering_data (pandas.DataFrame): DataFrame containing the clustering data of each user
            column_name (str): Column to pivot the table

        Returns:
            pivot_table (pandas.DataFrame): DataFrame containing the pivot table of the clustering data
        """
        #Here we assert that the cluster column exists in the clustering data
        assert column_name in clustering_data.columns, "The 'cluster' column does not exist in the clustering data."
        
        #Here we create a pivot table of the clustering data based on the column name
        if column_name == "most_active_time_day":
            #Here we create a pivot table of the clustering data based on the most active time of day
            pivot = pd.pivot_table(clustering_data, values = 'favourite_genre', index=['cluster'], columns = ['most_active_time_day'], aggfunc='count')
        elif column_name == "most_common_click_length":
            #Here we create a pivot table of the clustering data based on the most common click length
            pivot = pd.pivot_table(clustering_data, values = 'favourite_genre', index=['cluster'], columns = ['most_common_click_length'], aggfunc='count')
        elif column_name == "ft_probability":
            #Here we create a pivot table of the clustering data based on the probability of a user following through with a movie
            #We first classify the probability of a user following through with a movie into low, medium or high
            clustering_data['ft_probability_class'] = clustering_data['ft_probability'].apply(lambda x: self.__classify_ft_probability(x))
            #We then create the pivot table
            pivot = pd.pivot_table(clustering_data, values = 'favourite_genre', index=['cluster'], columns = ['ft_probability_class'], aggfunc='count')
        else:
            print("The column name is not valid.")
        
        #Here we divide each row by the sum of the row and multiply by 100 to obtain the percentage of each category
        pivot = pivot.div(pivot.sum(axis=1), axis=0)*100

        #Here we return the pivot table
        return pivot
    
    def __classify_ft_probability(self, ft_probability: float) -> str:
        """
        Function that classifies the probability of a user following through with a movie into low, medium or high

        Args:
            ft_probability (float): Probability of a user following through with a movie

        Returns:
            ft_probability_class (str): Class of the probability of a user following through with a movie
        """
        #If the probability is less than 0.3, we return Low
        if ft_probability < 0.3:
            return 'Low'
        #If the probability is greater than 0.3 and less than or equal to 0.6, we return Medium
        elif ft_probability >= 0.3 and ft_probability <= 0.6:
            return 'Medium'
        #If the probability is greater than 0.6, we return High
        else:
            return 'High'


    def __get_clustering_data(self) -> pd.DataFrame:
        """
        Function that obtains the clustering data of each user and returns it in a DataFrame
        The data includes all 15 engineered features we created to perform K-Means clustering

        Returns:
            grouped_users (pandas.DataFrame): DataFrame containing the clustering data of each user
        """
        #Here we obtain the favourite genre of each user
        favourite_genre = self.__get_users_favorite_genres()
        #Here we obtain the average click duration of each user
        avg_click_duration = self.__get_avg_click_duration()
        #Here we obtain the most active time of day of each user
        most_active_time_day = self.__get_most_active_time_of_day()
        #Here we obtain the movie preference of each user (old or new)
        movie_preference = self.__get_movie_preference()
        #Here we obtain the time spent per day of each user
        time_spent_per_day = self.__get_time_spent_per_day()
        #Here we obtain the most active time of week of each user
        most_active_time_week = self.__get_most_active_time_of_week()
        #Here we obtain the most active season of each user
        most_active_season = self.__get_most_active_season()
        #Here we obtain the genre diversity of each user
        genre_diversity = self.__get_genre_diversity()
        #Here we obtain the probability of a user following through with a movie
        ft_probability = self.__get_ft_probability()
        #Here we obtain the most frequent click length of each user
        most_common_click_length = self.__get_most_frequent_click_length()
        #Here we obtain the average click duration during peak hours of each user
        avg_peak_hour_click_duration = self.__get_avg_peak_hour_click_duration()
        #Here we obtain the most active daytime of each user
        most_active_daytime = self.__get_most_active_daytime()
        #Here we obtain the total time spent by each user
        total_time_spent = self.__total_time_spent()
        #Here we obtain the consistency ratio of each user
        consistency_ratio = self.__get_consistency_ratio()
        #Here we obtain the average number of clicks per movie of each user
        avg_click_per_movie = self.__get_avg_click_per_movie()

        #Here we join all the DataFrames into one
        grouped_clustering_data = favourite_genre.join(avg_click_duration).join(most_active_time_day).join(movie_preference)\
            .join(time_spent_per_day).join(most_active_time_week).join(most_active_season).join(genre_diversity)\
            .join(ft_probability).join(most_common_click_length).join(avg_peak_hour_click_duration).join(most_active_daytime)\
            .join(total_time_spent).join(consistency_ratio).join(avg_click_per_movie)
        
        #Finally we return the grouped users
        return grouped_clustering_data        

    def __get_users_favorite_genres(self) -> pd.DataFrame:
        """
        Function that obtains the favourite genre of each user and returns it in a DataFrame

        Returns:
            grouped_users (pandas.DataFrame): DataFrame containing the favourite genre of each user
        """
        
        #First we make a copy of the cleaned data in order to not modify it
        cleaned_data = self.cleaned_data.copy()

        #Here we substitute the NaN values in the genre column with the string 'NaN'
        cleaned_data['genres'] = cleaned_data['genres'].fillna('NaN')

        #Here we split the genres column by each individual genre and stack them in order to obtain each genre in a separate row
        #We then reset the index and join the duration and user_id columns since they are useful for the next step
        df_genres_split = cleaned_data['genres'].str.split(', ', expand=True).stack().reset_index(level=1, drop=True)\
            .to_frame('genre').join(cleaned_data[['duration', 'user_id']]).reset_index(drop=True)

        #Now we group the dataset by user_id and genre and sum the duration of the clicks to obtain the genre with the highest click duration
        #i.e. the favourite genre of each user
        grouped_users = df_genres_split.groupby(['user_id', 'genre']).sum()
        #From here we extract the genre with the highest duration for each user
        grouped_users = grouped_users.reset_index().sort_values('duration', ascending=False).drop_duplicates(['user_id']).drop('duration', axis=1)

        #Here we fill the 'NaN' values with np.nan
        grouped_users['genre'] = grouped_users['genre'].replace('NaN', np.nan)
        
        #Finally we rename the column genre to favourite_genre and make the user_id the index
        grouped_users.rename(columns={'genre': 'favourite_genre'}, inplace=True)
        grouped_users.set_index('user_id', inplace=True)

        #Finally we return the grouped users
        return grouped_users
    
    def __get_avg_click_duration(self) -> pd.DataFrame:
        """
        Function that obtains the average click duration of each user and returns it in a DataFrame

        Returns:
            grouped_users (pandas.DataFrame): DataFrame containing the average click duration of each user
        """
        #First we make a copy of the cleaned data in order to not modify it
        cleaned_data = self.cleaned_data.copy()

        #Here we group the data by user_id and calculate the mean of the duration column
        grouped_users = cleaned_data.groupby('user_id')['duration'].mean().to_frame('avg_click_duration')
        
        #Finally we return the grouped users
        return grouped_users
    
    def __get_most_active_time_of_day(self) -> pd.DataFrame:
        """
        Function that obtains the most active time of day of each user and returns it in a DataFrame

        Returns:
            grouped_users (pandas.DataFrame): DataFrame containing the most active time of day of each user
        """
        #First we make a copy of the cleaned data in order to not modify it
        cleaned_data = self.cleaned_data.copy()

        #Here we classify the hour of the day into morning, afternoon or night and add it as a new column
        cleaned_data['time_of_day'] = cleaned_data['datetime'].apply(lambda x: self.__classify_hour(x.hour))

        #Here we group the data by user_id and time_of_day and sum the duration of the clicks to obtain the time of day with the highest click duration
        grouped_users = cleaned_data[['user_id', 'time_of_day', 'duration']].groupby(['user_id', 'time_of_day']).sum()\
            .reset_index().sort_values('duration', ascending=False).drop_duplicates(['user_id']).drop('duration', axis=1)
        
        #Finally we rename the column time_of_day to most_active_time_day and make the user_id the index
        grouped_users.rename(columns={'time_of_day': 'most_active_time_day'}, inplace=True)
        grouped_users.set_index('user_id', inplace=True)

        #Finally we return the grouped users
        return grouped_users
    
    def __get_movie_preference(self) -> pd.DataFrame:
        """
        Function that obtains the movie preference of each user (old or new) and returns it in a DataFrame

        Returns:
            grouped_users (pandas.DataFrame): DataFrame containing the movie preference of each user
        """
        #First we make a copy of the cleaned data in order to not modify it
        cleaned_data = self.cleaned_data.copy()

        #Here we classify the year of the movie into old or new and add it as a new column
        cleaned_data['is_old_movie'] = cleaned_data['release_date'].apply(lambda x: self.__classify_year(x.year))

        #Here we group the data by user_id and is_old_movie and sum the duration of the clicks to obtain the movie preference with the highest click duration
        grouped_users = cleaned_data[['user_id','is_old_movie', 'duration']].groupby(['user_id','is_old_movie']).sum().reset_index()\
            .sort_values('duration', ascending=False).drop_duplicates(['user_id']).drop('duration', axis=1).set_index('user_id')
        
        #Here we rename the column is_old_movie to is_old_movie_lover
        grouped_users.rename(columns={'is_old_movie': 'is_old_movie_lover'}, inplace=True)
    
        #Finally we return the grouped users
        return grouped_users
    
    def __get_time_spent_per_day(self) -> pd.DataFrame:
        """
        Function that obtains the time spent per day of each user and returns it in a DataFrame

        Returns:
            grouped_users (pandas.DataFrame): DataFrame containing the time spent per day of each user
        """
        #First we make a copy of the cleaned data in order to not modify it
        cleaned_data = self.cleaned_data.copy()

        #Now we make a new column that is only the date of the datetime column	
        cleaned_data['date'] = cleaned_data['datetime'].dt.date

        #Now we group by user_id and date and obtain the average duration spent by day for each user
        grouped_users = cleaned_data.groupby(['user_id', 'date'])['duration'].sum().groupby('user_id').mean().to_frame('duration')
        
        #Here we rename the column duration to avg_duration_spent_by_day
        grouped_users.rename(columns={'duration': 'avg_time_per_day'}, inplace=True)
        
        #Finally we return the grouped users
        return grouped_users
    
    def __get_most_active_time_of_week(self) -> pd.DataFrame:
        """
        Function that obtains the most active time of week of each user and returns it in a DataFrame

        Returns:
            grouped_users (pandas.DataFrame): DataFrame containing the most active time of week of each user
        """

        #First we make a copy of the cleaned data in order to not modify it
        cleaned_data = self.cleaned_data.copy()

        #Here we create a new column where we say if the movie was clicked on a weekend or not
        #We do this by checking if the weekday is greater than or equal to 4 (Friday)
        cleaned_data['weekend'] = cleaned_data['datetime'].apply(lambda x: "Weekend" if x.weekday() >= 4 else "Weekday")

        #Here we group the data by user_id and weekend and sum the duration of the clicks to obtain the time of the week with the highest total click duration
        grouped_users = cleaned_data[['user_id','weekend', 'duration']].groupby(['user_id','weekend']).sum().reset_index()\
            .sort_values('duration', ascending=False).drop_duplicates(['user_id']).drop('duration', axis=1).set_index('user_id')
        
        #Here we rename the column weekend to weekend_watcher
        grouped_users.rename(columns={'weekend': 'most_active_time_week'}, inplace=True)

        #Finally we return the grouped users
        return grouped_users
    
    def __get_most_active_season(self) -> pd.DataFrame:
        """
        Function that obtains the most active season of each user and returns it in a DataFrame

        Returns:
            grouped_users (pandas.DataFrame): DataFrame containing the most active season of each user
        """
        #First we make a copy of the cleaned data in order to not modify it
        cleaned_data = self.cleaned_data.copy()

        #Here we create a new column where we say the season of the year
        cleaned_data['season'] = cleaned_data['datetime'].apply(lambda x: self.__classify_season(x.month))

        #Here we group the data by user_id and season and sum the duration of the clicks to obtain the season with the highest total click duration
        grouped_users = cleaned_data[['user_id','season', 'duration']].groupby(['user_id','season']).sum().reset_index()\
            .sort_values('duration', ascending=False).drop_duplicates(['user_id']).drop('duration', axis=1).set_index('user_id')
        
        #Here we rename the column season to most_active_season
        grouped_users.rename(columns={'season': 'most_active_season'}, inplace=True)

        #Finally we return the grouped users
        return grouped_users
    
    def __get_genre_diversity(self) -> pd.DataFrame:
        """
        Function that obtains the genre diversity of each user and returns it in a DataFrame

        Returns:
            grouped_users (pandas.DataFrame): DataFrame containing the genre diversity of each user
        """

        #First we make a copy of the cleaned data in order to not modify it
        cleaned_data = self.cleaned_data.copy()

        #Here we obtain a list of the unique user_ids
        user_ids = list(cleaned_data['user_id'].unique())

        #Here we obtain the genres by user
        genres_by_user = self.__get_user_genres()

        #Here we obtain the total number of different genres in the dataset
        total_diff_genres = len(cleaned_data[~ cleaned_data.genres.isna()].genres.apply(lambda x: x.split(', ')).explode().unique())

        #Here we initialize a list to store the genre diversity of each user
        genre_diversity = []

        #Here we iterate over the user_ids
        for user_id in user_ids:
            #Here we obtain the number of different genres that the user clicked on divided by the total number of different genres in the dataset
            diff_genres = len(genres_by_user[user_id])/total_diff_genres if len(genres_by_user[user_id]) != 0 else np.nan
            #Here we append the genre diversity of the user
            genre_diversity.append(diff_genres)

        #Here we create a DataFrame with the user_id and genre_diversity columns
        grouped_users = pd.DataFrame({'user_id': user_ids, 'genre_diversity': genre_diversity}).set_index('user_id')

        #Finally we return the grouped users
        return grouped_users
    
    def __get_ft_probability(self) -> pd.DataFrame:
        """
        Function that obtains the probability of a user following through with a movie and returns it in a DataFrame

        Returns:
            grouped_users (pandas.DataFrame): DataFrame containing the probability of a user following through with a movie
        """

        #First we make a copy of the cleaned data in order to not modify it
        cleaned_data = self.cleaned_data.copy()

        #Here we create a new column that states if the duration of the click is greater than or equal to 300 seconds (5 minutes)
        #If it is, we say that the user followed through with the movie, if not, we say that the user did not follow through with the movie
        cleaned_data['greater_than_threshold'] = cleaned_data['duration'].apply(lambda x: "True" if x >= self.FOLLOW_THROUGH_THRESHOLD else np.nan)

        #Here we group the data by user_id and greater_than_threshold and count the number of clicks to obtain the number of times the user followed through with the movie
        grouped_users = cleaned_data.groupby('user_id')['greater_than_threshold'].count().to_frame().rename(columns={'greater_than_threshold': 'count'}).sort_index()

        #Here we add a new column that counts the total number of clicks of each user
        grouped_users['total_count'] = cleaned_data.groupby('user_id')['title'].count().sort_index().to_list()

        #Here we add a new column that calculates the probability of the user following through with the movie
        #This is done by dividing the number of times the user followed through with the movie by the total number of clicks of the user
        grouped_users['ft_probability'] = grouped_users['count'] / grouped_users['total_count']

        #Here we drop the count and total_count columns
        grouped_users.drop(['count', 'total_count'], axis=1, inplace=True)

        #Finally we return the grouped users
        return grouped_users
    
    def __get_most_frequent_click_length(self) -> pd.DataFrame:
        """
        Function that obtains the most frequent click length of each user and returns it in a DataFrame

        Returns:
            grouped_users (pandas.DataFrame): DataFrame containing the most frequent click length of each user
        """

        #First we make a copy of the cleaned data in order to not modify it
        cleaned_data = self.cleaned_data.copy()

        #Here we classify the duration of the click into short, medium or long and add it as a new column
        cleaned_data['click_length'] = cleaned_data['duration'].apply(lambda x: self.__classify_click_lengths(x))

        #Here we group the data by user_id and click_length and count the number of clicks to obtain the click length with the highest total click duration
        grouped_users = cleaned_data.groupby('user_id')['click_length'].value_counts().reset_index(name='count')\
            .sort_values(['user_id','count'], ascending=False).drop_duplicates('user_id').drop('count', axis=1).reset_index(drop=True)
        
        #Here we rename the column click_length to most_common_click_length
        grouped_users.rename(columns={'click_length':'most_common_click_length'}, inplace=True)
        #Here we make the user_id the index
        grouped_users.set_index('user_id', inplace=True)

        #Finally we return the grouped users
        return grouped_users
    
    def __get_avg_peak_hour_click_duration(self) -> pd.DataFrame:
        """
        Function that obtains the average click duration during peak hours of each user and returns it in a DataFrame

        Returns:
            grouped_users (pandas.DataFrame): DataFrame containing the average click duration during peak hours of each user
        """

        #First we make a copy of the cleaned data in order to not modify it
        cleaned_data = self.cleaned_data.copy()

        #Here we classify the hour of the day into peak or non-peak and add it as a new column
        cleaned_data['is_peak_hour'] = cleaned_data['datetime'].apply(lambda x: self.__classify_peak_hours(x.hour))

        #Here we group the data by user_id and is_peak_hour and calculate the mean of the duration column to obtain the average click duration during peak hours
        grouped_users = cleaned_data[cleaned_data['is_peak_hour'] == 1].groupby('user_id')['duration'].mean().to_frame()\
            .rename(columns={'duration': 'mean_duration_peak_hour'})
        
        #Finally we return the grouped users
        return grouped_users
    
    def __get_most_active_daytime(self) -> pd.DataFrame:
        """
        Function that obtains the most active daytime of each user and returns it in a DataFrame

        Returns:
            grouped_users (pandas.DataFrame): DataFrame containing the most active daytime of each user
        """
        #First we make a copy of the cleaned data in order to not modify it
        cleaned_data = self.cleaned_data.copy()

        #Here we classify the hour of the day into morning, afternoon or night and add it as a new column
        cleaned_data['time_of_day'] = cleaned_data['datetime'].apply(lambda x: self.__classify_daytime(x.hour))

        #Here we group the data by user_id and time_of_day and sum the duration of the clicks to obtain the daytime with the highest click duration
        grouped_users = cleaned_data[['user_id', 'time_of_day', 'duration']].groupby(['user_id', 'time_of_day']).sum()\
            .reset_index().sort_values('duration', ascending=False).drop_duplicates(['user_id']).drop('duration', axis=1)
        
        #Finally we rename the column time_of_day to most_active_time_day and make the user_id the index
        grouped_users.rename(columns={'time_of_day': 'most_active_daytime'}, inplace=True)
        grouped_users.set_index('user_id', inplace=True)

        #Finally we return the grouped users
        return grouped_users
    
    def __total_time_spent(self) -> pd.DataFrame:
        """
        Function that obtains the total time spent by each user and returns it in a DataFrame

        Returns:
            grouped_users (pandas.DataFrame): DataFrame containing the total time spent by each user
        """
        #First we make a copy of the cleaned data in order to not modify it
        cleaned_data = self.cleaned_data.copy()

        #Here we group the data by user_id and sum the duration of the clicks to obtain the total time spent by each user
        grouped_users = cleaned_data.groupby('user_id')['duration'].sum().to_frame("total_time_spent")

        #Finally we return the grouped users
        return grouped_users
    
    def __get_consistency_ratio(self) -> pd.DataFrame:
        """
        Function that obtains the consistency ratio of each user and returns it in a DataFrame

        Returns:
            grouped_users (pandas.DataFrame): DataFrame containing the consistency ratio of each user
        """

        #First we make a copy of the cleaned data in order to not modify it
        cleaned_data = self.cleaned_data.copy()

        #Here we add a new column that is the datetime converted to date in format YYYY-MM-DD
        cleaned_data['date'] =cleaned_data['datetime'].dt.date

        #Here we group the dataframe by user_id and obtain the number of different days where the user clicked on a movie
        grouped_users = cleaned_data.groupby('user_id')['date'].nunique().to_frame("consistency_ratio")

        #Now we obtain the ratio of the number of different days where the user clicked on a movie divided by the total number of days
        grouped_users['consistency_ratio'] = grouped_users['consistency_ratio']/cleaned_data['date'].nunique()

        #Finally we return the grouped users
        return grouped_users
    
    def __get_avg_click_per_movie(self) -> pd.DataFrame:
        """
        Function that obtains the average number of clicks per movie of each user and returns it in a DataFrame

        Returns:
            grouped_users (pandas.DataFrame): DataFrame containing the average number of clicks per movie of each user
        """
        #First we make a copy of the cleaned data in order to not modify it
        cleaned_data = self.cleaned_data.copy()

        #Here we group the data by user_id and movie_id and count the number of clicks to obtain the average number of clicks per movie
        grouped_users = cleaned_data.groupby('user_id')['movie_id'].value_counts().groupby('user_id').mean().to_frame("avg_click_per_movie")

        #Finally we return the grouped users
        return grouped_users
    
    def __classify_peak_hours(self, hour: int) -> str:
        """
        Function that classifies the hour of the day into peak or non-peak

        Args:
            hour (int): Hour of the day

        Returns:
            peak_hour (str): Peak hour of the day
        """
        #If the hour is between 18 and 23, we return 1 (Peak Hour)
        if hour >= 18 and hour <= 23:
            return 1
        #If the hour is not between 18 and 23, we return 0 (Non-Peak Hour)
        else:
            return 0
    
    def __classify_click_lengths(self, duration: float) -> str:
        """
        Function that classifies the duration of a click into short, medium or long

        Args:
            duration (float): Duration of the click

        Returns:
            click_length (str): Length of the click
        """
        #If the duration is less than or equal to 600 seconds (10 minutes), we return Short
        if duration <= 600:
            return 'Short'
        #If the duration is greater than 600 seconds (10 minutes) and less than or equal to 3600 seconds (1 hour), we return Medium
        elif duration > 600 and duration <= 3600:
            return 'Medium'
        #If the duration is greater than 3600 seconds (1 hour), we return Long
        else: 
            return 'Long'
    
    def __classify_season(self, month: int) -> str:
        """
        Function that classifies the month of the year into seasons

        Args:
            month (int): Month of the year

        Returns:
            season (str): Season of the year
        """
        #If the month is between 12 and 2, we return Winter
        if month == 12 or month == 1 or month == 2:
            return 'Winter'
        #If the month is between 3 and 5, we return Spring
        elif month >= 3 and month <= 5:
            return 'Spring'
        #If the month is between 6 and 8, we return Summer
        elif month >= 6 and month <= 8:
            return 'Summer'
        #If the month is between 9 and 11, we return Autumn
        else:
            return 'Autumn'
        
    
    def __classify_year(self, year: int) -> str:
        """
        Function that classifies if a year is an old movie or not

        Args:
            year (int): Year of the movie

        Returns:
            year (str): Year of the movie
        """
        #Here we check if the year is NaN, if it is, we return NaN
        if year != year:
            return np.nan
        else:
            #Here we check if the year is less than or equal to 2010, if it is, we return True
            if year <= 2010:
                return "True"
            #Here we check if the year is greater than 2010, if it is, we return False
            else:
                return "False"

    def __classify_hour(self, hour: int) -> str:
        """
        Function that classifies the hour of the day into morning, afternoon or night

        Args:
            hour (int): Hour of the day

        Returns:
            time_of_day (str): Time of the day
        """
        #If the hour is between 4 and 12, we return Morning
        if hour >= 4 and hour < 12:
            return 'Morning'
        #If the hour is between 12 and 20, we return Afternoon
        elif hour >= 12 and hour < 20:
            return 'Afternoon'
        #If the hour is between 20 and 4, we return Night
        else:
            return 'Night'
        
    def __classify_daytime(self, hour: int) -> str:
        """
        Function that classifies the hour of the day into daytime or nighttime

        Args:
            hour (int): Hour of the day

        Returns:
            daytime (str): Daytime of the day
        """

        #If the hour is between 6 and 18, we return Daytime
        if hour >= 6 and hour < 18:
            return 'Daytime'
        #If the hour is between 18 and 6, we return Nighttime
        else:
            return 'Nighttime'

    def __clean_data(self, raw_dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Function that cleans the raw anonymised movie watch history of opted-in Netflix users (UK)

        Args:
            raw_dataset (pandas.DataFrame): Raw anonymised movie watch history of opted-in Netflix users (UK)

        Returns:
            cleaned dataset (pandas.DataFrame): Cleaned dataset
        """
        #Here we remove rows where the duration of a click is not greater than 0
        cleaned_dataset = raw_dataset[raw_dataset.duration >= 0].reset_index(drop=True)

        #Here we convert the release date and datetime columns to datetime objects
        #We coerce the errors to NaT (Not a Time) so that we can handle them later
        cleaned_dataset['release_date'] = pd.to_datetime(cleaned_dataset['release_date'], errors='coerce')
        cleaned_dataset['datetime'] = pd.to_datetime(cleaned_dataset['datetime'], errors='coerce')
        #Here we convert the duration column to a float64
        cleaned_dataset['duration'] = cleaned_dataset['duration'].astype('float64')

        #Here we replace the 'NOT AVAILABLE' values with NaN in the genres column
        cleaned_dataset['genres'] = cleaned_dataset['genres'].replace('NOT AVAILABLE', np.nan)

        #Finally, we noticed that there are some movie titles that are the same but have different genres
        #We will then substitute the genres of movies with the same title with the most common genre (if it exists, if not, we will leave it as NaN)
        cleaned_dataset['genres'] = cleaned_dataset.groupby('title')['genres'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))
        #We do the same for the release date
        cleaned_dataset['release_date'] = cleaned_dataset.groupby('title')['release_date'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))

        #Finally, we return the cleaned dataset
        return cleaned_dataset
    
    def __get_lsh_user_data(self) -> pd.DataFrame:
        """
        Function that obtains the title and genre of the maximum top 10 movies that each user clicked on regarding the number of clicks.
        This data will later be used to perform Locality Sensitive Hashing (LSH) to obtain the most similar users.

        Args:
            None

        Returns:
            user_data (pandas.DataFrame): DataFrame containing user data to perform LSH.
        """
        #Here we group the data by user_id, title and genres and count the number of clicks
        #We then sort the values by user_id and clicks in descending order
        #We then group the data by user_id and select the top 10 movies that the user clicked on
        user_data = self.cleaned_data.groupby(['user_id', 'title', 'genres']).size().reset_index(name='clicks')\
            .sort_values(by=['user_id', 'clicks'], ascending=False).groupby('user_id').head(10).reset_index(drop=True)
        
        #Finally, we return the user data sorted by user_id
        return user_data.sort_values(by='user_id').reset_index(drop=True)
    
    def __get_user_genres(self) -> defaultdict:
        """
        Function that obtains the unique genres in which a user clicked on.
        This data will later be used to create the characteristic matrix.

        Returns:
            genres_by_user (defaultdict): Dictionary containing the genres in which a user clicked on.
        """
        #Here we initialize a defaultdict to store the genres in which a user clicked on
        genres_by_user = defaultdict(list)

        #Here we iterate over the rows of the user data
        for _, row in self.cleaned_data.iterrows():
            #Here we append the genres to the list of genres of the corresponding user
            #We do this by checking if the genres column value is not NaN
            if row['genres'] == row['genres']:
                #Here we split the genres by comma and append them to the list of genres of the corresponding user
                genres_by_user[row['user_id']].extend(row['genres'].split(', '))
                #Here we remove duplicates from the list of genres of the corresponding user
                genres_by_user[row['user_id']] = list(set(genres_by_user[row['user_id']]))  

            else:
                #If the genres column value is NaN, we append an empty list to the list of genres of the corresponding user
                genres_by_user[row['user_id']].extend([])

        #Finally, we return the genres by user
        return genres_by_user
