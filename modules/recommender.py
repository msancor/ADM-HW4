import heapq
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
pd.options.mode.chained_assignment = None
from collections import defaultdict, Counter


class Recommender():
    """
    This class implements a recommender system based on Locality Sensitive Hashing (LSH). The recommender system is used to recommend movies to users.
    The recommender system uses the title and genre of the maximum top 10 movies that Netflix users clicked on regarding the number of clicks to recommend movies to users.
    The recommender system works as follows:

    1) We first create a characteristic matrix. This is a matrix where each column represents a unique user and each row represents a unique genre.
       The value of each cell is 1 if the user clicked on a movie with that genre, 0 otherwise.

    2) We then create a signature matrix. This is a matrix where each column represents a unique user and each row represents the minhash value of a hash function.
       The signature of a given user approximates the Jaccard similarity between the user and the other users.

    3) We then perform banding on the signature matrix. This is, we divide the signature matrix into bands of n rows each and then hash each user to a given bucket depending on their signature in each band.
       This ensures that users that are similar are hashed to the same bucket without being exactly similar in the whole column of the signature matrix.

    After this, we can recommend movies to a user by following the next steps:

    1) We take as an input the user id of a user.

    2) We obtain the similar users to the user. This is, users that share at least one bucket with the user.

    3) We obtain the top n similar users (i.e. users that share the most buckets with the user)

    4) We obtain the recommendations for the user by obtaining the movies the top n similar users have in common (if any) and recommend those movies based on the total number of clicks by these users. 
       If not, we try to propose the most clicked movies by the most similar user first, followed by the other user.

    Methods:
        get_recommendations: Function that obtains the top 5 recommendations for a user with the given user id.

    Attributes:
        user_data (pandas.DataFrame): DataFrame containing the user data.

        vocabulary (dict): Dictionary containing the index of each genre.

        users (dict): Dictionary containing the index of each user.

        genres_by_user (defaultdict): Dictionary containing the genres in which a user clicked on.

        characteristic_matrix (numpy.ndarray): Characteristic matrix.

        signature_matrix (numpy.ndarray): Signature matrix.

        buckets (defaultdict): Dictionary containing the buckets to save each user.
    """
    #Here we define the constants of the class
    #We define the number of recommendations we want to obtain for a user
    NUMBER_OF_RECOMMENDATIONS = 5
    #We define the closest prime number to the maximum shingle id. This was obtained by a quick google search
    CLOSEST_PRIME_NUMBER = 29
    #We define the number of top similar users we want to obtain for a user
    TOP_N_SIMILAR_USERS = 2
    #We define the number of hash functions we want to use for minhashing
    NUM_HASHES = 100
    #We define the number of bands we want to use for banding
    BANDS = 20
    #We define the number of rows we want to use for banding
    ROWS = 5
     
    def __init__(self, user_data: pd.DataFrame):
        """
        Class constructor.

        Args:
            user_data (pandas.DataFrame): DataFrame containing the user data.
        """
        #Here we define the user data
        self.user_data = user_data
        #Here we define the vocabulary of the genres. This is, the unique genres in the dataset
        self.vocabulary = self.__get_genres_index()
        #Here we define the users in the dataset. This is, the unique users in the dataset
        self.users = self.__get_user_index()
        #Here we define the genres in which each user clicked on. This are the unique genres in which a user clicked on
        self.genres_by_user = self.__get_user_genres()
        #Here we create the characteristic matrix. This is a matrix where each column represents a user and each row represents a genre.
        self.characteristic_matrix = self.__get_characteristic_matrix()
        #Here we create the signature matrix. This is a matrix where each column represents a user and each row represents the minhash value of a hash function.
        self.signature_matrix = self.__get_signature_matrix()
        #Here we create the buckets and hash each user to a bucket depending on their signature in a given band of the signature matrix.
        self.buckets = self.__get_buckets()

    def get_recommendations(self, user_id: str = "", random: bool = False) -> pd.DataFrame:
        """
        Function that obtains the top 5 recommendations for a user with the given user id.
        It does that via Locality Sensitive Hashing (LSH):
        
        1) We obtain the similar users to the user (i.e. users that share at least one bucket with the user)
        
        2) We obtain the top n similar users (i.e. users that share the most buckets with the user)
        
        3) We obtain the recommendations for the user by obtaining the movies the top n similar users have in common (if any) 
        and recommend those movies based on the total number of clicks by these users. 
        If not, we try to propose the most clicked movies by the most similar user first, followed by the other user.

        Args:
            user_id (str): User id.
            random (bool): Boolean indicating whether to get recommendations for a random user or not.

        Returns:
            recommendations (pandas.DataFrame): DataFrame containing the top 5 recommendations for a user.
        """
        #Here we get a random user id if random=True
        user_id = np.random.choice(list(self.users.keys())) if random else user_id
        
        #We have to assert that the user_id is in the dataset
        assert user_id in self.users.keys(), 'The user id is not in the dataset'
        
        #First we obtain the similar users to the user (i.e. users that share at least one bucket with the user)
        similar_users = self.__get_similar_users(user_id)
        #Now we obtain the top n similar users (i.e. users that share the most buckets with the user)
        top_n_similar_users = self.__get_top_n_similar_users(user_id, similar_users)
        #Now we obtain the recommendations for the user
        recommendations = self.__get_recommendations(user_id, top_n_similar_users)

        #Finally, we return the recommendations
        return recommendations
    
    def __get_similar_users(self, user_id: str) -> List[str]:
        """
        Function that obtains the similar users to a user (i.e. users that share at least one bucket with the user).

        Args:
            user_id (str): User id.

        Returns:
            similar_users (list): List containing the similar users to a user.
        """
        #First we have to obtain the one hot vector for the genres of the user
        one_hot_vector = self.__get_one_hot_vector(self.genres_by_user[user_id])

        #Now we have to obtain the signature vector for the user
        #This vector is obtained by performing minhashing on the one hot vector
        signature_vector = self.__get_signature_vector(one_hot_vector, self.__get_hash_functions())
        
        #Now we have to obtain the similar users to the user
        #First we initialize a list to store the similar users
        similar_users = []
        #Then we perform banding on the signature vector. This is, we divide the signature vector into bands of n rows each
        #We then find the similar users by hashing the user to a bucket depending on their signature in each band and finding the users that share at least one bucket with the user
        for band in range(self.BANDS):
            #Here we obtain the bucket id
            bucket_id = int(''.join(map(str, signature_vector[band*self.ROWS:(band+1)*self.ROWS])))
            #Here we extend the similar users with the users in the bucket
            similar_users.extend(self.buckets[band][bucket_id])
        
        #Finally, we return the similar users, this are the users that share at least one bucket with the user
        return similar_users
    
    def __get_one_hot_vector(self, user_genres: List[str]) -> np.ndarray:
        """
        Function that creates a one-hot vector for the genres of a user.
        This is a vector where each element represents a genre and the value of the element is:
        
        - 1 if the user clicked on a movie with that genre 
        
        - 0 otherwise.

        Args:
            user_genres (list): List containing the genres of a user.

        Returns:
            one_hot_vector (numpy.ndarray): One-hot vector containing the genres of a user.
        """
        #Here we initialize a list to store the one-hot vector
        one_hot_vector = np.zeros(len(self.vocabulary))
        #Here we iterate over the genres of a user
        for genre in user_genres:
            #Here we get the index of the genre
            index = self.vocabulary[genre]
            #Here we set the value of the one-hot vector to 1
            one_hot_vector[index] = 1

        #Finally, we return the one-hot vector
        return one_hot_vector
    
    def __get_signature_vector(self, user_one_hot: np.ndarray, hash_functions: List[callable]) -> np.ndarray:
        """
        Function that creates a signature vector from the characteristic matrix.
        The signature vector is created by performing minhashing on the one-hot vector of a user.
        Minhashing is performed by the following steps:

        1) We first get the indexes of the rows where the user clicked on a movie.
        2) We then obtain the value of each hash function for each row where the user clicked on a movie.
        3) We then update the value of the signature vector if the value of the hash function is smaller than the value of the signature vector.

        Args:
            user_one_hot (numpy.ndarray): One-hot vector containing the genres of a user.
            hash_functions (list): List of hash functions (callables).

        Returns:
            signature_vector (numpy.ndarray): Signature vector.
        """
        #Here we initialize a numpy array to store the signature vector
        signature_vector = np.full(len(hash_functions), np.inf)

        #We first get the indexes of the rows where the user clicked on a movie
        user_indexes = np.where(user_one_hot == 1)[0]

        #Now we perform minhashing
        #We only need to iterate over the rows where the user clicked on a movie
        for row in user_indexes:
            #Here we iterate over the hash functions
            for i, hash_function in enumerate(hash_functions):
                #Here we obtain the value of the hash function
                hash_value = hash_function(row)
                #Here we check if the value of the hash function is smaller than the value of the signature vector
                if hash_value < signature_vector[i]:
                    #Here we update the value of the signature vector
                    signature_vector[i] = hash_value
        
        #Finally, we return the signature vector as an integer vector
        return signature_vector.astype(int)
    
    def __get_hash_functions(self) -> List[callable]:
        """
        Function that returns a list of random hash functions. The functions are of the form:
        h(x) = (a*x + b) % p, where a and b are random integers and p is the closest prime number to the maximum shingle id.
        These functions are used to perform minhashing and obtain the signature vector of a user.

        Returns:
            hash_functions (list): List of hash functions (callables).
        """

        #First, we set a seed to ensure reproducibility
        np.random.seed(1)

        #Here we generate random integers between 0 and the maximum shingle id
        a = np.random.randint(0, max(self.vocabulary.values()), size=self.NUM_HASHES)
        b = np.random.randint(0, max(self.vocabulary.values()), size=self.NUM_HASHES)

        #Here we initialize a list to store the hash functions
        hash_functions = []
        #Here we iterate over the number of hash functions
        for i in range(self.NUM_HASHES):
            #Here we generate a random hash function
            hash_function = self.__generate_hash_function(a[i], b[i])
            #Here we append the hash function to the list of hash functions
            hash_functions.append(hash_function)
        
        #Finally, we return the list of hash functions
        return hash_functions
    
    def __generate_hash_function(self, a: int, b: int) -> callable:
        """
        Function that generates a hash function. The function is of the form:
        h(x) = (a*x + b) % p, where a and b are random integers and p is the closest prime number to the maximum shingle id.

        Args:
            a (int): Random integer.
            b (int): Random integer.

        Returns:
            hash_function (callable): Hash function.
        """
        #Finally, we return the hash function
        return lambda x: (a*x + b) % self.CLOSEST_PRIME_NUMBER

    def __get_top_n_similar_users(self, user_id: str, similar_user_list: List[str]) -> List[str]:
        """
        Function that obtains the top n similar users to a user (i.e. users that share the most buckets with the user)

        Args:
            user_id (str): The user we want to obtain similar users from.
            similar_user_list (list): List with all the similar users (i.e. the users that shared at least one bucket with the user)

        Returns:
            top_n_similar_users (list): List containing the user_id of the top n similar users.
        """

        #First we have to ensure we dont have the user in the list of similar users
        user_list = [user for user in similar_user_list if user != user_id]
        #Now we count the number of times each user appears in the list of similar users
        user_list = Counter(user_list)
        #Now we can convert the Counter to a list of tuples
        user_list = [(count, user) for user, count in user_list.items()]

        #The idea is to use a heap to store the resulting tuples of (count, user_id)
        #In this way we can easily obtain the top n similar users
        #We initialize a heap
        heap = []
        #Here we iterate over the similar users and add them to the heap
        #We multiply the count by -1 since we want to build a maxheap
        for count, user in user_list:
            heapq.heappush(heap, ((-1)*count, user))

        #Now we can obtain the top n similar users
        #First we initialize the list to save the top n similar users
        top_n_similar_users = []
        #Here we iterate n times
        for _ in range(self.TOP_N_SIMILAR_USERS):
            #Here we pop the tuple from the heap
            _, user = heapq.heappop(heap)
            #Here we append the user tuple to the list of top n similar users
            top_n_similar_users.append(user)

        #Finally, we return the top n similar users
        return top_n_similar_users
    
    def __get_recommendations(self, user_id: str, top_n_similar_users: List[str]) -> pd.DataFrame:
        """
        Function that obtains at maximum n recommendations for a user.
        To recommend at most five movies given a user_id, we use the following procedure:

        1) We obtain the movies the top n similar users have in common (if any) and recommend those movies based on the total number of clicks by these users.
        2) If there are no more common movies, we try to propose the most clicked movies by the most similar user first, followed by the other user.

        Args:
            user_id (str): User id of the user we want to obtain recommendations from.
            top_n_similar_users (list): List containing the top n similar users for a given user.

        Returns:
            recommendations (pandas.DataFrame): DataFrame containing the top n recommendations for a user.
        """

        #Here we obtain the dataframe with all the potential movies we can recommend to the user
        potential_movies = self.user_data[self.user_data['user_id'].isin(top_n_similar_users)]
        #We can also add a new column to the dataframe containing the Jaccard similarity between the user and the similar user
        potential_movies['jaccard_similarity'] = potential_movies['user_id'].apply(lambda x: self.__jaccard_similarity(x, user_id))

        #Now we can obtain our top recommendations for the user. This is a long process, so we will explain it step by step.
        #1) First we group the potential movies by title since we want to see if there are common movies between the users
        #2) We then aggregate the data by summing the clicks between the same movies, counting the number of users that clicked on the movie (i.e. if the movie is common between the users) 
        # and taking the maximum Jaccard similarity between the users if they have the movie in common
        #3) We then sort the values by the number of users that clicked on the movie, the Jaccard similarity and the number of clicks in descending order
        # By doing this we ensure that the movies that are common between the users are recommended first, followed by the movies by the most similar user and finally the movies by the other user
        #always sorted by the total number of clicks
        #4) Finally, we obtain the top 5 movies and return the title of the movies in a DataFrame
        recommendations = potential_movies.groupby('title').agg({'clicks': 'sum', 'user_id': 'count', 'jaccard_similarity': 'max'})\
            .sort_values(by=['user_id', 'jaccard_similarity', 'clicks'], ascending=False).head(self.NUMBER_OF_RECOMMENDATIONS).index.to_frame().reset_index(drop=True)
        
        #Here we add 1 to the index of the recommendations in order to start the index from 1
        recommendations.index += 1

        #Here we rename the column to 'Recommended Movies'
        recommendations.columns = ['Recommended Movies for User {}'.format(user_id)]
        
        #Finally, we return the recommendations
        return recommendations
    
    def __jaccard_similarity(self, user1: str, user2: str) -> float:
        """
        Function that computes the Jaccard similarity between two users.

        Args:
            user1 (str): User id of the first user.
            user2 (str): User id of the second user.

        Returns:
            jaccard_similarity (float): Jaccard similarity between the two users.
        """
        #Here we obtain the sets of genres for each user
        set_1 = set(self.genres_by_user[user1])
        set_2 = set(self.genres_by_user[user2])

        #Here we compute the intersection of the two sets
        intersection = set_1.intersection(set_2)
        #Here we compute the union of the two sets
        union = set_1.union(set_2)

        #Here we compute the Jaccard similarity
        jaccard_similarity = len(intersection) / len(union)
    
        #Finally, we return the Jaccard similarity
        return jaccard_similarity
            

    def __get_unique_genres(self) -> List[str]:
        """
        Function that obtains the unique genres in the dataset.
        These genres will be used to create the characteristic matrix based on shingles.

        Returns:
            unique_genres (list): List containing the unique genres in the dataset.
        """
        #Here we obtain the unique genres in the dataset
        #We do this by converting the genres column to a list of lists where each list contains the genres of a movie
        unique_genres = self.user_data.genres.apply(lambda x: x.lower().split(', ')).to_list()
        
        #We then flatten the list of lists and remove duplicates
        unique_genres = sorted(list(set([item for sublist in unique_genres for item in sublist])))
        
        #Finally, we return the unique genres
        return unique_genres
    
    def __get_genres_index(self) -> Dict[str, int]:
        """
        Function that obtains the index of each genre.

        Returns:
            genres_index (dict): Dictionary containing the index of each genre.
        """
        #Here we initialize a dictionary to store the index of each genre
        genres_index = {}

        #Here we get the unique genres
        unique_genres = self.__get_unique_genres()

        #Here we iterate over the unique genres
        for i, genre in enumerate(unique_genres):
            #Here we add the genre to the dictionary
            genres_index[genre] = i
        
        #Finally, we return the genre index
        return genres_index
    
    def __get_user_index(self) -> Dict[str, int]:
        """
        Function that obtains the index of each user.

        Returns:
            user_index (dict): Dictionary containing the index of each user.
        """
        #Here we obtain the unique users in the dataset
        unique_users = sorted(list(self.user_data.user_id.unique()))
        #Here we initialize a dictionary to store the index of each user
        user_index = {}
        #Here we iterate over the unique users
        for i, user in enumerate(unique_users):
            #Here we add the user to the dictionary
            user_index[user] = i
        
        #Finally, we return the user index
        return user_index

    
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
        for _, row in self.user_data.iterrows():
            #Here we append the genres to the list of genres of the corresponding user
            #We append each individual genre in lowercase by separating the genres by a comma and a space
            genres_by_user[row['user_id']].extend([word.lower() for word in row['genres'].split(', ')])
            #Here we remove duplicates from the list of genres of the corresponding user
            genres_by_user[row['user_id']] = list(set(genres_by_user[row['user_id']]))  

        #Finally, we return the genres by user
        return genres_by_user
    
    def __get_characteristic_matrix(self) -> np.ndarray:
        """
        Function that creates a characteristic matrix from the user data.
        The characteristic matrix is a matrix where each column represents a user and each row represents a genre.
        The value of each cell is 1 if the user clicked on a movie with that genre, 0 otherwise.

        Returns:
            characteristic_matrix (numpy.ndarray): Characteristic matrix.
        """
        #First we initialize a list to store the characteristic matrix
        characteristic_matrix = []
        #Here we iterate over the users in the order of the user index
        for user in self.users.keys():
            #Here we get the one hot vector for the genres of a user
            one_hot_vector = self.__get_one_hot_vector(self.genres_by_user[user])
            #Here we append the one hot vector to the characteristic matrix
            characteristic_matrix.append(one_hot_vector)

        #After, we can stack the one hot vectors horizontally to obtain the characteristic matrix
        characteristic_matrix = np.stack(characteristic_matrix, axis=1)

        #Finally, we return the characteristic matrix
        return characteristic_matrix
    
    def __get_signature_matrix(self) -> np.ndarray:
        """
        Function that creates a signature matrix from the characteristic matrix.
        The signature matrix is a matrix where each column represents a user and each row represents the minhash value of a hash function.

        Returns:
            signature_matrix (numpy.ndarray): Signature matrix.
        """
        #Here we initialize a list to store the signature matrix
        signature_matrix = []
        #Here we get the hash functions
        hash_functions = self.__get_hash_functions()
        #Here we iterate over the users
        for user in self.users.keys():
            #Here we get the one hot vector for the genres of a user
            one_hot_vector = self.__get_one_hot_vector(self.genres_by_user[user])
            #Here we get the signature vector for a user
            signature_vector = self.__get_signature_vector(one_hot_vector, hash_functions)
            #Here we append the signature vector to the signature matrix
            signature_matrix.append(signature_vector)
        
        #After, we can stack the signature vectors horizontally to obtain the signature matrix
        signature_matrix = np.stack(signature_matrix, axis=1)

        #Finally, we return the signature matrix
        return signature_matrix
    
    def __get_buckets(self) -> List[defaultdict]:
        """
        Function that obtains the buckets and hashes each user to a bucket depending on the signature matrix.
        This ensures that users that are similar are hashed to the same bucket.
        
        Returns:
            buckets (list): List containing the buckets to save each user.
        """
        #Here we initialize a list to store the buckets
        buckets = []
        
        #Here we will perform banding on the signature matrix. This is, we will divide the signature matrix into bands of n rows each
        #And then hash each user to a given bucket depending on their signature in each band
        #Here we iterate over the bands of the signature matrix
        for band in range(self.BANDS):
            #Here we create a defaultdict to store the buckets for each band
            band_buckets = defaultdict(list)
            #Here we iterate over the columns of the signature matrix
            for user, col in self.users.items():
                #We have to make sure that signatures that are the same but in different bands are hashed to different buckets
                #This is why we will create a dictionary for each band and then store all the dictionaries in a list
                #Here we convert the signature to an integer and then hash it to a bucket for the given band
                signature = int(''.join(map(str, self.signature_matrix[band*self.ROWS:(band+1)*self.ROWS, col])))
                #Thus we can append the user_id to the bucket
                band_buckets[signature].append(user)
            
            #Finally, we append the band buckets to the list of buckets
            buckets.append(band_buckets)
        
        #Finally, we return the buckets
        return buckets


    