#Here we import the necessary libraries
from pyspark import SparkConf, SparkContext, RDD
from typing import List, Tuple
import pandas as pd
import numpy as np
import warnings
import logging
import prince

#Here we ignore FutureWarnings and PerformanceWarnings for a more clean output
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

class FAMD():
    """
    This class performs Factor Analysis of Mixed Data (FAMD) on the features of a dataset containing both numerical and categorical features.
    It does this by using the prince library. For more information about the implementation, please refer to the documentation of the prince library.
    """

    #Here we define the number of iterations
    NUMBER_OF_ITERATIONS = 3

    def __init__(self, features_dataframe: pd.DataFrame, number_of_components: int = 2, random_state: int = 42):
        """
        Constructor for the FAMD class

        Args:
            features_dataframe (pd.DataFrame): Dataframe containing the features
            number_of_components (int, optional): Number of components. Defaults to 2.
            random_state (int, optional): Determines random number generation for centroid initialization. Defaults to 42.
        """
        #Here we initialize the features dataframe
        self.features_dataframe = features_dataframe.copy()
        #Here we initialize the number of principal components we want to obtain
        self.number_of_components = number_of_components
        #Here we initialize the random state
        self.random_state = random_state
        #Here we initialize the class that will perform the FAMD:
        self.famd = prince.FAMD(n_components=self.number_of_components,
                                n_iter=self.NUMBER_OF_ITERATIONS,
                                copy=True,
                                check_input=True,
                                engine='sklearn',
                                random_state=self.random_state)
        
    def fit(self) -> pd.DataFrame:
        """
        Function that fits the FAMD model to the features dataframe

        Returns:
            pd.DataFrame: Dataframe containing the coordinates of the features in the principal components space

        """
        #Here we fit the FAMD model to the features dataframe
        self.famd.fit(self.features_dataframe)
        
        #Here we return a dataframe containing the coordinates of the features in the principal components space
        return self.famd.row_coordinates(self.features_dataframe)
    
class KMeans():
    def __init__(self, k: int=4, max_iter: int=300, tol: float=0.0001, random_state: int=42, verbose: bool=False):
        """
        Constructor for the KMeans class

        Args:
            k (int, optional): Number of clusters. Defaults to 3.
            max_iter (int, optional): Maximum number of iterations of the k-means algorithm for a single run. 
            tol (float, optional): Relative tolerance with regards to inertia to declare convergence. 
            random_state (int, optional): Determines random number generation for centroid initialization. 
            verbose (bool, optional): Verbosity mode.
        """
        #Here we initialize the k number of clusters
        self.k = k
        #Here we initialize the maximum number of iterations
        self.max_iter = max_iter
        #Here we initialize the tolerance
        self.tol = tol
        #Here we initialize the random state
        self.random_state = random_state
        #Here we initialize the verbose mode
        self.verbose = verbose
        #Here we initialize the logger since we will use it to avoid too much output
        self.logger = logging.getLogger('py4j.java_gateway')

    def fit(self, data: np.ndarray) -> 'KMeans':
        """
        Function that fits the model to the data i.e. performs KMeans clustering to a dataset
        We perform the KMeans clustering using Spark RDDs and MapReduce in the following way:
        1) We initialize the centroids by sampling k points from the data RDD
        2) We iterate until convergence or until the maximum number of iterations is reached:
            a) Map step: We assign each point to its closest centroid
            b) Reduce step: We compute the new centroids

        Args:
            data (np.ndarray): Data to cluster

        Returns:
            self: Returns an instance of self
        """
        #Here we initialize the Spark context
        sc = self.__initialize_spark_context()
        #Here we parallelize the data in order to use the Spark RDDs
        data = sc.parallelize(data)
        #Here we initialize the centroids. For KMeans, we sample k points from the data RDD.
        self.centroids = self._initialize_centroids(data)

        #Here we iterate until convergence or until the maximum number of iterations is reached
        for i in range(self.max_iter):
            #Here we assign each point to its closest centroid
            #We do it by mapping each point to the index of the closest centroid (the one with the minimum inertia)
            closest_centroids = data.map(lambda point: self.__assign_centroids(point, self.centroids))
            #Here we compute the new centroids
            #We do it by reducing the closest centroids RDD by key (the index of the closest centroid) and computing the mean of the points assigned to each centroid
            new_centroids = closest_centroids.reduceByKey(lambda x,y: (x[0]+y[0], x[1]+y[1])).map(lambda x: x[1][0]/x[1][1]).collect()            
            
            #Here we check if the new centroids are close to the old centroids. If they are, we stop the iterations
            #We do it by checking if the distance between the old centroids and the new centroids is less than the tolerance
            if np.allclose(self.centroids, new_centroids, atol=self.tol):
                #Here we print the number of iterations in which the convergence was reached if the verbose mode is activated
                if self.verbose: print(f"Convergence reached after {i} iterations")
                break
            
            #Here we update the centroids
            self.centroids = new_centroids
        
        #At the end of the iterations, we obtain the labels of the points and the total inertia of the clustering
        self.labels = closest_centroids.map(lambda x: x[0]).collect()
        self.inertia = closest_centroids.map(lambda x: self.__inertia(x[1][0], self.centroids[x[0]])).sum()

        #Here we stop the Spark context
        sc.stop()
        #Here we return an instance of self
        return self
           
    
    def _initialize_centroids(self, data_rdd: RDD) -> List[np.ndarray]:
        """
        Function that initializes the centroids for KMeans.
        It samples k points from the data RDD.

        Args:
            data_rdd (RDD): RDD containing the data

        Returns:
            list: List of centroids
        """
        #Here we sample k points from the data RDD without replacement and return them
        return data_rdd.takeSample(False, self.k, self.random_state)
    
    def __initialize_spark_context(self) -> SparkContext:
        """
        Function that initializes the Spark context

        Returns:
            SparkContext: Spark context
        """
        #Here we set the logging level to ERROR to avoid too much output
        self.logger.setLevel(logging.ERROR)
        #Here we initialize the configuration for the Spark context
        #We use the local[*] master to use all the cores of the machine
        conf = SparkConf().setAppName("KMeans").setMaster("local[*]")
        #Here we initialize the Spark context
        sc = SparkContext(conf=conf)

        #Here we return the Spark context
        return sc
    
    def __assign_centroids(self, point: np.ndarray, centroids: List[np.ndarray]) -> Tuple[int, np.ndarray]:
        """
        Function that assigns a point to the closest centroid

        Args:
            point (np.ndarray): Point to assign
            centroids (List[np.ndarray]): List of centroids

        Returns:
            tuple: Tuple containing the index of the closest centroid and the point
        """
        #Here we compute the distances between the point and the centroids
        distances = [np.sum((point - centroid)**2) for centroid in centroids]
        #Here we return the index of the closest centroid and the point
        #We also return the point with a 1 to compute the mean in the reduce step
        return (np.argmin(distances), [point, 1])
    
    def __inertia(self, point: np.ndarray, centroid: np.ndarray) -> float:
        """
        Function that computes the inertia of a point given the centroid of its cluster
        Mathematically, the inertia of a point is the squared distance between the point and the centroid of its cluster

        Args:
            point (np.ndarray): Point
            centroid (np.ndarray): Centroid of the point
        
        Returns:
            float: Inertia of the point
        """
        #Here we compute the inertia of the point
        return np.sum((point - centroid)**2)
    
class KMeansPlusPlus(KMeans):
    def __init__(self, k=4, max_iter=300, tol=0.0001, random_state=42, verbose=False):
        """
        Constructor for the KMeansPlusPlus class

        Args:
            k (int, optional): Number of clusters. Defaults to 3.
            max_iter (int, optional): Maximum number of iterations of the k-means algorithm for a single run. 
            tol (float, optional): Relative tolerance with regards to inertia to declare convergence. 
            random_state (int, optional): Determines random number generation for centroid initialization. 
            verbose (bool, optional): Verbosity mode.
        """
        #Here we initialize from the parent class
        super().__init__(k, max_iter, tol, random_state, verbose)

    def _initialize_centroids(self, data_rdd: RDD) -> List[np.ndarray]:
        """
        Function that initializes the centroids for KMeans++
        It samples the first centroid from the data RDD and then samples the rest of the centroids from the data RDD using a probability distribution based on the distance to the closest centroid

        Args:
            data_rdd (RDD): RDD containing the data

        Returns:
            list: List of centroids
        """
        #Here we set a random seed for reproducibility
        np.random.seed(self.random_state)
        #Here we initialize the centroids list
        centroids = []
        #Here we sample the first centroid from the data RDD
        centroids.append(data_rdd.takeSample(False, 1, self.random_state)[0])
        
        #Here we iterate until we have k centroids
        for _ in range(self.k-1):
            #First we find the distance between each point and the closest centroid
            #We do it by mapping each point to the minimum of the sum of the squared distances between each point to each centroid
            min_distances = data_rdd.map(lambda point: self.__obtain_min_distance(point, centroids))
            #Then we compute the sum of the distances
            sum_distances = min_distances.sum()
            #Then we normalize the distances to obtain a probability distribution
            probabilities = min_distances.map(lambda distance: distance/sum_distances)

            #Here we sample a point from the data RDD using the probability distribution
            #First we obtain the index of the new centroid
            new_centroid_index = np.random.choice(np.arange(data_rdd.count()), p=probabilities.collect())
            #Here we obtain the new centroid
            new_centroid = data_rdd.take(new_centroid_index+1)[-1]

            #Here we append the new centroid to the centroids list
            centroids.append(new_centroid)

        #Here we return the centroids list
        return centroids

    def __obtain_min_distance(self, point:np.ndarray, centroids: List[np.ndarray]) -> float:
        """
        Function that obtains the minimum of the sum of the squared distances between each point and the closest centroid

        Args:
            point (np.ndarray): Point
            centroids (List[np.ndarray]): List of centroids

        Returns:
            float: Minimum of the sum of the squared distances between each point and the closest centroid
        """
        #Here we compute the distances between the point and the centroids
        distances = [np.sum((point - centroid)**2) for centroid in centroids]
        #Here we return the minimum of the distances
        return np.min(distances)
    
        

