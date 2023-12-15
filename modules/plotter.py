#Here we import the necessary libraries
from sklearn.metrics import silhouette_score
from modules.cluster import KMeans
import matplotlib.pyplot as plt
from kneed import KneeLocator
from typing import List
import pandas as pd
import numpy as np


#Here we set the style of the plots.
#First, we set as default that matplotlib plots text should be in LaTeX format.
plt.rcParams['text.usetex'] = True
#Here we set the font family to serif.
plt.rcParams['font.family'] = 'serif'
#Here we set the font size
plt.rcParams['font.size'] = 10
#Here we set the label size for axes
plt.rcParams['axes.labelsize'] = 10
#Here we set the label weight for axes
plt.rcParams['axes.labelweight'] = 'bold'
#Here we set the title size for axes
plt.rcParams['axes.titlesize'] = 10
#Here we set the ticks label size
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
#Here we set the legend font size
plt.rcParams['legend.fontsize'] = 10
#Here we set the figure title size
plt.rcParams['figure.titlesize'] = 20


class Plotter():
    """
    Class that contains the functions to plot the results of the data analysis.
    """
    #Here we define the number of hash functions used in the LSH algorithm
    NUMBER_OF_HASH_FUNCTIONS = 100
    #Here we define the maximum number of clusters to be used in the KMeans algorithm
    MAX_K = 10
    #Here we define the minimum number of clusters to be used in the KMeans algorithm
    MIN_K = 2

    def __init__(self):
        """
        Constructor of the class.
        """
        #Here we pass
        pass

    def plot_probability(self) -> None:
        """
        Function that plots the probability of a pair of users being hashed to the same bucket given they have a similarity score s
        for different values of b (number of bands).
        """
        #Here we define the x-axis
        x = np.linspace(0, 1, 100)

        #Here we plot the probability of a pair of users being hashed to the same bucket given they have a similarity score s
        plt.figure(figsize=(8, 4))
        plt.plot(x, self.__probability_function(x, self.NUMBER_OF_HASH_FUNCTIONS, 5), label='$b=5$', lw=2, alpha=0.6, color='red')
        plt.plot(x, self.__probability_function(x, self.NUMBER_OF_HASH_FUNCTIONS, 20), label='$b=20$', lw=2, alpha=0.6, color='green')
        plt.plot(x, self.__probability_function(x, self.NUMBER_OF_HASH_FUNCTIONS, 25), label='$b=25$', lw=2, alpha=0.6, color='orange')
        plt.plot(x, self.__probability_function(x, self.NUMBER_OF_HASH_FUNCTIONS, 50), label='$b=50$', lw=2, alpha=0.6, color='blue')
        
        #Here we set the legends, labels and title
        plt.legend()
        plt.xlabel('Jaccard Similarity score')
        plt.ylabel('Probability')
        plt.title('Probability for $n=100$ hash functions and different values of $b$')
        pass

    def plot_principal_components(self, clustering_dataframe: pd.DataFrame, famd_dataframe: pd.DataFrame) -> None:
        """
        Function that plots the principal components by distinguishing the different categories for the most active time of day and the most common click length

        Args:
            clustering_dataframe (pd.DataFrame): Dataframe containing the dataset and its features
            famd_dataframe (pd.DataFrame): Dataframe containing the coordinates of the features in the principal components space
        """
        #Here we associate each row of the clustering dataframe to a color depending on the category it belongs to for the most active time of day
        colors = {'Morning':'red', 'Afternoon':'blue', 'Night':'green'}
        col = [colors[i] for i in clustering_dataframe['most_active_time_day']]
        #We create a list of legend entries for the most active time of day
        legend_entries = [plt.Line2D([0], [0], marker='o', color='w', label=key, markerfacecolor=value, markersize=10) for key, value in colors.items()]

        #Here we do the same but for the most_common_click_length
        colors_click = {'Short':'red', 'Medium':'blue', 'Long':'green'}
        col_click = [colors_click[i] for i in clustering_dataframe['most_common_click_length']]
        #We create a list of legend entries for the most common click length
        legend_entries_click = [plt.Line2D([0], [0], marker='o', color='w', label=key, markerfacecolor=value, markersize=10) for key, value in colors_click.items()]

        #Here we plot the principal components by distinguishing the different categories for the most active time of day and the most common click length
        _, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].scatter(famd_dataframe[0].to_list(), famd_dataframe[1].to_list(), c=col, alpha=0.5)
        axes[0].legend(legend_entries, colors.keys())
        axes[0].set_xlabel('First component')
        axes[0].set_ylabel('Second component')
        axes[0].set_title('Most Active Time of Day')
        axes[1].scatter(famd_dataframe[0].to_list(), famd_dataframe[1].to_list(), c=col_click, alpha=0.5)
        axes[1].legend(legend_entries_click, colors_click.keys())
        axes[1].set_xlabel('First component')
        axes[1].set_ylabel('Second component')
        axes[1].set_title('Most Common Click Length')
        pass

    def plot_elbow(self, data: np.ndarray) -> None:
        """
        Function that plots the elbow curve for the KMeans clustering algorithm

        Args:
            data (np.ndarray): Data to be clustered
        """
        #Here we compute the inertia for different values of k
        inertias = []
        for k in range(self.MIN_K, self.MAX_K+1):
            kmeans = KMeans(k=k, random_state=42).fit(data)
            inertias.append(kmeans.inertia)
        
        #Here we compute the elbow point
        kn = KneeLocator(range(self.MIN_K, self.MAX_K+1), inertias, curve='convex', direction='decreasing')

        #Here we plot the elbow curve
        plt.figure(figsize=(8, 4))
        plt.plot(range(self.MIN_K, self.MAX_K+1), inertias, marker='o', color='blue', alpha=0.6, lw=2)
        plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed' , colors='red', lw=2, alpha=0.6, label='Elbow Point')
        plt.title('Elbow Curve')
        plt.ylabel('Inertia')
        plt.xlabel('$k$')
        plt.grid(True)
        plt.legend()
        pass

    def plot_silhouette(self, data: np.ndarray) -> None:
        """
        Function that plots the silhouette score for the KMeans clustering algorithm

        Args:
            data (np.ndarray): Data to be clustered
        """
        #Here we compute the silhouette score for different values of k
        silhouette_scores = []
        for k in range(self.MIN_K, self.MAX_K+1):
            kmeans = KMeans(k=k, random_state=42).fit(data)
            silhouette_scores.append(silhouette_score(data, kmeans.labels, sample_size=10000))
        
        #Here we plot the silhouette score
        plt.figure(figsize=(8, 4))
        plt.plot(range(self.MIN_K, self.MAX_K+1), silhouette_scores, marker='o', color='blue', alpha=0.6, lw=2)
        plt.vlines(np.argmax(silhouette_scores)+2, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', colors='red', lw=2, alpha=0.6, label='Max Silhouette')
        plt.title('Silhouette Score')
        plt.ylabel('Silhouette Score')
        plt.xlabel('$k$')
        plt.grid(True)
        plt.legend()
        pass

    def plot_centroids(self, data:np.ndarray, centroids: List[np.ndarray], labels: List[int], title: str = "k-Means"):
        """
        Function that plots the centroids of each cluster after performing KMeans

        Args:
            data (np.ndarray): Data to perform the clustering from.
            centroids (list); Centroids obtained with KMeans.
            labels (list): Labels obtained with KMeans.
            title (str): Title of the plot.
        """
        #Here we obtain the x and y values of the centroids
        x_centroid_values = [x for x,_ in centroids]
        y_centroid_values = [y for _,y in centroids]

        #Here we plot the clusters with the centroids
        plt.figure(figsize=(8,6))
        plt.scatter(data[:,0], data[:,1], c=labels, cmap='rainbow', alpha=0.5)
        plt.scatter(x_centroid_values, y_centroid_values, c='black', s=100, alpha=0.8, label="Centroids")
        plt.xlabel("First component")
        plt.ylabel("Second component")
        plt.title(f"Final {title} Centroids")
        plt.legend()
        pass


    def __probability_function(self, x: float, n: int, b: int) -> float:
        """
        Function that computes the probability of a pair of users being hashed to the same bucket given they have a similarity score s
        Args:
            x (float): Similarity score
            n (int): Number of hash functions
            b (int): Number of bands

        Returns:
            probability (float): Probability of a pair of users being hashed to the same bucket given they have a similarity score s
        """
        #Here we define the number of rows per band
        r = n/b
        #Here we compute the probability of a pair of users being hashed to the same bucket given they have a similarity score s
        return 1 - (1-x**r)**b
