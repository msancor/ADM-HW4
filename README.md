# Algorithmic Methods for Data Mining - Homework 4

This is a Github repository created to submit the fourth Homework of the **Algorithmic Methods for Data Mining (ADM)** course for the MSc. in Data Science at the Sapienza University of Rome.

--- 
## What's inside this repository?

1. `README.md`: A markdown file that explains the content of the repository.

2. `main.ipynb`: A [Jupyter Notebook](https://nbviewer.org/github/msancor/ADM-HW3/blob/main/main.ipynb) file containing all the relevant exercises and reports belonging to the homework questions, the *Command Line Question*, and the *Algorithmic Question*.

3. ``modules/``: A folder including 4 Python modules used to solve the exercises in `main.ipynb`. The files included are:

    - `__init__.py`: A *init* file that allows us to import the modules into our Jupyter Notebook.

    - `data_handler.py`: A Python file including a `DataHandler` class designed to handle data cleaning and feature engineering on Kaggle's [Netflix Clicks Dataset](https://www.kaggle.com/datasets/vodclickstream/netflix-audience-behaviour-uk-movies).

    - `recommender.py`: A Python file including a `Recommender` class designed to build a Recommendation Engine with LSH using user data obtained from Kaggle's [Netflix Clicks Dataset](https://www.kaggle.com/datasets/vodclickstream/netflix-audience-behaviour-uk-movies).

    - `cluster.py`: A Python file including three classes: `FAMD`, `KMeans`, and `KMeans++` designed to perform Factor Analysis of Mixed Data on Kaggle's [Netflix Clicks Dataset](https://www.kaggle.com/datasets/vodclickstream/netflix-audience-behaviour-uk-movies) and then perform parallelized k-Means and k-Means++ clustering using PySpark.

    - `plotter.py`: A Python file including a `Plotter` class designed to build auxiliary plots for the written report on `main.ipynb`.

4. `commandline.sh`: A bash script including the code to solve the *Command Line Question*.

5. `images/`: A folder containing a screenshot of the successful execution of the `commandline.sh` script.

6. ``.gitignore``: A predetermined `.gitignore` file that tells Git which files or folders to ignore in a Python project.

7. `LICENSE`: A file containing an MIT permissive license.

## Dataset

In this homework we worked with Kaggle's predefined [Netflix Clicks Dataset](https://www.kaggle.com/datasets/vodclickstream/netflix-audience-behaviour-uk-movies).

## Important Note

If the Notebook doesn't load through Github please try all of these steps:

1. Try compiling the Notebook through its [NBViewer](https://nbviewer.org/github/msancor/ADM-HW3/blob/main/main.ipynb).

2. Try downloading the Notebook and opening it in your local computer.

---

**Author:** Miguel Angel Sanchez Cortes

**Email:** sanchezcortes.2049495@studenti.uniroma1.it

*MSc. in Data Science, Sapienza University of Rome*
