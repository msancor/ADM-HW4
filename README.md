# Algorithmic Methods for Data Mining - Homework 4

This is a Github repository created to submit the fourth Homework of the **Algorithmic Methods for Data Mining (ADM)** course for the MSc. in Data Science at the Sapienza University of Rome.

--- 
## What's inside this repository?

1. `README.md`: A markdown file that explains the content of the repository.

2. `main.ipynb`: A [Jupyter Notebook](https://nbviewer.org/github/msancor/ADM-HW3/blob/main/main.ipynb) file containing all the relevant exercises and reports belonging to the homework questions, the *Command Line Question*, and the *Algorithmic Question*.

3. ``modules/``: A folder including 4 Python modules used to solve the exercises in `main.ipynb`. The files included are:

    - `__init__.py`: A *init* file that allows us to import the modules into our Jupyter Notebook.

    - `web_scraper.py`: A Python file including a `WebScraper` class designed to perform web scraping on the multiple pages of the [MSc. Degrees](https://www.findamasters.com/masters-degrees/msc-degrees/) website.

    - `html_parser.py`: A Python file including a `HTMLParser` class designed to parse the HTML files obtained by the web scraping process and extract relevant information.

    - `data_preprocesser.py`: A Python file including a `DataPreprocesser` class designed to pre-process text data in order to obtain information and build a Search Engine.

    - `search_engine.py`: A Python file including three classes: `SearchEngine`, `TopKSearchEngine`, and `WeightedTopKSearchEngine` designed to implement different versions of a Search Engine that queries information from our MSc. courses dataset.

    - `map_plotter.py`: A Python file including a `MapPlotter` class designed to plot a map including the results of our Search Engines.

4. `commandline.sh`: A bash script including the code to solve the *Command Line Question*.

5. ``.gitignore``: A predetermined `.gitignore` file that tells Git which files or folders to ignore in a Python project.

6. `LICENSE`: A file containing an MIT permissive license.

## Dataset



## Important Note

If the Notebook doesn't load through Github please try all of these steps:

1. Try compiling the Notebook through its [NBViewer](https://nbviewer.org/github/msancor/ADM-HW3/blob/main/main.ipynb).

2. Try downloading the Notebook and opening it in your local computer.

---

**Author:** Miguel Angel Sanchez Cortes

**Email:** sanchezcortes.2049495@studenti.uniroma1.it

*MSc. in Data Science, Sapienza University of Rome*
