# Revisiting Data Visualization with t-SNE: A Reproduction Study
**Description:** This is an ML reproducibility report for the paper ["Visualizing Data using t-SNE"](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf). 
* First, we reproduce the t-SNE visualizations on the three datasets considered in the paper, comparing t-SNE to Sammon Mapping, Isomap and Locally Linear Embedding visualizations.
* We then perform ablation studies that verify the effectiveness of t-SNE in alleviating the crowding problem, the use of PCA reduction in data preprocessing, and generalization error of classifiers trained on the visualization data.
* Overall, we found it relatively straightforward to reproduce the results.
* For future practitioners, we suggest preprocessing the data by using the principal components that explain 90\% of the dataset variance.
  
`report.pdf` contains a description of all our experiments and results.

## Installation
Before running the project, you need to set up the required environment. Follow these steps:

**1. Clone the Repository:**
```
git clone https://github.com/j-c-carr/Revisiting-Data-Visualization-with-t-SNE.git
cd Revisiting-Data-Visualization-with-t-SNE
```
**2. Create a Virtual Environment (Optional but Recommended):**
```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```
**3. Install Dependencies:**
```
pip install -r requirements.txt
```

## Usage
To use this project, follow these steps:

**1. Run Jupyter Notebooks:**
* Launch Jupyter Notebook in the project directory:
```
jupyter notebook
```
* Open the relevant Jupyter notebooks, such as:
  - `reproduce_original_plots.ipynb`: reproduces the [original paper](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)'s results
  - `pca_analysis.ipynb`: ablation studies for studying the effect of PCA reduction on t-SNE visualization
  - `crowding_problem.ipynb`: compares t-SNE to symmetric SNE in order to visualize the "Crowding Problem" (described in Section 3.2.1 of `report.pdf`)

**2. Explore the Code**
* Review the codebase:
  - See `symmetric_sne.py` and `sammon.py` for implementations of symmetric SNE and sammon mapping visualization methods.
  - See `data_loader.py` and `utils.py` for data loading and helper functions. 
 
**3. Customize and Experiment:**
* Feel free to customize parameters and experiment with the code.
* Note any additional instructions provided within the notebooks.
