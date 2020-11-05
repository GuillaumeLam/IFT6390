# IFT 6390 Kaggle Competition 1 ArXiv Classification

To make sure all dependencies work, a virtual environement was created with pipenv. To start the env and launch the notebooks, simply type in the root folder:

```
pipenv shell
jupyter notebook
```

To access the virtual env kernel, you will need to change it in the jupyter notebook. You can follow the steps at: https://stackoverflow.com/questions/47295871/is-there-a-way-to-use-pipenv-with-jupyter-notebook

## Structure

In the folder **notebooks**, lies the notebooks used throughout the development of the code used for the competition and the report. All model files have the correspoding model with some having some extra code that was used during development. The notebook preprocessing contains the code for the preprocessing of the dataset. The method developped was exported to the folder **src** to be called in other notebooks. In that folder, there is also the class for Bernoulli Naive Baye's to be called by other notebooks. The notebook, gridsearch.ipynb, was used to  create the pipeline for gridsearch to find the optimal values for the hyperparameters for the best perfoming model/ model used for submission to Kaggle. Finally, the notebook, pipeline.ipynb, was used to train a model with the full dataset, predict on the test dataset, and build the csv file for the submission to Kaggle. 

In the root folder, all the csv files were the submissions to Kaggle.

The Pipfiles are the files that contain the specifications for the project.
