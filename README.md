# MlOpsWithCats
This repository allows you to train the model to predict whether spaceship's crewmate got into anomaly or not.
Here is kaggle competition link:
https://www.kaggle.com/competitions/tsumladvanced2025

## Project setup:
After cloning repository in /ML_Module_1/ folder you should run:
`poetry install`
Project setup is done!

## Running the project:
In /ML_Module_1 folder:
### To train:
`poetry run python ./src/mlwithcats/code/model.py --path_to_dataset="<path_to_dataset>" train`
### To predict:
`poetry run python ./src/mlwithcats/code/model.py --path_to_dataset="<path_to_dataset>" predict`

Results get saved into /ML_Module_1/src/mlwithcats/predictions folder.

### Datasets:
You can find initial datasets for testing/training following this path:
`/ML_Module_1/src/mlwithcats/data`
