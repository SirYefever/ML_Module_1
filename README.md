# MlOpsWithCats
This repository allows you to train the model to predict whether spaceship's crewmate got into anomaly or not.
Here is kaggle competition link:
https://www.kaggle.com/competitions/tsumladvanced2025

## Running the project:
In MlWithCats folder:
### To train:
`poetry run python ./src/mlwithcats/code/model.py --path_to_dataset="<path_to_dataset>" train`
### To predict:
`poetry run python ./src/mlwithcats/code/model.py --path_to_dataset="<path_to_dataset>" predict`

Results get saved into MlWithCats/src/mlwithcats/predictions folder.

### Datasets:
You can find initial datasets for testing/training following this path:
`/MlWithCats/src/mlwithcats/data`
