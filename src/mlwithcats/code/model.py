import os
import shutil
import pandas as pd
from catboost import CatBoostClassifier
import mlflow.catboost
import argparse
import optuna
from sklearn.model_selection import cross_val_score
from clearml.automation import RandomSearch
from clearml import Dataset

from clearml import Task
from clearml.automation import HyperParameterOptimizer, UniformIntegerParameterRange, DiscreteParameterRange
from clearml.automation.optuna import OptimizerOptuna

def recreate_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(path)


class My_Classifier_Model:

    predictions_path = "/home/xee/dev/python/MlWithCats/src/mlwithcats/predictions/result.csv"
    model_folder_path = "/home/xee/dev/python/MlWithCats/src/mlwithcats/code/model"

    result = pd.DataFrame()
    y = pd.DataFrame()

    def objective(self, trial):
        """
        Define the objective function for Optuna hyperparameter optimization.

        Parameters:
        trial (optuna.trial.Trial)

        Returns:
        float: Mean accuracy score
        """
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 100.0, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
            'grow_policy': trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']),
        }

        model = CatBoostClassifier(**params, logging_level='Silent')
        score = cross_val_score(model, self.result, self.y,
                                cv=5, scoring='accuracy').mean()
        return score

    def train(self, path_to_dataset):
        """
        Train a CatBoost classifier on the given dataset using Optuna for
        hyperparameter optimization.

        Parameters:
        path_to_dataset (str): Path to the CSV file containing the dataset

        Raises:
        Exception: If saving the model fails

        Returns:
        None.
        """
        print("Training...")

        data = pd.read_csv(path_to_dataset)
        X = data.drop('Transported', axis=1)
        self.y = data['Transported']

        booleanColumns = ["CryoSleep", "VIP"]
        bools = X.copy()[booleanColumns]
        bools["PassengerId"] = X["PassengerId"]

        numericColumns = ["RoomService", "FoodCourt",
                          "ShoppingMall", "Spa", "VRDeck", "Age"]
        numerics = X.copy()[numericColumns]
        numerics["PassengerId"] = X["PassengerId"]

        columnsForDummies = ["HomePlanet", "Destination"]
        dummies = pd.get_dummies(X.copy()[columnsForDummies])
        dummies["PassengerId"] = X["PassengerId"]

        self.result = numerics.merge(dummies, on='PassengerId', how='left')
        self.result = self.result.merge(bools, on='PassengerId', how='left')
        self.result = self.result.drop("PassengerId", axis=1)

        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=5)
        best_params = study.best_params

        model = CatBoostClassifier(**best_params, logging_level='Silent')
        model.fit(self.result, self.y, logging_level='Silent')

        try:
            recreate_folder(self.model_folder_path)
            mlflow.catboost.save_model(
                cb_model=model, path=self.model_folder_path)
            print("Model saved into a folder.")
        except:
            print("Saving model into folder failed.")

    def predict(self, path_to_dataset):
        """
        Predict outcomes based on the input dataset and save results to a CSV file.

        Parameters:
        path_to_dataset (str): Path to the input dataset in CSV format.

        Returns:
        None.

        """
        print("Predicting...")

        test_data = pd.read_csv(path_to_dataset)

        booleanColumns = ["CryoSleep", "VIP"]
        bools_test = test_data.copy()[booleanColumns]
        bools_test["PassengerId"] = test_data["PassengerId"]

        numericColumns = ["RoomService", "FoodCourt",
                          "ShoppingMall", "Spa", "VRDeck", "Age"]
        numerics_test = test_data.copy()[numericColumns]
        numerics_test["PassengerId"] = test_data["PassengerId"]

        columnsForDummies = ["HomePlanet", "Destination"]
        dummies_test = pd.get_dummies(test_data.copy()[columnsForDummies])
        dummies_test["PassengerId"] = test_data["PassengerId"]

        result_test = numerics_test.merge(
            dummies_test, on='PassengerId', how='left')
        result_test = result_test.merge(
            bools_test, on='PassengerId', how='left')
        result_test = result_test.drop("PassengerId", axis=1)

        model = mlflow.catboost.load_model(self.model_folder_path)

        predictions = model.predict(result_test)

        output = pd.DataFrame(
            {'PassengerId': test_data.PassengerId, 'Transported': predictions})
        try:
            output.to_csv(
                self.predictions_path, index=False)
            print("Result saved into " + self.predictions_path)
        except:
            print("Saving result to file failed!")


if __name__ == '__main__':
    task = Task.init(task_name="Executing_model", project_name="MlWithCats")
    classifier = My_Classifier_Model()

    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument('method', choices=['train', 'predict'])
    parser.add_argument(
        '--path_to_dataset', help="Full path to your dataset either for training or predicting.")
    args = parser.parse_args()

    if args.method == "train":
        dataset = Dataset.create(dataset_name="train.csv", dataset_project="MlWithCats")
        dataset.add_files(path=args.path_to_dataset)
        task.upload_artifact('mlflow_model', artifact_object=classifier.model_folder_path)
        classifier.train(args.path_to_dataset)
    elif args.method == "predict":
        dataset = Dataset.create(dataset_name="test.csv", dataset_project="MlWithCats")
        dataset.add_files(path=args.path_to_dataset)
        classifier.predict(args.path_to_dataset)
    dataset.upload()
    dataset.finalize()
