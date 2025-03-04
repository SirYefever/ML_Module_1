import os
import shutil
import pandas as pd
from catboost import CatBoostClassifier
import mlflow.catboost
import argparse


def recreate_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(path)


class My_Classifier_Model:

    predictions_path = "/home/xee/dev/python/MlWithCats/src/mlwithcats/predictions/result.csv"
    model_folder_path = "/home/xee/dev/python/MlWithCats/src/mlwithcats/code/model"

    def train(path_to_dataset, self):
        print("Training...")

        data = pd.read_csv(path_to_dataset)
        X = data.drop('Transported', axis=1)
        y = data['Transported']

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

        result = numerics.merge(dummies, on='PassengerId', how='left')
        result = result.merge(bools, on='PassengerId', how='left')
        result = result.drop("PassengerId", axis=1)

        model = CatBoostClassifier()

        model.fit(result, y, logging_level='Silent')
        try:
            recreate_folder(self.model_folder_path)
            mlflow.catboost.save_model(
                cb_model=model, path=self.moder_folder_path)
        except:
            print("Saving model files into folder failed.")

    def predict(path_to_dataset, self):
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
    classifier = My_Classifier_Model()

    parser = argparse.ArgumentParser(description="Your script description")
    parser.add_argument('method', choices=['train', 'predict'])
    parser.add_argument(
        '--path_to_dataset', help="Full path to your dataset either for training or predicting.")
    args = parser.parse_args()

    if args.method == "train":
        classifier.train(args.path_to_dataset)
    elif args.method == "predict":
        classifier.predict(args.path_to_dataset)
