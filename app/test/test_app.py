import json
import unittest
from unittest.mock import patch

import pandas as pd

from run import app


class FlaskAppTest(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    @patch("app.src.models.predict.load_artifacts", return_value="../models/objects")
    @patch("app.src.data.predict.make_dataset.make_dataset")
    @patch("app.src.models.predict.load_production_model")
    def test_predict_endpoint(self, mock_load_production_model, mock_make_dataset, mock_load_artifacts):
        init_cols = [
            "PassengerId",
            "Pclass",
            "Name",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Ticket",
            "Fare",
            "Cabin",
            "Embarked",
        ]
        mock_make_dataset.return_value = pd.DataFrame(
            data=[[10, 2, "Nasser, Mrs Nicholas", "female", 14, 1, 0, "237736", 30.0708, None, "C"]], columns=init_cols)
        mock_load_artifacts.return_value = "../models/objects"

        data = [[10, 2, "Nasser, Mrs Nicholas", "female", 14, 1, 0, "237736", 30.0708, None, "C"]]

        with patch("app.src.models.predict.load_production_model") as mock_load_production_model:
            mock_load_production_model.return_value.predict.return_value.tolist.return_value = [1]

            response = self.app.post("/predict", json=data)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(json.loads(response.data), {"Predicted value": [1]})

    @patch("app.src.models.train_model.evaluate_model")
    def test_train_model_endpoint(self, mock_evaluate_model):
        response = self.app.get("/train-model")
        self.assertEqual(response.status_code, 200)

    def test_root_endpoint(self):
        response = self.app.get("/")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertEqual(data, {"Proyecto": "Mod. 4 - Ciclo de vida de modelos IA"})

    def test_load_dataset_endpoint(self):
        response = self.app.post("/load-dataset")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.get_data(as_text=True))
        self.assertIn("message", data)


if __name__ == "__main__":
    unittest.main()
