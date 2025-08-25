import unittest
import pandas as pd
import torch
from torch.utils.data import DataLoader
import os
import json


from main import (
    load_dataset, preprocess, SleepDataset, build_model,
    train_and_evaluate, evaluate, save_model, load_model,
    predict_new_user
)
from tests.TestUtils import TestUtils


class TestSleepQualityYaksha(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_obj = TestUtils()
        cls.df = load_dataset("Sleep_health_and_lifestyle_dataset.csv")
        cls.X_train, cls.y_train, cls.X_test, cls.y_test, cls.input_dim = preprocess(cls.df, fit=True)
        cls.dataset = SleepDataset(cls.X_train, cls.y_train)
        cls.dataloader = DataLoader(cls.dataset, batch_size=4, shuffle=False)
        cls.model = build_model(cls.input_dim, num_classes=3)

    # ---------------- load_dataset ----------------
    def test_load_dataset(self):
        try:
            result = isinstance(self.df, pd.DataFrame)
            self.test_obj.yakshaAssert("TestLoadDataset", result, "functional")
            print("TestLoadDataset =", "Passed" if result else "Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestLoadDataset", False, "functional")
            print("TestLoadDataset = Failed | Exception:", e)

    # ---------------- preprocess ----------------
    def test_preprocess(self):
        try:
            result = self.X_train.shape[1] == self.input_dim
            self.test_obj.yakshaAssert("TestPreprocess", result, "functional")
            print("TestPreprocess =", "Passed" if result else "Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestPreprocess", False, "functional")
            print("TestPreprocess = Failed | Exception:", e)

    # ---------------- SleepDataset ----------------
    def test_sleep_dataset_length(self):
        try:
            result = len(self.dataset) == len(self.X_train)
            self.test_obj.yakshaAssert("TestSleepDatasetLength", result, "functional")
            print("TestSleepDatasetLength =", "Passed" if result else "Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestSleepDatasetLength", False, "functional")
            print("TestSleepDatasetLength = Failed | Exception:", e)

    def test_sleep_dataset_getitem(self):
        try:
            x, y = self.dataset[0]
            result = isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
            self.test_obj.yakshaAssert("TestSleepDatasetGetItem", result, "functional")
            print("TestSleepDatasetGetItem =", "Passed" if result else "Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestSleepDatasetGetItem", False, "functional")
            print("TestSleepDatasetGetItem = Failed | Exception:", e)

    # ---------------- build_model ----------------
    def test_build_model(self):
        try:
            sample_input = self.X_train[0:1]
            output = self.model(sample_input)
            result = output.shape[1] == 3
            self.test_obj.yakshaAssert("TestBuildModel", result, "functional")
            print("TestBuildModel =", "Passed" if result else "Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestBuildModel", False, "functional")
            print("TestBuildModel = Failed | Exception:", e)

    # ---------------- train_and_evaluate ----------------
    def test_train_and_evaluate_runs(self):
        try:
            train_loader = DataLoader(self.dataset, batch_size=8, shuffle=True)
            test_loader = DataLoader(SleepDataset(self.X_test, self.y_test), batch_size=8)
            # Run only 1 epoch to check logic works
            train_and_evaluate(self.model, train_loader, test_loader, epochs=1)
            result = True
            self.test_obj.yakshaAssert("TestTrainAndEvaluate", result, "functional")
            print("TestTrainAndEvaluate =", "Passed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestTrainAndEvaluate", False, "functional")
            print("TestTrainAndEvaluate = Failed | Exception:", e)

    # ---------------- evaluate ----------------
    def test_evaluate(self):
        try:
            loader = DataLoader(SleepDataset(self.X_test, self.y_test), batch_size=8)
            acc = evaluate(self.model, loader)
            result = isinstance(acc, float)
            self.test_obj.yakshaAssert("TestEvaluate", result, "functional")
            print("TestEvaluate =", "Passed" if result else "Failed")
        except Exception as e:
            self.test_obj.yakshaAssert("TestEvaluate", False, "functional")
            print("TestEvaluate = Failed | Exception:", e)

    # ---------------- save_model + load_model ----------------
    def test_save_and_load_model(self):
        try:
            save_model(self.model, "yaksha_sleep_model.pth")
            loaded = load_model("yaksha_sleep_model.pth", self.input_dim)
            sample_input = self.X_train[0:1]
            out1 = self.model(sample_input)
            out2 = loaded(sample_input)
            result = out1.shape == out2.shape
            self.test_obj.yakshaAssert("TestSaveAndLoadModel", result, "functional")
            print("TestSaveAndLoadModel =", "Passed" if result else "Failed")
            os.remove("yaksha_sleep_model.pth")
        except Exception as e:
            self.test_obj.yakshaAssert("TestSaveAndLoadModel", False, "functional")
            print("TestSaveAndLoadModel = Failed | Exception:", e)

    def test_new_user_prediction_from_file(self):
        try:
            with open("new_user.txt", "r") as f:
                new_user = json.load(f)

            result, probs = predict_new_user(self.model, new_user)

            # Set your expected label here based on known behavior
            expected_label = "Average"

            check = (result == expected_label)
            self.test_obj.yakshaAssert("TestNewUserPredictionFromFile", check, "functional")
            print("TestNewUserPredictionFromFile =", "Passed" if check else f"Failed | Got {result}")
        except Exception as e:
            self.test_obj.yakshaAssert("TestNewUserPredictionFromFile", False, "functional")
            print("TestNewUserPredictionFromFile = Failed | Exception:", e)


if __name__ == "__main__":
    unittest.main()
