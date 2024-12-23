import pandas as pd
import numpy as np
import mlflow

from copy import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from mlflow.models import infer_signature

from column import CommodityFlow
from sstruct import Stage, TrainTestPair, Pairs


# Load data and provide other classes with read data capability
# all raed centralized here so there would be no duplicate raw data read
class ScenarioManager:
    def __init__(self):
        self.dataloader = None
        self.datacleaner = None
        self.datatransform = None
        self.train = None
        return

    def set_dataloader(self, dataloader):
        self.dataloader = dataloader
        return self

    def set_datacleaner(self, datacleaner):
        self.datacleaner = datacleaner
        return self

    def set_datatransform(self, datatransform):
        self.datatransform = datatransform
        return self

    def set_train(self, train):
        self.train = train
        return self

    def default_path(self):
        df = self.dataloader.load_data()
        df_cleaned = self.datacleaner.clean_data(df)
        pairs = self.datatransform.transform_data(df_cleaned)
        self.train.train_data(pairs)
        return self
        
# for better visibility, located here, once it become large, it will has its own file
# load data from disk or other source to programs
# alter the column to enumerated column name
class DataLoader:
    def __init__(self, path,chunk=1000):
        self.path = path
        self.raw_data = None
        self.chunk_size = chunk

    def load_data(self):
        df = pd.read_csv(self.path, nrows=self.chunk_size)
        df = self.replace_columns(df)
        self.raw_data = df
        return copy(self.raw_data)

    def replace_columns(self, df):
        # it has minus one because index in python began in 0
        replace_map = {df.columns[cf.value-1]:cf for cf in CommodityFlow}
        return df.rename(columns=replace_map)

# for better visibility, located here, once it become large, it will has its own file
class DataCleaner:
    def __init__(self):
        self.cleaned_data = None

    def basic_removal(self):
        return [
            # using states to limit column number
            CommodityFlow.ORIGIN_DISTRICT,
            CommodityFlow.ORIGIN_CFS_AREA,
            CommodityFlow.DESTINATION_DISTRICT,
            CommodityFlow.DESTINATION_CFS_AREA,

            # we use NAICS as goods categarization
            CommodityFlow.SCTG,
            CommodityFlow.QUARTER,

            # we use actual route instead of geodesic distance
            CommodityFlow.SHIPMENT_DISTANCE_CIRCLE,

            # we disable all options
            CommodityFlow.IS_TEMPERATURE_CONTROLLED,
            CommodityFlow.IS_EXPORT,
            CommodityFlow.EXPORT_COUNTRY,
            CommodityFlow.HAZMAT,
            CommodityFlow.WEIGHT_FACTOR,
        ]

    def clean_data(self, df):
        self.cleaned_data = df
        (self
            .remove_columns(self.basic_removal())
            .remove_nan_rows()
         )
        return copy(self.cleaned_data)
    
    def remove_columns(self, columns):
        self.cleaned_data.drop(columns, axis=1, inplace=True)
        return self

    def remove_nan_rows(self):
        self.cleaned_data.dropna(inplace=True)
        return self

# for better visibility, located here, once it become large, it will has its own file
class DataTransform:
    def __init__(self):
        self.ohe = {}
        self.mm = {}

        self.train_pair = None
        self.valid_pair = None
        self.test_pair = None

    def transform_data(self, df):
        self.transformed_data = df
        (self
            .split_stage()
            .one_hot_encoding()
            .minmax()
            .reapply()
            .shape_check()
         )

        return self.get_pairs()
    
    def get_pairs(self):
        return Pairs(self.train_pair, self.valid_pair, self.test_pair)

    def print_shapes(self):
        ddict = {
            Stage.TRAIN.name: {
                "X": self.train_pair.X.shape,
                "y": self.train_pair.y.shape,
            },
            Stage.VALID.name: {
                "X": self.valid_pair.X.shape,
                "y": self.valid_pair.y.shape,
            },
            Stage.TEST.name: {
                "X": self.test_pair.X.shape,
                "y": self.test_pair.y.shape,
            } 
        }
        print(pd.DataFrame(ddict))
        return self

    def shape_check(self):
        train_column_num = self.train_pair.X.shape[1]
        valid_column_num = self.valid_pair.X.shape[1]
        test_column_num = self.test_pair.X.shape[1]
        if train_column_num != valid_column_num:
            raise ValueError("number of column in train and valid should be the same")
        if train_column_num != test_column_num:
            raise ValueError("number of column in train and test should be the same")
        if test_column_num != valid_column_num:
            raise ValueError("number of column in test and valid should be the same")
        return self


    def split_stage(self):
        train_cols = CommodityFlow.train(self.transformed_data.columns)
        train = self.transformed_data[train_cols]
        test = self.transformed_data[CommodityFlow.target()]
        Xtr, Xt, ytr, yt = train_test_split(
                train,
                test,
                test_size=0.2,
                random_state=42,
        )
        Xv, Xte, yv, yte = train_test_split(
                Xt,
                yt,
                test_size=0.5,
                random_state=42,
        )
        self.train_pair = TrainTestPair(Xtr, ytr, Stage.TRAIN)
        self.valid_pair = TrainTestPair(Xv, yv, Stage.VALID)
        self.test_pair = TrainTestPair(Xte, yte, Stage.TEST)
        return self

    def one_hot_encoding(self):
        for cat in self._catcol(self.train_pair):
            # it is very likely that we dont encode every possible enums
            # because we only use partial data
            self.ohe[cat] = OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist')
            arrayed = self.train_pair.X[[cat]]
            self.ohe[cat].fit(arrayed)
            self._transform_ohe(self.train_pair, cat)
        return self

    def _transform_ohe(self, pair, cat):
        arrayed = pair.X[[cat]]
        transformed = self.ohe[cat].transform(arrayed)
        encoded_columns = self.ohe[cat].get_feature_names_out([cat.name])
        new_columns = pd.DataFrame(transformed, columns=encoded_columns)
        pair.X.drop(cat, axis=1, inplace=True)
        arrayed = np.concat([np.array(pair.X), new_columns], axis=1)
        all_columns = list(pair.X.columns) + list(new_columns.columns)
        pair.X = pd.DataFrame(arrayed, columns=all_columns)
        return self

    def minmax(self):
        for num in self._numcol(self.train_pair):
            self.mm[num] = MinMaxScaler()
            arrayed = self.train_pair.X[[num]]
            self.mm[num].fit(arrayed)
            self._transform_mm(self.train_pair, num)
        return self

    def _transform_mm(self, pair, num):
        arrayed = pair.X[[num]]
        transformed = self.mm[num].transform(arrayed)
        pair.X[num] = transformed

    def _catcol(self, pair):
        return list(set(
            pair.X.columns) & 
            set(CommodityFlow.categorical()
        ))
    def _numcol(self, pair):
        return list(set(
            pair.X.columns) & 
            set(CommodityFlow.numerical()
        ))

    def reapply(self):
        for pair in [self.valid_pair, self.test_pair]:
            for cat in self._catcol(pair):
                self._transform_ohe(pair, cat)
            for num in self._numcol(pair):
                self._transform_mm(pair, num)
        return self

# for better visibility, located here, once it become large, it will has its own file
class Train:
    def __init__(self):
        self.train_pair = None
        self.valid_pair = None
        self.test_pair = None

        self.is_using_tracker = False
        self.tracker_path = None
        self.experiment_name = None

    def set_tracker_path(self, tracker_path):
        self.is_using_tracker = True
        self.tracker_path = tracker_path
        return self

    def set_train_name(self, name):
        self.experiment_name = name
        return self

    def train_data(self, pairs):
        self.train_pair = pairs.get_train_pair()
        self.valid_pair = pairs.get_valid_pair()
        self.test_pair = pairs.get_test_pair()
        if self.is_using_tracker:
            return self.train_data_with_tracker()
        return self

    def train_data_with_tracker(self):
        mlflow.set_tracking_uri(uri=self.tracker_path)
        mlflow.set_experiment(self.experiment_name)

        params, lr = self.basic_linear_regression()
        train_mse, valid_mse = self.train_validate(params, lr)

        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metric("train_mse", train_mse)
            mlflow.log_metric("valid_mse", valid_mse)
            mlflow.log_metric("total_trained", self.train_pair.X.shape[0])
            mlflow.log_metric("total_features", self.train_pair.X.shape[1])
            model_info = mlflow.sklearn.log_model(
                sk_model=lr,
                artifact_path="cfs_model",
                signature=infer_signature(self.train_pair.x_array(), lr.predict(self.train_pair.x_array())),
                input_example=self.train_pair.x_array(),
                registered_model_name="hello",
            )
        return self

    def train_validate(self, param, learn):
        learn.fit(self.train_pair.x_array(), self.train_pair.y)
        y_pred_train = learn.predict(self.train_pair.x_array())
        train_mse = mean_squared_error(self.train_pair.y, y_pred_train)
        y_pred_valid = learn.predict(self.valid_pair.x_array())
        valid_mse = mean_squared_error(self.valid_pair.y, y_pred_valid)
        return train_mse, valid_mse

    def basic_linear_regression(self):
        return {}, LinearRegression()

        


DATASET_PATH = "dataset/cfs_2017.csv"
TRACKER_PATH = "http://mlflow:5000" # see dock r compose for details
if __name__ == "__main__":
    dataloader = DataLoader(DATASET_PATH)
    datacleaner = DataCleaner()
    datatransform = DataTransform()
    train = (Train()
        .set_tracker_path(TRACKER_PATH)
        .set_train_name("humamtest"))
    (ScenarioManager()
        .set_dataloader(dataloader)
        .set_datacleaner(datacleaner)
        .set_datatransform(datatransform)
        .set_train(train)
        .default_path()
     )
