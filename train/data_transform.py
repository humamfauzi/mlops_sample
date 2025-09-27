import pandas as pd
import numpy as np
import os 
import pickle

from abc import ABC, abstractmethod
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
from train.column import TabularColumn
from train.sstruct import Pairs, Stage, FeatureTargetPair
from train.wrapper import ProcessWrapper
from repositories.dummy import DummyMLflowRepository, Manifest
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum

class TabularDataTransform(ABC):
    @abstractmethod
    def transform_data(self, df: pd.DataFrame) -> Pairs:
        pass

class TransformationMethods(Enum):
    # would replace the original column with the transformation
    REPLACE = 1
    # would append the transformation to the original column; the original
    # column would still exist
    APPEND = 2

    # would append the transformation to the original column and remove the original
    APPEND_AND_REMOVE = 3

@dataclass
class Keeper:
    # The collector of the transformation
    # Collect all neccessary transformation properties so it could be
    # reproduced later by the server
    name: str
    column: Enum
    function: any
    methods: TransformationMethods

class Transformer:
    def __init__(self, column: TabularColumn):
        self.column = column
        self.keepers = []

    @classmethod
    def parse_instruction(cls, properties: dict, call: List[dict]):
        column = TabularColumn.from_string(properties.get("reference"))
        c = cls(column)
        for step in call:
            if step["type"] == "log_transformation":
                for col in step["columns"]:
                    c.keepers.append(
                        Keeper(
                            name="log",
                            column=column.from_enum(col),
                            function=ProcessWrapper(np.log, np.exp),
                            methods=TransformationMethods[step["condition"].upper()]
                        )
                    )
            elif step["type"] == "normalization":
                for col in step["columns"]:
                    c.keepers.append(
                        Keeper(
                            name="normalization",
                            column=column.from_enum(col),
                            function=Normalizer(norm='l2'),
                            methods=TransformationMethods[step["condition"].upper()]
                        )
                    )
            elif step["type"] == "min_max_transformation":
                for col in step["columns"]:
                    c.keepers.append(
                        Keeper(
                            name="min_max",
                            column=column.from_enum(col),
                            function=MinMaxScaler(),
                            methods=TransformationMethods[step["condition"].upper()]
                        )
                    )
            elif step["type"] == "one_hot_encoding":
                for col in step["columns"]:
                    c.keepers.append(
                        Keeper(
                            name="one_hot_encoding",
                            column=column.from_enum(col),
                            function=OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist'),
                            methods=TransformationMethods.APPEND_AND_REMOVE
                        )
                    )
            elif step["type"] == "standardization":
                for col in step["columns"]:
                    c.keepers.append(
                        Keeper(
                            name="standardization",
                            column=column.from_enum(col),
                            function=StandardScaler(),
                            methods=TransformationMethods[step["condition"].upper()]
                        )
                    )
            return c
    
    def _save_manifest(self, input_data):
        # TODO: Require some tracking mechanism
        pass

    def _split_stage(self, transformed_data: pd.DataFrame):
        # Based on column enums, choose all column that belong to a feature
        train_cols = self.column.feature(transformed_data.columns)
        train = transformed_data[train_cols]

        # Based on column enums, choose all column that belong to be a target
        # Some column might be an ID therefore we still need to declare which one is the target
        test = transformed_data[self.column.target()]

        # double split, first is the train and test
        # and the second one is for valid and test based on previous test
        # TODO: make the test size configurable
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

        # Collect it as a pair for easier grouping
        train_pair = FeatureTargetPair(Xtr, ytr, Stage.TRAIN)
        valid_pair = FeatureTargetPair(Xv, yv, Stage.VALID)
        test_pair = FeatureTargetPair(Xte, yte, Stage.TEST)
        return train_pair, valid_pair, test_pair

    def _setup_transformation(self, train_pair: FeatureTargetPair):
        # Fitting all the transformation function ONLY to training data to avoid information leakage
        # this is only the fitting part, the application of the transformation happen later
        for keeper in self.keepers:
            if keeper.column in train_pair.X.columns:
                keeper.function.fit(train_pair.X[keeper.column].to_numpy().reshape(-1, 1))
            else:
                keeper.function.fit(train_pair.y.to_numpy().reshape(-1, 1))
        return self
        
    def _applies_transformation_to_all(self, train_pair, validation_pair, test_pair):
        for keeper in self.keepers:
            if keeper.methods == TransformationMethods.REPLACE:
                return self._replace(keeper, train_pair, validation_pair, test_pair)
            elif keeper.methods == TransformationMethods.APPEND:
                return self._append(keeper, train_pair, validation_pair, test_pair)
            elif keeper.methods == TransformationMethods.APPEND_AND_REMOVE:
                return self._append_and_remove(keeper, train_pair, validation_pair, test_pair)

    def _replace(self, keeper, train_pair, validation_pair, test_pair):
        '''
        replace the original column with the transformed column 
        '''
        for pairs in [train_pair, validation_pair, test_pair]:
            if keeper.column in train_pair.X.columns:
                transformed = keeper.function.transform(pairs.X[keeper.column].to_numpy().reshape(-1, 1))
                pairs.X[keeper.column] = transformed
            else:
                pairs.y = pd.DataFrame(keeper.function.transform(pairs.y.to_numpy().reshape(-1,1)))
        return self

    def _append(self, keeper, train_pair, validation_pair, test_pair):
        '''
        append the transformed column and keep the original column
        '''
        for pairs in [train_pair, validation_pair, test_pair]:
            if keeper.column not in pairs.X.columns:
                raise ValueError(f"column {keeper.column} is not a feature")
            transformed = keeper.function.transform(pairs.X[keeper.column].to_numpy().reshape(-1, 1))
            pairs.X[keeper.column] = transformed
        return self

    def _append_and_remove(self, keeper, train_pair, validation_pair, test_pair):
        '''
        append all the new columns and remove the original column
        '''
        for pairs in [train_pair, validation_pair, test_pair]:
            if keeper.column not in pairs.X.columns:
                raise ValueError(f"column {keeper.column} is not a feature")
            transformed = keeper.function.transform(pairs.X[[keeper.column]])
            encoded_columns = keeper.function.get_feature_names_out([keeper.column.name])
            new_columns = pd.DataFrame(transformed, columns=encoded_columns, dtype='int', index=pairs.X.index)

            pairs.X = pd.concat([
                pairs.X.drop(columns=[keeper.column]), 
                new_columns
            ], axis=1)
        return self

    def _shape_check(self, train_pair, validation_pair, test_pair):
        return self

    def _save_transformation(self):
        pass

    def execute(self, input_data: pd.DataFrame) -> Pairs:
        # save the transformation manifest
        # contain information about the model input
        self._save_manifest(input_data)

        # all data should be splitted first here
        # so all encoding and transformation is based on training data
        # to prevent learning leakage. All transformation happen between
        # splitting and reapplied
        train_pair, validation_pair, test_pair = self._split_stage(input_data)
        (self
            # call all collected transformation based on train data but
            # does not applies it to train immdiately
            ._setup_transformation(train_pair)

            # call all transformation function and applies it to all stages
            ._applies_transformation_to_all(train_pair, validation_pair, test_pair)

            # ensure that every stage has the same column count
            ._shape_check(train_pair, validation_pair, test_pair)

            # call function to save the transformation to model repository
            # for future retrieval
            ._save_transformation()
        )
        return self._get_pairs(train_pair, validation_pair, test_pair)

    def _get_pairs(self, train_pair, validation_pair, test_pair) -> Pairs:
        return Pairs(train_pair, validation_pair, test_pair)


class DataTransformLazyCall(TabularDataTransform):
    def __init__(self, column: TabularColumn, repository: DummyMLflowRepository):
        self.train_pair: Optional[FeatureTargetPair] = None
        self.valid_pair: Optional[FeatureTargetPair] = None
        self.test_pair: Optional[FeatureTargetPair] = None
        self.column = column

        self.transformed_data: Optional[pd.DataFrame] = None
        self.save_function_container = []
        self.keeper_array = []

        self.repository = repository
        self.transformation_save_function_container = []
    
    def get_pairs(self) -> Pairs:
        return Pairs(self.train_pair, self.valid_pair, self.test_pair)

    def add_log_transformation(self, column, methods: TransformationMethods):
        '''
        Add log transformation to a column
        it does not need transform and fit like min max because
        it does not need any anchoring point
        '''
        transformation_name = "log"
        if column not in self.column:
            raise ValueError(f"column {column} not exist")
        if column not in self.column.numerical():
            raise ValueError(f"column {column} is not numerical")
        pw = ProcessWrapper(np.log, np.exp)
        keeper = Keeper(
            name=transformation_name,
            column= column, 
            function= pw, 
            methods= methods)
        self.keeper_array.append(keeper)
        def save():
            self.repository.save_transformation(
                func=pw,
                parent_run_id=self.repository.get_parent_run_id(),
                column=column,
                transformation_name=transformation_name,
            )
        self.transformation_save_function_container.append(save)
        return self

    def add_min_max_transformation(self, column, methods: TransformationMethods):
        '''
        Add min max transformation to a column
        it does not need transform and fit like min max because
        it does not need any anchoring point
        '''
        transformation_name = "min_max"
        if column not in self.column:
            raise ValueError(f"column {column} not exist")
        if column not in self.column.numerical():
            raise ValueError(f"column {column} is not numerical")
        keeper = Keeper(
            name=transformation_name,
            column= column, 
            function= MinMaxScaler(), 
            methods= methods
            )
        self.keeper_array.append(keeper)
        def save():
            self.repository.save_transformation(
                func=keeper.function,
                parent_run_id=self.repository.get_parent_run_id(),
                column=column,
                transformation_name=transformation_name,
            )
        self.transformation_save_function_container.append(save)
        return self

    def add_one_hot_encoding_transformation(self, column):
        '''
        Add one hot encoding transformation to a column
        it only offers transformation methods of append and remove
        '''
        transformation_name = "one_hot_encoding"
        if column not in self.column:
            raise ValueError(f"column {column} not exist")
        if column not in self.column.categorical():
            raise ValueError(f"column {column} is not categorical")
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist')
        keeper = Keeper(
            name=transformation_name,
            column= column, 
            function= ohe, 
            methods= TransformationMethods.APPEND_AND_REMOVE
            )
        self.keeper_array.append(keeper)
        def save():
            self.repository.save_transformation(
                func=ohe,
                parent_run_id=self.repository.get_parent_run_id(),
                column=column,
                transformation_name=transformation_name,
            )
        self.transformation_save_function_container.append(save)
        return self

    def _setup_transformation(self):
        '''
        Fitting all transformation to the training data
        the transformation is not in this function
        '''
        for keeper in self.keeper_array:
            if keeper.column in self.train_pair.X.columns:
                keeper.function.fit(self.train_pair.X[keeper.column].to_numpy().reshape(-1, 1))
            else:
                keeper.function.fit(self.train_pair.y.to_numpy().reshape(-1, 1))
        return self


    def _replace(self, keeper):
        '''
        replace the original column with the transformed column 
        '''
        for pairs in [self.train_pair, self.valid_pair, self.test_pair]:
            if keeper.column in self.train_pair.X.columns:
                transformed = keeper.function.transform(pairs.X[keeper.column].to_numpy().reshape(-1, 1))
                pairs.X[keeper.column] = transformed
            else:
                pairs.y = pd.DataFrame(keeper.function.transform(pairs.y.to_numpy().reshape(-1,1)))
            return self

    def _append(self, keeper):
        '''
        append the transformed column and keep the original column
        '''
        for pairs in [self.train_pair, self.valid_pair, self.test_pair]:
            if keeper.column not in pairs.X.columns:
                raise ValueError(f"column {keeper.column} is not a feature")
            transformed = keeper.function.transform(self.train_pair.X[keeper.column].to_numpy().reshape(-1, 1))
            pairs.X[keeper.column] = transformed
        return self

    def _append_and_remove(self, keeper):
        '''
        append all the new columns and remove the original column
        '''
        for pairs in [self.train_pair, self.valid_pair, self.test_pair]:
            if keeper.column not in pairs.X.columns:
                raise ValueError(f"column {keeper.column} is not a feature")

            # Transform and create new columns
            transformed = keeper.function.transform(pairs.X[[keeper.column]])
            encoded_columns = keeper.function.get_feature_names_out([keeper.column.name])
            new_columns = pd.DataFrame(transformed, columns=encoded_columns, dtype='int', index=pairs.X.index)

            # Remove original and concatenate new columns
            pairs.X = pd.concat([
                pairs.X.drop(columns=[keeper.column]), 
                new_columns
            ], axis=1)
        return self

    def _applies_transformation_to_all(self):
        '''
        applies all transformation to all stages based on the training data
        '''
        for keeper in self.keeper_array:
            if keeper.methods == TransformationMethods.REPLACE:
                return self._replace(keeper)
            elif keeper.methods == TransformationMethods.APPEND:
                return self._append(keeper)
            elif keeper.methods == TransformationMethods.APPEND_AND_REMOVE:
                return self._append_and_remove(keeper)

    def transform_data(self, df):
        self.transformed_data = df
        (self
            # save the transformation manifest
            # contain information about the model input
            ._save_manifest()

            # all data should be splitted first here
            # so all encoding and transformation is based on training data
            # to prevent learning leakage. All transformation happen between
            # splitting and reapplied
            ._split_stage()

            # call all collected transformation based on train data but
            # does not applies it to train immdiately
            ._setup_transformation()

            # call all transformation function and applies it to all stages
            ._applies_transformation_to_all()

            # ensure that every stage has the same column count
            ._shape_check()

            # call function to save the transformation to model repository
            # for future retrieval
            ._save_transformation()
        )

        return self.get_pairs()

    def _get_pairs(self) -> Pairs:
        return Pairs(self.train_pair, self.valid_pair, self.test_pair)

    def _split_stage(self):
        if self.transformed_data is None:
            raise ValueError("transformed_data should exist")

        train_cols = self.column.feature(self.transformed_data.columns)
        train = self.transformed_data[train_cols]
        test = self.transformed_data[self.column.target()]
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
        self.train_pair = FeatureTargetPair(Xtr, ytr, Stage.TRAIN)
        self.valid_pair = FeatureTargetPair(Xv, yv, Stage.VALID)
        self.test_pair = FeatureTargetPair(Xte, yte, Stage.TEST)
        return self

    def _shape_check(self):
        if self.train_pair is None:
            raise ValueError("train pair should exist")
        if self.test_pair is None:
            raise ValueError("train pair should exist")
        if self.valid_pair is None:
            raise ValueError("train pair should exist")
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

    def _save_transformation(self):
        for save_fn in self.transformation_save_function_container:
            save_fn()
        self._save_manifest()
        return self
    
    def _save_manifest(self):
        df = self.transformed_data
        manifest = []
        for column in df.columns:
            if column == self.column.target():
                continue
            if column in self.column.categorical():
                manifest.append(Manifest.create_categorical_input_item(
                    key=column,
                    enumeration=df[column].unique().tolist()
                ).to_dict())
            elif column in self.column.numerical():
                manifest.append(Manifest.create_numerical_input_item(
                    key=column,
                    min=df[column].min(),
                    max=df[column].max()
                ).to_dict())
        self.repository.save_transformation_manifest(
            manifest=manifest,
            parent_run_id=self.repository.get_parent_run_id()
        )
        return self


class DataTransformDeprecated(TabularDataTransform):
    def __init__(self, enum: TabularColumn):
        self.ohe = {}
        self.mm = {}
        self.transformed_data = None

        self.run_name = None

        self.train_pair = None
        self.valid_pair = None
        self.test_pair = None
        self.enum = enum

    def transform_data(self, df) -> Pairs:
        self.transformed_data = df
        (self
            # all data should be splitted first here
            # so all encoding and transformation is based on training data
            # to prevent learning leakage. All transformation happen between
            # splitting and reapplied
            .split_stage()
            
            # each enum should provide which column is a category
            # so this function only call that function and match it
            # with the repsective column

            # dataframe should already use enum based column
            # see data loader for details
            .one_hot_encoding()

            # each enum should provide which column is a numeric
            # so this function only call that function and match it
            # with the repsective column
            # dataframe should already use enum based column
            # see data loader for details
            .minmax()

            # all those transformation now applied to both
            # valid and test data
            .reapply()
            
            # before we proceed, we need to check the shape
            # this is to ensure that rows does not get modified
            # and check column expansion due to one hot encoding
            # all train, valid, and test should have same column count
            .shape_check()

            # we need to save the tranformation because all future incoming
            # inference use the same data structure as raw data
            # in order our model to perform, it need to be transformed first.
            # we save the transformation to a pickle; saved in mlflow instance
            # we will load those instance later when we serve the model as inference
            .save_transformation()
         )

        return self.get_pairs()

    # while the transformation is not a training, we still need to collect
    # artifacts for transformation therefore we need to know the run name
    def set_run_name(self, name: str):
        self.run_name = name
        return self
    
    # pairs always return three part that later accessed by train
    # the type is not particularly important as long as it has
    # feature, target, and stage (train, valid, or test)
    def get_pairs(self) -> Pairs:
        return Pairs(self.train_pair, self.valid_pair, self.test_pair)

    def print_shapes(self):
        if self.train_pair is None:
            raise ValueError("train pair should exist")
        if self.test_pair is None:
            raise ValueError("train pair should exist")
        if self.valid_pair is None:
            raise ValueError("train pair should exist")
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
        return self

    # Currently we dont have insight about the best practice for saving
    # artifacts. Based on our knowledge, we need to store it based on
    # the column. Some might column require more transformation than others
    # current available transformation are one hot encoding and min max scale
    # stored as a map. If we have other transformation, we will add more key.
    # while it not scalable, it is the best we have.
    def save_transformation(self):
        dir = "artifacts/preprocess"
        for key, value in self.ohe.items():
            # TODO should find another way to represent that something stored as ohe
            # or other transformation methods; working with magic string is risky
            self.save_preprocessing_as_object(dir, "ohe", key, value)
        for key, value in self.mm.items():
            self.save_preprocessing_as_object(dir, "mm", key, value)
        # this function transfer all file under dir to mlflow artifact folder
        # called preprocess. Better than upload individual file
        # for testing, we only test the saving folder part
        # TODO maybe we can use mock testing for all mlflow methods
        mlflow.log_artifacts(dir, artifact_path="preprocess")
        return self

    def save_preprocessing_as_object(self, dir, prefix, key, obj):
        os.makedirs(f'{dir}/{key}', exist_ok=True)
        path = f'{dir}/{key}/{prefix}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        return path

    def shape_check(self):
        if self.train_pair is None:
            raise ValueError("train pair should exist")
        if self.test_pair is None:
            raise ValueError("train pair should exist")
        if self.valid_pair is None:
            raise ValueError("train pair should exist")
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
        if self.transformed_data is None:
            raise ValueError("transformed_data should null")

        train_cols = self.enum.feature(self.transformed_data.columns)
        train = self.transformed_data[train_cols]
        test = self.transformed_data[self.enum.target()]
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
        self.train_pair = FeatureTargetPair(Xtr, ytr, Stage.TRAIN)
        self.valid_pair = FeatureTargetPair(Xv, yv, Stage.VALID)
        self.test_pair = FeatureTargetPair(Xte, yte, Stage.TEST)
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
        # need to use cat.bane for feature names out becuase append operation
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

    # some column might get dropped in data cleaning
    # there fore we need to find intersect between
    # current column and the categorical column
    def _catcol(self, pair):
        return list(set(
            pair.X.columns) & 
            set(self.enum.categorical())
        )
    # same as above but for numerical columns
    def _numcol(self, pair):
        return list(set(
            pair.X.columns) & 
            set(self.enum.numerical())
        )

    def reapply(self):
        for pair in [self.valid_pair, self.test_pair]:
            # we need to validate that the column exist in the dataframe
            # and the column is being transformed
            # putting column that not transformed would raise KeyError
            for cat in list(set(self._catcol(pair)) & set(self.ohe.keys())):
                self._transform_ohe(pair, cat)
            for num in list(set(self._numcol(pair)) & set(self.mm.keys())):
                self._transform_mm(pair, num)
        return self
