import time
import json
import pandas as pd
import numpy as np
import os 
import pickle

from abc import ABC, abstractmethod
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
from column.cfs2017 import TabularColumn
from train.sstruct import Pairs, Stage, FeatureTargetPair
from train.wrapper import ProcessWrapper
from repositories.struct import TransformationObject, TransformationInstruction, TransformationMethods
from dataclasses import dataclass
from typing import  List

class TabularDataTransform(ABC):
    @abstractmethod
    def transform_data(self, df: pd.DataFrame) -> Pairs:
        pass

@dataclass
class Keeper:
    # The collector of the transformation
    # Collect all neccessary transformation properties so it could be
    # reproduced later by the server
    id: str
    name: str
    column: TabularColumn
    function: any
    method: TransformationMethods
    inverse_transform: bool
    type: str
    options: dict = None

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "column": self.column,
            "method": self.method.name,
            "inverse_transform": self.inverse_transform,
            "type": self.type
        }

class Transformer:
    def __init__(self, facade, column: TabularColumn):
        self.column = column
        self.keepers = []
        self.facade = facade

    def keeper_builder(self,typee,  column, col, condition, count):
        c = column.from_enum(col).name

        if typee == "one_hot_encoding" and condition != TransformationMethods.APPEND_AND_REMOVE.name:
            # One-hot encoding always expands into new columns and drops the
            # original; a config asking for "replace" used to be accepted and
            # then silently ignored.
            raise ValueError(
                f"one_hot_encoding only supports condition "
                f"{TransformationMethods.APPEND_AND_REMOVE.name!r}, got {condition!r}"
            )

        inverse_transform = False
        if col.upper() == column.target():
            inverse_transform = True
        return {
            "log_transformation": lambda: Keeper(
                id = f"log-{count:02d}-{col}",
                name="log",
                column=c,
                function=ProcessWrapper(np.log, np.exp),
                method=TransformationMethods[condition],
                inverse_transform=inverse_transform,
                type="pkl",
            ),
            "normalization": lambda: Keeper(
                id=f"norm-{count:02d}-{col}",
                name="normalization",
                column=c,
                function=Normalizer(norm='l2'),
                method=TransformationMethods[condition],
                inverse_transform=inverse_transform,
                type="pkl",
            ),
            "min_max_transformation": lambda: Keeper(
                id=f"minmax-{count:02d}-{col}",
                name="min_max",
                column=c,
                function=MinMaxScaler(),
                method=TransformationMethods[condition],
                inverse_transform=inverse_transform,
                type="pkl",
            ),
            "one_hot_encoding": lambda: Keeper(
                id=f"ohe-{count:02d}-{col}",
                name="one_hot_encoding",
                column=c,
                function=OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist'),
                method=TransformationMethods.APPEND_AND_REMOVE,
                inverse_transform=inverse_transform,
                type="pkl",
            ),
            "standardization": lambda: Keeper(
                id=f"std-{count:02d}-{col}",
                name="standardization",
                column=c,
                function=StandardScaler(),
                method=TransformationMethods[condition],
                inverse_transform=inverse_transform,
                type="pkl",
            ),
        }[typee]()

    @classmethod
    def parse_instruction(cls, properties: dict, call: List[dict], facade):
        column = TabularColumn.from_string(properties.get("reference"))
        c = cls(facade, column)
        count = -1
        for step in call:
            count += 1
            for col in step["columns"]:
                keeper = c.keeper_builder(step["type"], column, col, step["condition"].upper(), count)
                c.keepers.append(keeper)
        return c

    def _save_manifest(self, df: pd.DataFrame):
        """Record the input contract the server will validate against.

        Every column surviving cleaning must be accounted for: either it is the
        target, it is the primary id, or it is classified as numerical or
        categorical. A column in none of those buckets used to be dropped
        silently, which produced a model the server could never feed -- the
        failure only appeared at inference time, long after training.

        The order comes from ``column.feature``, the same call that selects the
        training matrix, so the manifest cannot disagree with the column order
        the model was fitted on.
        """
        if self.facade is None:
            return self

        target = self.column.target()
        primary_id = self.column.primary_id()
        numerical = set(self.column.numerical())
        categorical = set(self.column.categorical())

        unclassified = [
            col for col in df.columns
            if col not in (target, primary_id)
            and col not in numerical
            and col not in categorical
        ]
        if unclassified:
            raise ValueError(
                f"Cannot build the transformation manifest: {unclassified} are "
                f"neither numerical nor categorical in "
                f"{type(self.column).__name__}. Classify them, or exclude them "
                f"with a filter_columns step. Dropping them silently would "
                f"produce a model the server cannot feed."
            )

        inputs = self.column.feature(df.columns)
        column_properties = []
        for col in inputs:
            if col in numerical:
                minn, maxx = df[col].min(), df[col].max()
                column_properties.append({
                    "name": col,
                    "type": "numerical",
                    "min": int(minn),
                    "max": int(maxx)
                })
            elif col in categorical:
                uniques = df[col].unique().tolist()
                column_properties.append({
                    "name": col,
                    "type": "categorical",
                    "available_values": uniques
                })
        self.facade.set_object_transformation("transformation.allowed_columns", json.dumps(column_properties))
        return self

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
        self._save_pairing_metadata(train_pair, valid_pair, test_pair)
        return train_pair, valid_pair, test_pair

    def _save_pairing_metadata(self, train_pair, valid_pair, test_pair):
        if self.facade is None:
            return self
        self.facade.set_pair_size("train", len(train_pair.X.index), len(train_pair.X.columns))
        self.facade.set_pair_size("valid", len(valid_pair.X.index), len(valid_pair.X.columns))
        self.facade.set_pair_size("test", len(test_pair.X.index), len(test_pair.X.columns))
        return self

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
        start = time.time()
        for keeper in self.keepers:
            start_loop = time.time()
            if keeper.method == TransformationMethods.REPLACE:
                self._replace(keeper, train_pair, validation_pair, test_pair)
            elif keeper.method == TransformationMethods.APPEND:
                self._append(keeper, train_pair, validation_pair, test_pair)
            elif keeper.method == TransformationMethods.APPEND_AND_REMOVE:
                self._append_and_remove(keeper, train_pair, validation_pair, test_pair)
            loop_time = int((time.time() - start_loop) * 1000)
            if self.facade is not None:
                self.facade.set_transformation_time(keeper.name, loop_time)
        time_ms = int((time.time() - start) * 1000)
        if self.facade is not None:
            self.facade.set_total_transformation_time(time_ms)
        return self

    def _replace(self, keeper, train_pair, validation_pair, test_pair):
        '''
        replace the original column with the transformed column 
        '''
        for pairs in [train_pair, validation_pair, test_pair]:
            if keeper.column in train_pair.X.columns:
                transformed = keeper.function.transform(pairs.X[keeper.column].to_numpy().reshape(-1, 1))
                pairs.X[keeper.column] = transformed
            else:
                inverse = getattr(keeper.function, "inverse_transform", None)
                if not callable(inverse):
                    err_msg = f"Transformation {keeper.name} does not support inverse_transform and tried to applied in column target"
                    raise ValueError(err_msg)
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
            pairs.X[f"{keeper.column}_{keeper.name}"] = transformed
        return self

    def _append_and_remove(self, keeper, train_pair, validation_pair, test_pair):
        '''
        append all the new columns and remove the original column
        '''
        for pairs in [train_pair, validation_pair, test_pair]:
            if keeper.column not in pairs.X.columns:
                raise ValueError(f"column {keeper.column} is not a feature")
            transformed = keeper.function.transform(pairs.X[[keeper.column]].to_numpy().reshape(-1, 1))
            encoded_columns = keeper.function.get_feature_names_out([keeper.column])
            new_columns = pd.DataFrame(transformed, columns=encoded_columns, dtype='int', index=pairs.X.index)
            pairs.X = pd.concat([
                pairs.X.drop(columns=[keeper.column]), 
                new_columns
            ], axis=1)
        return self

    def _shape_check(self, train_pair, validation_pair, test_pair):
        train_cols = train_pair.X.shape[1]
        valid_cols = validation_pair.X.shape[1]
        test_cols = test_pair.X.shape[1]

        if not (train_cols == valid_cols == test_cols):
            raise ValueError(
                f"Feature column count mismatch across splits: "
                f"train={train_cols}, valid={valid_cols}, test={test_cols}"
            )
        return self

    def _save_transformation(self):
        if self.facade is None:
            return self
        objects = [TransformationObject(filename=f"{k.id}.pkl", object=k.function) for k in self.keepers]
        self.facade.save_transformation_object(objects)

        instruction = [TransformationInstruction(**k.to_dict()) for k in self.keepers]
        self.facade.save_transformation_instruction(instruction)
        return self

    def execute(self, input_data: pd.DataFrame) -> Pairs:
        print("Starting data transformation process...")
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