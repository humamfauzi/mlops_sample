import pandas as pd
from typing import List
from enum import Enum
import numpy as np

from sklearn.metrics import mean_squared_error
from repositories.repo import Facade
from .data_io import Disk
from .data_cleaner import Cleaner
from dataclasses import dataclass
from sklearn import metrics as mm
# TODO: should be generalized for all tabular column
from column.cfs2017 import TabularColumn

class TransformationMethods(Enum):
    # would replace the original column with the transformation
    REPLACE = 1
    # would append the transformation to the original column; the original
    # column would still exist
    APPEND = 2

    # would append the transformation to the original column and remove the original
    APPEND_AND_REMOVE = 3
@dataclass
class Config:
    intent: str
    check_against: str
    n_samples: int
    metrics: List[str]
    seed: int

@dataclass
class Transformation:
    available_input: List[str]
    transformation: object
    itransformation: object

@dataclass
class Inference:
    transformations: Transformation
    model: callable


class PostTest:
    # TODO: should support multiple cleaner
    def __init__(self, 
        configs: List[Config], 
        facade: Facade, 
        loader: Disk, 
        cleaner: Cleaner, 
        column_reference: TabularColumn):
        self.facade = facade
        self.loader = loader
        self.configs = configs
        self.cleaner = cleaner
        self.column = column_reference
        pass

    @classmethod
    def parse_instruction(cls, properties: dict, call: List[dict],cleaner: Cleaner,  facade: Facade ):
        configs = []
        for c in call:
            configs.append(Config(
                intent=c.get("intent", ""),
                check_against=c.get("check_against", "actual_value"),
                n_samples=c.get("n_samples", 1000),
                metrics=c.get("metrics", ["mse", "rmse"]),
                seed=c.get("seed", 42)
            ))
        column = TabularColumn.from_string(properties.get("reference"))
        loader = Disk(facade, properties.get("path", ""), properties.get("file", ""))
        loader.load_random_rows_via_csv(
            column=column,
            n_rows=properties.get("n_rows", 1000),
            random_state=properties.get("random_state", 42),
            load_options={}
        )
        return cls(
            configs=configs,
            facade=facade,
            loader=loader,
            cleaner=cleaner,
            column_reference=column
        )

    def reconstruct_inference(self, run_id: str):
        '''
        Reconstruct the inference machine from the saved run
        Similar to the construct method in server/transformation.py and server/model.py

        :param self: self
        :param run_id: main run id of the desired inference machine
        :type run_id: str
        '''
        instructions = self.facade.load_transformation_instruction(run_id)
        transformations, itransformations = [], []
        for step in instructions:
            fobject = self.facade.load_transformation_object(run_id, step.id, step.type)
            if step.inverse_transform:
                func = getattr(fobject.object, "inverse_transform", None)
                if callable(func):
                    itransformations.append({
                        "name": step.name,
                        "column": step.column,
                        "method": step.method,
                        "function": func
                    })
            else:
                func = getattr(fobject.object, "transform", None)
                get_feature_names_out = getattr(fobject.object, "get_feature_names_out", None)
                if callable(func):
                    transformations.append({
                        "name": step.name,
                        "column": step.column,
                        "method": step.method,
                        "function": func,
                        "feature_names": get_feature_names_out,
                    })
        transformation = Transformation(
            available_input=self.facade.get_available_input(run_id),
            transformation=transformations,
            itransformation=itransformations
        )

        model = self.facade.get_model_best_model(run_id).object
        return Inference(transformations=transformation, model=model)

    def pick_random_samples(self) -> pd.DataFrame:
        return self.cleaner.execute(self.loader.execute(None))

    def check(self, inference: Inference, samples: pd.DataFrame, metrics: List[str]) -> dict:
        actual_outcome = samples[self.column.target()].copy()
        transformed = samples[[ai.name for ai in inference.transformations.available_input]].copy()
        for transformation in inference.transformations.transformation:
            column = transformation["column"]
            if column in self.column.categorical():
                input = transformed[column].astype(str)
            else:
                input = transformed[column].astype(float)
            method = transformation.get("method", "")
            if method == TransformationMethods.REPLACE.name:
                # TODO: need to do something better than this to convert the type
                transformed[column] = transformed[column].astype(float)
                transformed.loc[:, column] = transformation["function"](input.to_numpy().reshape(-1, 1))
            elif method == TransformationMethods.APPEND.name:
                appended = transformation["function"](input.to_numpy().reshape(-1, 1))
                transformed[f"{column}_{transformation['name']}"] = appended
            elif method == TransformationMethods.APPEND_AND_REMOVE.name:
                appended = transformation["function"](input.to_numpy().reshape(-1, 1))
                encoded_columns = transformation["feature_names"]([column])
                new_columns = pd.DataFrame(appended, columns=encoded_columns, dtype='int', index=transformed.index)
                transformed = pd.concat([transformed.drop(column, axis=1), new_columns], axis=1)
        result = inference.model.predict(transformed)
        for it in inference.transformations.itransformation:
            func = it["function"]
            column = it["column"]
            result = func(result)
        return {
            metric: self.metric_map()[metric](actual_outcome, result) for metric in metrics
        }

    def metric_map(self):
        return {
            "mse": mean_squared_error,
            "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": lambda y_true, y_pred: np.mean(np.abs(y_true - y_pred))
        }

    def store_metrics(self, result: dict):
        for metric, value in result.items():
            print(f"Post Test Metric {metric}: {value}")
            self.facade.set_metric("post_test", metric, value)

    def execute(self, _ : any):
        '''
        execute callable is the standard scenario manager interface
        any defined module in the traning configs should have this function
        so scenario manager could execute and call the intended function
        '''
        run_id = self.facade.current_run_id
        if run_id == "":
            raise ValueError("Post Test require reference run id. Should be main run_id")

        for pt in self.configs:
            self.facade.set_post_test_row_size(run_id, pt.n_samples)
            self.facade.set_post_test_intent(run_id, pt.intent)

            inference_machine = self.reconstruct_inference(run_id)
            samples = self.pick_random_samples()
            result = self.check(inference_machine, samples, pt.metrics)
            self.store_metrics(result)

        # return the original object
        return object

