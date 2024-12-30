import mlflow
from train.data_loader import TabularDataLoader
# all run initiate here
# all run component should be included here and called in
# desired scenario
class ScenarioManager:
    def __init__(self):
        self.dataloader = None
        self.datacleaner = None
        self.datatransform = None
        self.model = None
        self.run_name = None
        self.is_using_tracker = False
        return

    def set_run_name(self, name):
        self.run_name = name
        return self

    def set_dataloader(self, dataloader: TabularDataLoader):
        self.dataloader = dataloader
        return self

    def set_datacleaner(self, datacleaner):
        self.datacleaner = datacleaner
        return self

    def set_datatransform(self, datatransform):
        self.datatransform = datatransform
        self.datatransform.set_run_name(self.run_name)
        return self

    def set_train(self, train):
        self.model = train
        self.model.set_run_name(self.run_name)
        self.model.set_tracker(self.is_using_tracker)
        return self

    def set_tracking(self, path, name):
        self.tracking_path = path
        self.experiment_name = name
        mlflow.set_tracking_uri(uri=self.tracking_path)
        mlflow.set_experiment(self.experiment_name)
        return self

    def start_run(self, run_name):
        mlflow.start_run(run_name=run_name)
        self.is_using_tracker = True
        return self

    def end_run(self):
        run = mlflow.active_run()
        mlflow.end_run()

    def default_path(self):
        df = self.dataloader.load_data()
        df_cleaned = self.datacleaner.clean_data(df)
        pairs = self.datatransform.transform_data(df_cleaned)
        self.model.train_data(pairs)
        return self
