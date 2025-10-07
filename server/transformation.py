from repositories.repo import Facade

class Transformation:
    def __init__(self, repository: Facade):
        self.repository: Facade = repository
        pass