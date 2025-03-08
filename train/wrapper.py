
class PreprocessFitTransformWrapper:
    '''
    Using same interface like any other preprocess function in
    sklearn. So we can use it interchangably with other sklearn preprocess function
    without knowing what the actual function is.

    used for transformation without anchoring points like log, shift etc.

    while it does not use fit at all, it still would check whether the function is already fitted or not
    so it behaves like a normal sklearn preprocess function
    '''
    def __init__(self, function):
        self.function = function
        self.is_fitted = False

    def fit(self, X):
        self.is_fitted = True
        return self

    def transform(self, X):
        if not self.is_fitted:
            raise Exception("PreprocessFitTransformWrapper is not fitted yet")
        return self.function(X)

    def fit_transform(self, X, y=None):
        return self.function(X)

class ProcessWrapper:
    '''
    Same as PreprocessFitTransformWrapper need an inverse function

    while it does not use fit at all, it still would check whether the function is already fitted or not
    so it behaves like a normal sklearn preprocess function
    '''
    def __init__(self, function, refunction):
        self.function = function
        self.refunction = refunction
        self.is_fitted = False

    def fit(self, X):
        self.is_fitted = True
        return self

    def transform(self, X):
        if not self.is_fitted:
            raise Exception("ProcessWrapper is not fitted yet")
        return self.function(X)

    def inverse_transform(self, X):
        if not self.is_fitted:
            raise Exception("ProcessWrapper is not fitted yet")
        return self.refunction(X)