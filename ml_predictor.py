from sklearn import linear_model

class MlPredictor:
    """
    Setup a new machine learning predictor.

    :param data: preprocessed training data
    """
    def __init__(self, data):
        coded_data = data.replace('a', 0).replace('b',1)
        paths = coded_data.iloc[:,1:]
        scores = coded_data.iloc[:,0]
        self.predictor = linear_model.LinearRegression()
        self.predictor.fit(paths, scores)

    """
    Predict a user's risk tolerance.

    :param user_path: the path the user took as dataframe
    """
    def predict(self, user_path):
        return list(self.predictor.predict(user_path.replace('a',0).replace('b',1)))
