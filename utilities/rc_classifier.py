from sklearn.linear_model import RidgeClassifier

class RidgeClassifierModel:
    def __init__(self, x_train, x_test, y_train, y_test):
        super(RidgeClassifierModel, self).__init__()
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.clf = RidgeClassifier()
        self.clf = self.clf.fit(self.x_train, self.y_train)

    def run(self):
        return self.clf.predict(self.x_test)

    def get_proba(self):
        # RidgeClassifier không hỗ trợ predict_proba nên dùng decision_function
        return self.clf.decision_function(self.x_test)