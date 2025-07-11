from sklearn.linear_model import SGDClassifier

class SGDClassifierModel:
    def __init__(self, x_train, x_test, y_train, y_test):
        super(SGDClassifierModel, self).__init__()
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.clf = SGDClassifier(loss="log_loss")  # hoặc loss="hinge", tùy mục tiêu
        self.clf = self.clf.fit(self.x_train, self.y_train)

    def run(self):
        return self.clf.predict(self.x_test)

    def get_proba(self):
        return self.clf.predict_proba(self.x_test)