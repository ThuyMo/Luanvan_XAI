from sklearn.dummy import DummyClassifier

class DummyBaseClassifier:
    def __init__(self, x_train, x_test, y_train, y_test):
        super(DummyBaseClassifier, self).__init__()
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        # Mô hình Dummy - chiến lược mặc định là "most_frequent"
        self.clf = DummyClassifier(strategy="most_frequent")
        self.clf = self.clf.fit(self.x_train, self.y_train)

    def run(self):
        return self.clf.predict(self.x_test)

    def get_proba(self):
        return self.clf.predict_proba(self.x_test)