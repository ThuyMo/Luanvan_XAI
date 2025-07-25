from sklearn.naive_bayes import GaussianNB
import pickle

class NBClassifier:
    def __init__(self, x_train, x_test, y_train, y_test):
        super(NBClassifier, self).__init__()
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.clf = GaussianNB()
        self.clf = self.clf.fit(self.x_train, self.y_train)
        # with open(f'/content/Luanvan_XAI/models/{self.clf.__class__.__name__}.pkl', 'wb') as file_name:
        #   pickle.dump(self.clf, file_name)

    def run(self):
        return self.clf.predict(self.x_test)

    def get_proba(self):
        return self.clf.predict_proba(self.x_test)