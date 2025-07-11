from sklearn.neighbors import KNeighborsClassifier

class KNNClassifier:
    def __init__(self, x_train, x_test, y_train, y_test):
        super(KNNClassifier, self).__init__()
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        self.clf = KNeighborsClassifier(n_neighbors=5)  # bạn có thể thay đổi số lượng hàng xóm
        self.clf = self.clf.fit(self.x_train, self.y_train)

    def run(self):
        return self.clf.predict(self.x_test)

    def get_proba(self):
        return self.clf.predict_proba(self.x_test)