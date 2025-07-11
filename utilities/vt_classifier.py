from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

class VotingClassifierModel:
    def __init__(self, x_train, x_test, y_train, y_test):
        super(VotingClassifierModel, self).__init__()
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        # Các mô hình thành phần
        clf1 = LogisticRegression(max_iter=1000)
        clf2 = DecisionTreeClassifier()
        clf3 = SVC(kernel='rbf', probability=True)  # Cần probability=True nếu dùng voting='soft'

        # VotingClassifier kết hợp các mô hình
        self.clf = VotingClassifier(
            estimators=[('lr', clf1), ('dt', clf2), ('svc', clf3)],
            voting='soft'  # 'soft' dùng xác suất, 'hard' dùng dự đoán đa số
        )

        self.clf = self.clf.fit(self.x_train, self.y_train)

    def run(self):
        return self.clf.predict(self.x_test)

    def get_proba(self):
        return self.clf.predict_proba(self.x_test)