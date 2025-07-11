import pandas
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split

from utilities.mlp_classifier import MLPClassifier
from utilities.nb_classifier import NBClassifier
from utilities.rf_classifier import RFClassifier
from utilities.svmp_classifier import SVMPClassifier
from utilities.svmr_classifier import SVMRClassifier
from utilities.ab_classifier import AdaBoostModel
from utilities.bg_classifier import BaggingModel
from utilities.blstm_classifier import BiLSTMClassifier
from utilities.bnb_classifier import BNBClassifier
from utilities.cnb_classifier import CNBClassifier
from utilities.cnn_classifier import CNNClassifier
from utilities.dm_classifier import DummyBaseClassifier
from utilities.dt_classifier import DTClassifier
from utilities.ett_classifier import ETClassifier
from utilities.gb_classifier import GBTClassifier
from utilities.gru_classifier import GRUClassifier
from utilities.hgb_classifier import HGBTClassifier
from utilities.kn_classifier import KNNClassifier
from utilities.lda_classifier import LDAClassifier
from utilities.lr_classifier import LRClassifier
from utilities.lstm_classifier import LSTMClassifier
from utilities.mnb_classifier import MNBClassifier
from utilities.nsvc_classifier import NuSVCClassifier
from utilities.qda_classifier import QDAClassifier
from utilities.rc_classifier import RidgeClassifierModel
from utilities.rnn_classifier import RNNClassifier
from utilities.sgd_classifier import SGDClassifierModel
from utilities.st_classifier import StackingClassifierModel
from utilities.tfl_classifier import TransferLearningClassifier
from utilities.tfm_classifier import TransformerClassifier
from utilities.vt_classifier import VotingClassifierModel

class Classifier:
    def __init__(self, data, model, class_label):
        super(Classifier, self).__init__()
        self.data = data
        self.model = model
        self.class_label = class_label

    def _preprocess(self):
        le = preprocessing.LabelEncoder()
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        non_numerics = self.data.select_dtypes(exclude=numerics)
        for column in non_numerics:
            le.fit(non_numerics[column])
            self.data[column] = le.transform(non_numerics[column])
        self.data.fillna(0, inplace=True)

        scaler = preprocessing.MinMaxScaler()
        _columns = _features = [x for x in self.data.columns if x != self.class_label]
        self.data[_columns] = scaler.fit_transform(self.data[_columns])

    def run(self):
        self._preprocess()
        feature_cols = [f for f in self.data.columns if f != self.class_label]
        x = self.data[feature_cols]
        y = self.data[self.class_label]
          
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

        classifier = None
        if self.model == "NB":
            classifier = NBClassifier(x_train, x_test, y_train, y_test)
        elif self.model == "SVMP":
            classifier = SVMPClassifier(x_train, x_test, y_train, y_test)
        elif self.model == "SVMR":
            classifier = SVMRClassifier(x_train, x_test, y_train, y_test)
        elif self.model == "MLP":
            classifier = MLPClassifier(x_train, x_test, y_train, y_test)
        elif self.model == "RF":
            classifier = RFClassifier(x_train, x_test, y_train, y_test)
        elif self.model == "AB":
            classifier = AdaBoostModel(x_train, x_test, y_train, y_test)
        elif self.model == "BG":
            classifier = BaggingModel(x_train, x_test, y_train, y_test)
        elif self.model == "BLSTM":
            classifier = BiLSTMClassifier(x_train, x_test, y_train, y_test)
        elif self.model == "BNB":
            classifier = BNBClassifier(x_train, x_test, y_train, y_test)
        elif self.model == "CNB":
            classifier = CNBClassifier(x_train, x_test, y_train, y_test)
        elif self.model == "CNN":
            classifier = CNNClassifier(x_train, x_test, y_train, y_test)
        elif self.model == "DB":
            classifier = DummyBaseClassifier(x_train, x_test, y_train, y_test)
        elif self.model == "DT":
            classifier = DTClassifier(x_train, x_test, y_train, y_test)
        elif self.model == "ET":
            classifier = ETClassifier(x_train, x_test, y_train, y_test)
        elif self.model == "GBT":
            classifier = GBTClassifier(x_train, x_test, y_train, y_test)
        elif self.model == "GRU":
            classifier = GRUClassifier(x_train, x_test, y_train, y_test)
        elif self.model == "HGBT":
            classifier = HGBTClassifier(x_train, x_test, y_train, y_test)
        elif self.model == "KNN":
            classifier = KNNClassifier(x_train, x_test, y_train, y_test)
        elif self.model == "LDA":
            classifier = LDAClassifier(x_train, x_test, y_train, y_test)
        elif self.model == "LR":
            classifier = LRClassifier(x_train, x_test, y_train, y_test)
        elif self.model == "LSTM":
            classifier = LSTMClassifier(x_train, x_test, y_train, y_test)
        elif self.model == "MNB":
            classifier = MNBClassifier(x_train, x_test, y_train, y_test)
        elif self.model == "NuSVC":
            classifier = NuSVCClassifier(x_train, x_test, y_train, y_test)
        elif self.model == "QDA":
            classifier = QDAClassifier(x_train, x_test, y_train, y_test)
        elif self.model == "Ridge":
            classifier = RidgeClassifierModel(x_train, x_test, y_train, y_test)
        elif self.model == "RNN":
            classifier = RNNClassifier(x_train, x_test, y_train, y_test)
        elif self.model == "SGD":
            classifier = SGDClassifierModel(x_train, x_test, y_train, y_test)
        elif self.model == "Stacking":
            classifier = StackingClassifierModel(x_train, x_test, y_train, y_test)
        elif self.model == "TFL":
            classifier = TransferLearningClassifier(x_train, x_test, y_train, y_test, y_train.nunique())
        elif self.model == "TF":
            classifier = TransformerClassifier(x_train, x_test, y_train, y_test)
        elif self.model == "VT":
            classifier = VotingClassifierModel(x_train, x_test, y_train, y_test)

        y_pred = classifier.run()
        accuracy = metrics.accuracy_score(y_test, y_pred)

        classified_data = x_test
        labels = y_pred

        proba = classifier.get_proba()
        probabilities = []
        for proba in proba:
            probabilities.append(round(max(proba), 2))

        classified_data["_class"] = labels
        classified_data["_confidence"] = probabilities
        feature_names = [x for x in classified_data.columns if x not in ["_class", "_confidence"]]
        print(feature_names)

        return accuracy, classified_data, feature_names