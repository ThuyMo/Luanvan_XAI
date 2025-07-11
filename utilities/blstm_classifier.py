import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.utils import to_categorical

class BiLSTMClassifier:
    def __init__(self, x_train, x_test, y_train, y_test):
        super(BiLSTMClassifier, self).__init__()

        # Chuyển dữ liệu thành NumPy array
        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        # Định dạng input cho RNN: (samples, timesteps, features)
        self.x_train = np.expand_dims(x_train, axis=1)  # shape: (n_samples, 1, n_features)
        self.x_test = np.expand_dims(x_test, axis=1)

        # One-hot encoding cho y_train
        self.y_train = to_categorical(y_train)
        self.y_test_raw = y_test  # giữ lại y_test chưa encode để dùng sau

        # Xây dựng mô hình Bidirectional LSTM
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(64), input_shape=(1, self.x_train.shape[2])))
        self.model.add(Dense(self.y_train.shape[1], activation='softmax'))

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.x_train, self.y_train, epochs=20, batch_size=16, verbose=0)

    def run(self):
        y_pred = self.model.predict(self.x_test)
        return np.argmax(y_pred, axis=1)

    def get_proba(self):
        return self.model.predict(self.x_test)