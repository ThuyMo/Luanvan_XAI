import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

class LSTMClassifier:
    def __init__(self, x_train, x_test, y_train, y_test):
        super(LSTMClassifier, self).__init__()

        # Reshape input to (samples, timesteps=1, features)
        self.x_train = np.expand_dims(x_train, axis=1)
        self.x_test = np.expand_dims(x_test, axis=1)

        # One-hot encode labels
        self.y_train = to_categorical(y_train)
        self.y_test_raw = y_test  # keep raw labels for run()

        self.model = Sequential()
        self.model.add(LSTM(64, input_shape=(self.x_train.shape[1], self.x_train.shape[2]), activation='tanh'))
        self.model.add(Dense(self.y_train.shape[1], activation='softmax'))

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model.fit(self.x_train, self.y_train, epochs=20, batch_size=16, verbose=0)

    def run(self):
        y_pred = self.model.predict(self.x_test)
        return np.argmax(y_pred, axis=1)

    def get_proba(self):
        return self.model.predict(self.x_test)