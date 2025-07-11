import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical

class RNNClassifier:
    def __init__(self, x_train, x_test, y_train, y_test, num_classes=None):
        super(RNNClassifier, self).__init__()
        self.x_train = np.expand_dims(x_train, axis=1)  # (samples, timesteps=1, features)
        self.x_test = np.expand_dims(x_test, axis=1)
        self.y_train = to_categorical(y_train)
        self.y_test = y_test
        self.num_classes = self.y_train.shape[1] if num_classes is None else num_classes

        self.model = Sequential()
        self.model.add(SimpleRNN(64, input_shape=(self.x_train.shape[1], self.x_train.shape[2]), activation='relu'))
        self.model.add(Dense(self.num_classes, activation='softmax'))

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.x_train, self.y_train, epochs=20, batch_size=16, verbose=0)

    def run(self):
        predictions = self.model.predict(self.x_test)
        return np.argmax(predictions, axis=1)

    def get_proba(self):
        return self.model.predict(self.x_test)