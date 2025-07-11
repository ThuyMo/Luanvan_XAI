import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

class CNNClassifier:
    def __init__(self, x_train, x_test, y_train, y_test, input_shape=None, num_classes=None):
        super(CNNClassifier, self).__init__()
        
        # Nếu chưa xác định thì tự động xác định
        self.input_shape = input_shape or x_train.shape[1:]
        self.num_classes = num_classes or len(np.unique(y_train))

        # One-hot encode label nếu chưa encode
        self.y_train = to_categorical(y_train, self.num_classes)
        self.y_test = to_categorical(y_test, self.num_classes)
        
        self.x_train = x_train
        self.x_test = x_test

        # Xây dựng model CNN
        self.model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(self.x_train, self.y_train, epochs=10, batch_size=32, verbose=0)

    def run(self):
        pred = self.model.predict(self.x_test)
        return np.argmax(pred, axis=1)

    def get_proba(self):
        return self.model.predict(self.x_test)