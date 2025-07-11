from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

class TransferLearningClassifier:
    def __init__(self, x_train, x_test, y_train, y_test, num_classes):
        super(TransferLearningClassifier, self).__init__()
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = to_categorical(y_train, num_classes)
        self.y_test = to_categorical(y_test, num_classes)
        self.num_classes = num_classes

        # Create base model
        base_model = MobileNetV2(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
        base_model.trainable = False  # Freeze base model

        # Add custom head
        inputs = Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        self.model = Model(inputs, outputs)
        self.model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        self.model.fit(self.x_train, self.y_train, epochs=5, batch_size=32, verbose=1)

    def run(self):
        preds = self.model.predict(self.x_test)
        return preds.argmax(axis=1)

    def get_proba(self):
        return self.model.predict(self.x_test)