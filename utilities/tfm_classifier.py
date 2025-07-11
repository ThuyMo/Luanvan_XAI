import numpy as np
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

class TransformerClassifier:
    def __init__(self, x_train, x_test, y_train, y_test):
        super(TransformerClassifier, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_len = 128  # tùy chỉnh độ dài chuỗi

        self.x_train = self._tokenize(x_train)
        self.x_test = self._tokenize(x_test)

        self.y_train = tf.keras.utils.to_categorical(y_train)
        self.y_test_raw = y_test

        # Xây mô hình Transformer
        bert = TFBertModel.from_pretrained("bert-base-uncased")
        input_ids = Input(shape=(self.max_len,), dtype=tf.int32, name="input_ids")
        attention_mask = Input(shape=(self.max_len,), dtype=tf.int32, name="attention_mask")

        outputs = bert(input_ids, attention_mask=attention_mask)[1]  # pooled output
        outputs = Dense(self.y_train.shape[1], activation="softmax")(outputs)

        self.model = Model(inputs=[input_ids, attention_mask], outputs=outputs)
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=3,
            batch_size=16,
            verbose=1,
        )

    def _tokenize(self, texts):
        # texts phải là list of strings
        tokens = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="tf",
        )
        return {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]}

    def run(self):
        preds = self.model.predict(self.x_test)
        return np.argmax(preds, axis=1)

    def get_proba(self):
        return self.model.predict(self.x_test)