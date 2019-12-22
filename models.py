import keras
from classification_models import Classifiers
from keras import Sequential, Model
from keras.applications import DenseNet121
from keras.callbacks import Callback
from sklearn.metrics import cohen_kappa_score
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras_applications.resnext import ResNeXt50, preprocess_input

from keras import backend as K


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):
        X_val, y_val = self.validation_data[:2]
        y_val = y_val.sum(axis=1) - 1

        y_pred = self.model.predict(X_val) > 0.5
        y_pred = y_pred.astype(int).sum(axis=1) - 1

        _val_kappa = cohen_kappa_score(
            y_val,
            y_pred,
            weights='quadratic'
        )

        self.val_kappas.append(_val_kappa)

        print(f"val_kappa: {_val_kappa:.4f}")

        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save('model.h5')

        return


def build_model():
    densenet = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    model = Sequential()
    model.add(densenet)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(102, activation='sigmoid'))

    return model


def preprocessing(x):
    return preprocess_input(x, backend=keras.backend,
                            layers=keras.layers,
                            models=keras.models,
                            utils=keras.utils)


def patch_resnext50():
    resnext50 = ResNeXt50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3),
        backend=keras.backend,
        layers=keras.layers,
        models=keras.models,
        utils=keras.utils
    )

    model = Sequential()
    model.add(resnext50)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(6, activation='sigmoid'))

    return model


def patch_resnet18():
    resnet18, _ = Classifiers.get('resnet18')

    base_model = resnet18(input_shape=(112, 112, 3), include_top=False)
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(6, activation='softmax')(x)
    model = Model(inputs=[base_model.input], outputs=[output])

    return model
