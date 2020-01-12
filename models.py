import keras
from classification_models import Classifiers
from keras import Sequential, Model
from keras.applications import DenseNet121
from keras.callbacks import Callback
from sklearn.metrics import cohen_kappa_score
from keras.layers import GlobalAveragePooling2D, Dropout, Dense, Flatten
from keras_applications.resnext import ResNeXt50, preprocess_input
from keras_applications.efficientnet import EfficientNetB4

from keras import backend as K


def densenet121():
    densenet = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    x = GlobalAveragePooling2D()(densenet.output)
    x = Dropout(0.5)(x)
    x = Dense(102, activation='softmax')(x)

    model = Model(densenet.input, x)

    return model


def preprocessing(x):
    return preprocess_input(x, backend=keras.backend,
                            layers=keras.layers,
                            models=keras.models,
                            utils=keras.utils)

def efficientnetb4():
    efficientnet = EfficientNetB4(
        weights='imagenet',
        include_top=False,
        input_shape=(380, 380, 3),
        backend=keras.backend,
        layers=keras.layers,
        models=keras.models,
        utils=keras.utils
    )

    model = Sequential()
    model.add(efficientnet)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(102, activation='softmax'))

    return model


def resnet18():
    resnet, _ = Classifiers.get('resnet18')
    base_model = resnet(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    output = Dense(102, activation='softmax')(x)
    model = Model(inputs=[base_model.input], outputs=[output])

    return model
