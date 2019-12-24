import keras
from classification_models import Classifiers
from keras import Sequential, Model
from keras.applications import DenseNet121
from keras.callbacks import Callback
from sklearn.metrics import cohen_kappa_score
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras_applications.resnext import ResNeXt50, preprocess_input
from keras_applications.efficientnet import EfficientNetB4

from keras import backend as K


def densenet121():
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


def efficientnetb4(target_size):
    efficientnet = EfficientNetB4(
        weights='imagenet',
        include_top=False,
        input_shape=(target_size, target_size, 3),
        backend=keras.backend,
        layers=keras.layers,
        models=keras.models,
        utils=keras.utils
    )

    model = Sequential()
    model.add(efficientnet)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.7))
    model.add(Dense(102, activation='sigmoid'))

    return model


def patch_resnet18():
    resnet18, _ = Classifiers.get('resnet18')

    base_model = resnet18(input_shape=(112, 112, 3), include_top=False)
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(6, activation='softmax')(x)
    model = Model(inputs=[base_model.input], outputs=[output])

    return model
