# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


from a00_common_functions import *
import warnings
from keras.callbacks import Callback


class ModelCheckpoint_F2Score(Callback):
    """Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='max', period=1, patience=None, validation_data=()):
        super(ModelCheckpoint_F2Score, self).__init__()
        self.interval = period
        self.X_val, self.y_val = validation_data
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        if mode is 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            self.monitor_op = np.less
            self.best = np.Inf

        # part for early stopping
        self.epochs_from_best_model = 0
        self.patience = patience

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)

            y_pred = self.model.predict(self.X_val, verbose=0)
            score = 0
            for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                preds = y_pred.copy()
                preds[preds > thr] = 1
                preds[preds <= thr] = 0
                part_score = fbeta_score(self.y_val, preds.astype(np.uint8), beta=2, average='samples')
                print("F2Beta score: {:.6f} THR: {}".format(part_score, thr))
                score = max(score, part_score)

            if self.monitor_op(score, self.best):
                self.epochs_from_best_model = 0
            else:
                self.epochs_from_best_model += 1

            if self.save_best_only:
                current = score
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)

            if self.patience is not None:
                if self.epochs_from_best_model > self.patience:
                    print('Early stopping: {}'.format(self.epochs_from_best_model))
                    self.model.stop_training = True


def get_model_xception():
    from keras.models import Model
    from keras.applications.xception import Xception
    from keras.layers.core import Dense

    model = Xception(input_shape=(299, 299, 3), weights=None, include_top=False, pooling='avg')
    x = model.layers[-1].output
    x = Dense(7178, activation='sigmoid', name='predictions')(x)
    model = Model(inputs=model.input, outputs=x)
    return model


def get_model_resnet152():
    from keras.models import Model
    from base_nets.resnet152_v2 import resnet152_model_v2
    from keras.layers.core import Dense

    model = resnet152_model_v2(input_shape=(448, 448, 3), weights=None)
    x = model.layers[-2].output
    x = Dense(7178, activation='sigmoid', name='predictions')(x)
    model = Model(inputs=model.input, outputs=x)
    return model


def get_model_resnet50():
    from keras.models import Model
    from base_nets.resnet50_v2 import resnet50_model_v2
    from keras.layers.core import Dense

    model = resnet50_model_v2(input_shape=(224, 224, 3), weights=None)
    x = model.layers[-2].output
    x = Dense(7178, activation='sigmoid', name='predictions')(x)
    model = Model(inputs=model.input, outputs=x)
    return model


def get_model_resnet50_336():
    from keras.models import Model
    from base_nets.resnet50_v2 import resnet50_model_v2
    from keras.layers.core import Dense

    model = resnet50_model_v2(input_shape=(336, 336, 3), weights=None)
    x = model.layers[-2].output
    x = Dense(7178, activation='sigmoid', name='predictions')(x)
    model = Model(inputs=model.input, outputs=x)
    return model


def get_model_inception_resnet_v2():
    from keras.models import Model
    from keras.applications.inception_resnet_v2 import InceptionResNetV2
    from keras.layers.core import Dense

    model = InceptionResNetV2(input_shape=(299, 299, 3), weights=None)
    x = model.layers[-2].output
    x = Dense(7178, activation='sigmoid', name='predictions')(x)
    model = Model(inputs=model.input, outputs=x)
    return model


def get_model_mobilenet():
    from keras.models import Model
    from keras.applications.mobilenetv2 import MobileNetV2
    from keras.layers.core import Dense

    model = MobileNetV2(input_shape=(128, 128, 3), weights=None, alpha=1.0, depth_multiplier=1)
    x = model.layers[-2].output
    x = Dense(7178, activation='sigmoid', name='predictions')(x)
    model = Model(inputs=model.input, outputs=x)
    return model


def f2_mean_example(y_true, y_pred):
    from keras import backend as K
    agreement = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=1)
    true_positive = K.sum(K.round(K.clip(y_true, 0, 1)), axis=1)
    pred_positive = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=1)
    recall = agreement / (true_positive + K.epsilon())
    precision = agreement / (pred_positive + K.epsilon())
    f2score = (1+2**2)*((precision*recall)/(2**2*precision+recall+K.epsilon()))
    return K.mean(f2score)


def fbeta(y_true, y_pred, threshold_shift=0):
    from keras import backend as K
    beta = 2

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())


def f2beta_loss(Y_true, Y_pred):
    from keras import backend as K
    eps = 0.001
    false_positive = K.sum(Y_pred * (1 - Y_true), axis=-1)
    false_negative = K.sum((1 - Y_pred) * Y_true, axis=-1)
    true_positive = K.sum(Y_true * Y_pred, axis=-1)
    p = (true_positive) / (true_positive + false_positive + eps)
    r = (true_positive) / (true_positive + false_negative + eps)
    out = (5*p*r) / (4*p + r + eps)
    return -K.mean(out)


def preprocess_input(x):
    x /= 127.5
    x -= 1.
    return x