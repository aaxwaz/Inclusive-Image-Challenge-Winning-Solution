from a00_common_functions import *
import warnings
from keras.callbacks import Callback

class MyModelCheckpoint(Callback):
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
                 mode='auto', period=1):
        super(MyModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
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
                            #this_lr = str(K.eval(self.model.optimizer.lr))
                            #self.model.save_weights(filepath+'_currentLR_%s'%this_lr, overwrite=True)
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)

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
                 mode='max', period=1, patience=None, validation_data=(), rare_class_index=None):
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
        self.rare_class_index = rare_class_index
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
            filepath = self.filepath.format(epoch=epoch, **logs)

            y_pred = self.model.predict(self.X_val, verbose=0)
            score = 0
            best_tr = -1
            for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                if self.rare_class_index:
                    preds = y_pred[:, self.rare_class_index].copy()
                    preds[preds > thr] = 1
                    preds[preds <= thr] = 0
                    part_score = fbeta_score(self.y_val[:, self.rare_class_index], preds.astype(np.uint8), beta=2, average='samples')
                else:
                    preds = y_pred.copy()
                    preds[preds > thr] = 1
                    preds[preds <= thr] = 0
                    part_score = fbeta_score(self.y_val, preds.astype(np.uint8), beta=2, average='samples')
                print("F2Beta score: {:.6f} THR: {}".format(part_score, thr))
                if part_score > score:
                    best_tr = thr
                score = max(score, part_score)
                this_lr = str(K.eval(self.model.optimizer.lr))
            temp_file = '/'.join(filepath.split('/')[:-1]) + '/' + "Test:epoch_{}_F2Beta_score_{:.6f}_THR_{}_LR_{}.log".format(epoch, score, best_tr, this_lr)
            with open(temp_file, 'w') as f:
                f.write(' ')

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
                            this_lr = str(K.eval(self.model.optimizer.lr))
                            self.model.save_weights(filepath+'_currentLR_%s'%this_lr, overwrite=True)
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


class ModelCheckpoint_RMSE(Callback):
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
        super(ModelCheckpoint_RMSE, self).__init__()
        self.interval = period
        self.X_val, self.y_val = validation_data
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
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
            score = rmse(self.y_val, y_pred)
            print("RMSE score: {:.6f}".format(score))
            if score < self.best:
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

    model = resnet50_model_v2(input_shape=(299, 299, 3), weights=None)
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


def f2_mean_example(y_true, y_pred):
    from keras import backend as K
    agreement = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=1)
    true_positive = K.sum(K.round(K.clip(y_true, 0, 1)), axis=1)
    pred_positive = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=1)
    recall = agreement / (true_positive + K.epsilon())
    precision = agreement / (pred_positive + K.epsilon())
    f2score = (1+2**2)*((precision*recall)/(2**2*precision+recall+K.epsilon()))
    return K.mean(f2score)


### Weimin Model 
def get_model_NASNet():
    from keras.models import Model
    from keras.applications.nasnet import NASNetMobile
    from keras.layers.core import Dense

    print("\n\nUsing NASNetLarge\n\n")
    model = NASNetMobile(input_shape=(336, 336, 3), weights=None, include_top=False, pooling='avg')
    x = model.layers[-1].output
    x = Dense(7178, activation='sigmoid', name='predictions')(x)
    model = Model(inputs=model.input, outputs=x)
    return model

def get_model_densenet():
    from keras.models import Model
    from keras.applications.densenet import DenseNet121
    from keras.layers.core import Dense

    print("\n\nUsing DenseNet\n\n")
    model = DenseNet121(input_shape=(336, 336, 3), weights=None, include_top=False, pooling='avg')
    x = model.layers[-1].output
    x = Dense(7178, activation='sigmoid', name='predictions')(x)
    model = Model(inputs=model.input, outputs=x)
    return model

def get_model_densenet_top_543():
    from keras.models import Model
    from keras.applications.densenet import DenseNet121
    from keras.layers.core import Dense

    print("\n\nUsing DenseNet\n\n")
    model = DenseNet121(input_shape=(336, 336, 3), weights=None, include_top=False, pooling='avg')
    x = model.layers[-1].output
    x = Dense(543, activation='sigmoid', name='predictions')(x)
    model = Model(inputs=model.input, outputs=x)
    return model

def get_model_xception():
    from keras.models import Model
    from keras.applications.xception import Xception
    from keras.layers.core import Dense

    model = Xception(input_shape=(299, 299, 3), weights=None, include_top=False, pooling='avg')
    x = model.layers[-1].output
    x = Dense(7178, activation='sigmoid', name='predictions')(x)
    model = Model(inputs=model.input, outputs=x)
    return model

def get_model_xception_withMOE():
    from keras.models import Model
    from keras.applications.xception import Xception
    from keras.layers.core import Dense
    from moe_model import create_moe_layer

    model = Xception(input_shape=(336, 336, 3), weights=None, include_top=False, pooling='avg')
    x = model.layers[-1].output
    x = create_moe_layer(x)

    print(x)

    model = Model(inputs=model.input, outputs=x)
    return model
### Weimin Model 

def get_model_inception_resnet_v2():
    from keras.models import Model
    from keras.applications.inception_resnet_v2 import InceptionResNetV2
    from keras.layers.core import Dense

    model = InceptionResNetV2(input_shape=(299, 299, 3), weights=None)
    x = model.layers[-2].output
    x = Dense(7178, activation='sigmoid', name='predictions')(x)
    model = Model(inputs=model.input, outputs=x)
    return model

def get_model_inception_resnet_v2_resize(size=299):
    from keras.models import Model
    from keras.applications.inception_resnet_v2 import InceptionResNetV2
    from keras.layers.core import Dense

    model = InceptionResNetV2(input_shape=(size, size, 3), weights=None)
    x = model.layers[-2].output
    x = Dense(7178, activation='sigmoid', name='predictions')(x)
    model = Model(inputs=model.input, outputs=x)
    return model

def get_model_inception_resnet_v2_fv2():
    from keras.models import Model
    from keras.applications.inception_resnet_v2 import InceptionResNetV2
    from keras.layers.core import Dense

    model = InceptionResNetV2(input_shape=(299, 299, 3), weights=None, pooling='max', include_top=False)
    x = model.output
    x = Dense(7178, activation='sigmoid', name='predictions')(x)
    model = Model(inputs=model.input, outputs=x)
    return model

def get_model_inception_resnet_v2_fv1():
    from keras.models import Model
    from keras.applications.inception_resnet_v2 import InceptionResNetV2
    from keras.layers.core import Dense
    from keras.layers import Flatten 

    model = InceptionResNetV2(input_shape=(150, 150, 3), weights=None, pooling=None, include_top=False)
    x = Flatten()(model.output)

    x = Dense(7178, activation='sigmoid', name='predictions')(x)
    model = Model(inputs=model.input, outputs=x)
    return model

def get_model_resnext():
    model = ResNext((299,299,3), depth=[3, 4, 6, 3], cardinality=32, width=4, weight_decay=5e-4, classes=7178)
    return model 

def get_model_nasnet_large():
    from keras.models import Model
    from keras.applications.nasnet import NASNetLarge
    from keras.layers.core import Dense

    model = NASNetLarge(input_shape=(299, 299, 3), weights=None)
    x = model.layers[-2].output
    x = Dense(7178, activation='sigmoid', name='predictions')(x)
    model = Model(inputs=model.input, outputs=x)
    return model

def fbeta(y_true, y_pred, threshold_shift=0):
    from keras import backend as K
    beta = 1

    #y_true = K.constant(y_true)
    #y_pred = K.constant(y_pred)

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


def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)

def f2beta_loss(Y_true, Y_pred):
    from keras import backend as K
    eps = 0.001
    false_positive = K.sum(Y_pred * (1 - Y_true), axis=-1)
    false_negative = K.sum((1 - Y_pred) * Y_true, axis=-1)
    true_positive = K.sum(Y_true * Y_pred, axis=-1)
    p = (true_positive + eps) / (true_positive + false_positive + eps)
    r = (true_positive + eps) / (true_positive + false_negative + eps)
    out = (5*p*r + eps) / (4*p + r + eps)
    return -K.mean(out)


def preprocess_input(x):
    x /= 127.5
    x -= 1.
    return x

def get_model_inception_resnet_v2_491():
    from keras.models import Model
    from keras.applications.inception_resnet_v2 import InceptionResNetV2
    from keras.layers.core import Dense

    model = InceptionResNetV2(input_shape=(299, 299, 3), weights=None)
    x = model.layers[-2].output
    x = Dense(491, activation='sigmoid', name='predictions')(x)
    model = Model(inputs=model.input, outputs=x)
    return model