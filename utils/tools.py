# Copyright 2021 Vittorio Mazzia & Francesco Salvetti. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import tensorflow as tf
from pathlib import Path

def learn_scheduler(lr_dec, lr):
    def learning_scheduler_fn(epoch):
        lr_new = lr * (lr_dec ** epoch)
        return lr_new if lr_new >= 5e-5 else 5e-5
    return learning_scheduler_fn


def get_callbacks(csv_save_path, tb_log_save_path, saved_model_path, lr_dec, lr):
    log = tf.keras.callbacks.CSVLogger(csv_save_path)

    tb = tf.keras.callbacks.TensorBoard(log_dir=tb_log_save_path, histogram_freq=0)

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(saved_model_path, monitor='loss',
                                           save_best_only=True, save_weights_only=True, verbose=1)

    lr_decay = tf.keras.callbacks.LearningRateScheduler(learn_scheduler(lr_dec, lr))

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.9,
                              patience=4, min_lr=0.00001, min_delta=0.0001, mode='max')

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4, min_delta=0.001)

    return [log, tb, model_checkpoint, lr_decay, early_stop]


def marginLoss(y_true, y_pred):
    lbd = 0.5
    m_plus = 0.9
    m_minus = 0.1
    
    L = y_true * tf.square(tf.maximum(0., m_plus - y_pred)) + \
    lbd * (1 - y_true) * tf.square(tf.maximum(0., y_pred - m_minus))

    return tf.reduce_mean(tf.reduce_sum(L, axis=1))


def multiAccuracy(y_true, y_pred):
    label_pred = tf.argsort(y_pred,axis=-1)[:,-2:]
    label_true = tf.argsort(y_true,axis=-1)[:,-2:]
    
    acc = tf.reduce_sum(tf.cast(label_pred[:,:1]==label_true,tf.int8),axis=-1) + \
          tf.reduce_sum(tf.cast(label_pred[:,1:]==label_true,tf.int8),axis=-1)
    acc /= 2
    return tf.reduce_mean(acc,axis=-1)

def get_save_path(save_dir, model_name, dimension=None, subject=None, desc=None):
    new_path = Path(save_dir + f"/efficient_capsnet_{model_name}")
    if desc:
        new_path = new_path / f"{desc}"
    if dimension:
        new_path = new_path / f"{dimension}"
    if subject is not None:
        new_path = new_path / f"sub_{subject}"
    new_path.mkdir(parents=True, exist_ok=True)
    return new_path

def save_best_model(save_dir, test_results):
    best_fold_result = np.argmax(test_results)
    p = Path(save_dir / f"fold_{best_fold_result}.h5")
    p.replace(Path(p.parent, f"best_model{p.suffix}"))

    # remove the rest folds
    for child in save_dir.glob('fold_*.h5'):
        child.unlink(missing_ok=True)