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
from utils.layers import PrimaryCaps, FCCaps, Length, Mask


def efficient_capsnet_graph(input_shape, num_class, num_channels=32):
    """
    Efficient-CapsNet graph architecture.

    Parameters
    ----------   
    input_shape: list
        network input shape
    num_class: int
        number of output classes
    """
    inputs = tf.keras.Input(input_shape)
    
    x = tf.keras.layers.Conv2D(num_channels,5,activation="relu", padding='same', kernel_initializer='he_normal')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64,3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.Conv2D(64,3, activation='relu', padding='same', kernel_initializer='he_normal')(x)   
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128,3,2, activation='relu', padding='same', kernel_initializer='he_normal')(x)   
    x = tf.keras.layers.BatchNormalization()(x)

    x = PrimaryCaps(128, 9, -1, 8)(x)
    
    emo_caps = FCCaps(num_class,8)(x)
    
    emo_caps_len = Length(name='length_capsnet_output')(emo_caps)

    return tf.keras.Model(inputs=inputs,outputs=[emo_caps, emo_caps_len], name='Efficient_CapsNet')



def build_graph(input_shape, num_class, mode, verbose, num_channels=32):
    """
    Efficient-CapsNet graph architecture with reconstruction regularizer. The network can be initialize with different modalities.

    Parameters
    ----------   
    input_shape: list
        network input shape
    mode: str
        working mode ('train', 'test' & 'play')
    verbose: bool
    """
    inputs = tf.keras.Input(input_shape)

    efficient_capsnet = efficient_capsnet_graph(input_shape, num_class, num_channels=num_channels)

    if verbose:
        efficient_capsnet.summary()
        print("\n\n")
    
    _, digit_caps_len = efficient_capsnet(inputs)

    if mode == 'train':   
        return tf.keras.models.Model(inputs, digit_caps_len, name='Efficinet_CapsNet_Generator')
    elif mode == 'test':
        return tf.keras.models.Model(inputs, digit_caps_len, name='Efficinet_CapsNet_Generator')
    else:
        raise RuntimeError('mode not recognized')
