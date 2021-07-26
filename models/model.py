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

import os
import json
import numpy as np
import tensorflow as tf
from tqdm.notebook import tqdm
from utils.tools import get_callbacks, marginLoss, get_save_path
from models import efficient_capsnet_graph_deap


class Model(object):
    """
    A class used to share common model functions and attributes.
    
    ...
    
    Attributes
    ----------
    model_name: str
        name of the model (Ex. 'MNIST')
    mode: str
        model modality (Ex. 'test')
    config_path: str
        path configuration file
    verbose: bool
    
    Methods
    -------
    load_config():
        load configuration file
    load_graph_weights():
        load network weights
    predict(dataset_test):
        use the model to predict dataset_test
    evaluate(X_test, y_test):
        comute accuracy and test error with the given dataset (X_test, y_test)
    save_graph_weights():
        save model weights
    """
    def __init__(self, model_name, config, mode='test', verbose=True):
        self.model_name = model_name
        self.model = None
        self.mode = mode
        self.config = config
        self.verbose = verbose
    

    def load_graph_weights(self):
        try:
            self.model.load_weights(self.model_path)
        except Exception as e:
            print("[ERRROR] Graph Weights not found")
            
        
    def predict(self, dataset_test):
        return self.model.predict(dataset_test)
    

    def evaluate(self, X_test, y_test):
        print('-'*30 + f'{self.model_name} Evaluation' + '-'*30)
        y_pred =  self.model.predict(X_test)
        acc = np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0]
        test_error = 1 - acc
        print('Test acc:', acc)
        print(f"Test error [%]: {(test_error):.4%}")
        if self.model_name == "MULTIMNIST":
            print(f"N° misclassified images: {int(test_error*len(y_test)*self.config['n_overlay_multimnist'])} out of {len(y_test)*self.config['n_overlay_multimnist']}")
        else:
            print(f"N° misclassified images: {int(test_error*len(y_test))} out of {len(y_test)}")
        return acc, test_error


    def save_graph_weights(self):
        self.model.save_weights(self.model_path)



class EfficientCapsNet(Model):
    """
    A class used to manage an Efficiet-CapsNet model. 'model_name' and 'mode' define the particular architecure and modality of the 
    generated network.
    
    ...
    
    Attributes
    ----------
    model_name: str
        name of the model (Ex. 'MNIST')
    mode: str
        model modality (Ex. 'test')
    config_path: str
        path configuration file
    custom_path: str
        custom weights path
    verbose: bool
    
    Methods
    -------
    load_graph():
        load the network graph given the model_name
    train(dataset, initial_epoch)
        train the constructed network with a given dataset. All train hyperparameters are defined in the configuration file

    """
    def __init__(self, model_name, config, subject=None, fold=None, dimension=None, mode='test', custom_path=None, verbose=True, desc=""):
        Model.__init__(self, model_name, config, mode, verbose)
        self.subject = subject
        self.fold = fold
        self.dimension = dimension
        self.description = desc
        if custom_path != None:
            self.model_path = custom_path
        else:
            self.model_path = get_save_path(self.config['saved_model_dir'], self.model_name, dimension=self.dimension, subject=self.subject, desc=self.description) / f"best_model.h5"
        
        self.model_path_new_train = get_save_path(self.config['saved_model_dir'], self.model_name, dimension=self.dimension, subject=self.subject, desc=self.description) / f"fold_{self.fold}.h5"
        self.tb_path = get_save_path(self.config['tb_log_save_dir'], self.model_name, dimension=self.dimension, subject=self.subject, desc=self.description) / f"fold_{self.fold}"
        self.csv_log_path = get_save_path(self.config['csv_log_save_dir'], self.model_name, dimension=self.dimension, subject=self.subject, desc=self.description) / f"fold_{self.fold}.csv"
        self.load_graph()
    

    def load_graph(self):
        if self.model_name == 'DEAP':
            self.model = efficient_capsnet_graph_deap.build_graph(self.config['DEAP_INPUT_SHAPE'], self.config['DEAP_NUM_CLASS'], self.mode, self.verbose, num_channels=self.config["num_channels"])
            
    def train(self, dataset_train, initial_epoch=0):
        callbacks = get_callbacks(self.csv_log_path, self.tb_path, self.model_path_new_train, self.config['lr_dec'], self.config['lr'])

        if self.model_name == 'DEAP':
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['lr']),
                loss=[marginLoss, 'mse'],
                loss_weights=[1.],
                metrics=['accuracy'])
            steps=None

        print('-'*30 + f'{self.model_name} train' + '-'*30)

        history = self.model.fit(dataset_train,
          epochs=self.config[f'epochs'], steps_per_epoch=steps,
          batch_size=self.config['batch_size'], initial_epoch=initial_epoch,
          callbacks=callbacks)
        
        return history

    def evaluate(self, dataset_test):
        print('-'*30 + f'{self.model_name} Evaluation' + '-'*30)
        scores =  self.model.evaluate(dataset_test)
        print(f'Scores: {self.model.metrics_names[0]} of {scores[0]}; {self.model.metrics_names[1]} of {scores[1]*100}%')
        return scores
