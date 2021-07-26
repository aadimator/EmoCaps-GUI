import json

import numpy as np
import tensorflow as tf
import _pickle as cPickle

from collections import Counter
from models import EfficientCapsNet

NUM_CHANNELS = 19

def load_config(config_path):
    """
    Load config file
    """
    with open(config_path) as json_data_file:
        config = json.load(json_data_file)
    return config

def predict(data_path, model_path):
    subject = cPickle.load(open(data_path, 'rb'), encoding='latin1')
    data = subject['data']
    data = data[np.newaxis, :]
    print(f"Data Shape: {data.shape}")
    data_rnn = apply_mixup(data)[1:]
    print(f"Data RNN Shape: {data_rnn.shape}")

    test_data = generate_tf_data(data_rnn, 16)
    # print(f"Test data Shape: {tf.shape(test_data.as_numpy_iterator())}")
    for element in test_data:
        print(element.shape)
        break

    config = load_config('config.json')
    model_test = EfficientCapsNet("DEAP", config, mode='test', custom_path=model_path, verbose=False)
    model_test.load_graph_weights()
    pred = model_test.predict(test_data)
    pred = np.argmax(pred, 1)
    return pred


def dominant_emotion(emotion_labels):
    return Counter(emotion_labels).most_common(1)[0][0]

def vad_to_emotion(vad_labels):
    emotion_map = {
        'HVHAHD': 'Joy',
        'HVHALD': 'Surprise',
        'HVLAHD': 'Excited',
        'HVLALD': 'Calm',
        'LVHAHD': 'Anger',
        'LVHALD': 'Fear', #Disgust
        'LVLAHD': 'Disgust',
        'LVLALD': 'Sadness'
    }

    return [emotion_map[num] for num in vad_labels]

def class_to_vad(labels):
    label_map = {
        0: 'HVHAHD',
        1: 'HVHALD',
        2: 'HVLAHD',
        3: 'HVLALD',
        4: 'LVHAHD',
        5: 'LVHALD',
        6: 'LVLAHD',
        7: 'LVLALD'
    }

    return [label_map[num] for num in labels]


def apply_mixup(data):
    data_in = data.transpose(0,2,1)
    
    window_size = 128

    # 0 valence, 1 arousal, 2 dominance, 3 liking   
    # dimensions = ['valence', 'arousal', 'dominance', 'liking']

    data_inter_rnn	= np.empty([0, window_size, NUM_CHANNELS])

    trials = data_in.shape[0]

    # Data pre-processing
    for trial in range(0,trials):
        base_signal = (data_in[trial,0:128,:] + data_in[trial,128:256,:] + data_in[trial,256:384,:])/3

        data = data_in[trial,384:,:]

        # compute the deviation between baseline signals and experimental signals
        segments = (data.shape[-1] // 128) - 3
        for i in range(0,segments):
            data[i*128:(i+1)*128,:]= data[i*128:(i+1)*128,:] - base_signal

        #read data and label
        data = norm_dataset(data)
        data = segment_signal_without_transition(data, window_size)

        # rnn data process
        data_rnn    = data.reshape(int(data.shape[0]/window_size), window_size, NUM_CHANNELS)
        # append new data and label
        data_inter_rnn  = np.vstack([data_inter_rnn, data_rnn])

    return data_inter_rnn

def windows(data, size):
	start = 0
	while ((start+size) < data.shape[0]):
		yield int(start), int(start + size)
		start += size

def segment_signal_without_transition(data, window_size):
	# get data file name and label file name
	for (start, end) in windows(data, window_size):
		if((len(data[start:end]) == window_size)):
			if(start == 0):
				segments = data[start:end]
				segments = np.vstack([segments, data[start:end]])
			else:
				segments = np.vstack([segments, data[start:end]])
	return segments

def feature_normalize(data):
	mean = data[data.nonzero()].mean()
	sigma = data[data. nonzero ()].std()
	data_normalized = data
	data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean)/sigma
	# return shape: 9*9
	return data_normalized

def norm_dataset(dataset_1D):
	norm_dataset_1D = np.zeros([dataset_1D.shape[0], NUM_CHANNELS])
	for i in range(dataset_1D.shape[0]):
		norm_dataset_1D[i] = feature_normalize(dataset_1D[i])
	# return shape: m*32
	return norm_dataset_1D


def pre_process(data):
	return (data)[...,None].astype('float32')

def generator(image):
	return image

PARALLEL_INPUT_CALLS = 16
def generate_tf_data(data, batch_size):
    dataset_test = pre_process(data)
    dataset_test = tf.data.Dataset.from_tensor_slices(dataset_test)
    dataset_test = dataset_test.cache()
    # dataset_test = dataset_test.shuffle(buffer_size=DEAP_TRAIN_IMAGE_COUNT)
    dataset_test = dataset_test.map(generator,
        num_parallel_calls=PARALLEL_INPUT_CALLS)
    dataset_test = dataset_test.batch(batch_size)
    dataset_test = dataset_test.prefetch(-1)

    return dataset_test
