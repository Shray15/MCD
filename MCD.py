import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm
import librosa, librosa.display, librosa.util
from pathlib import Path
import h5py
import math
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.python import metrics
import tensorflow as tf
from re import search
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix,accuracy_score
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Add, Dense, Dropout, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,GlobalAveragePooling2D,Concatenate, ReLU, LeakyReLU,Reshape, Lambda


def load_data(data_path):# Open the HDF5 file
    with h5py.File(data_path, "r") as f:
        # Get the mfcc and labels datasets
        mfcc = f["mfcc"][:]
        labels = f["labels"][:]

        
        
        return mfcc, labels
    
train_data_path = 'train_10secs_44.1khz_top10labels.h5'
test_data_path = 'test_10secs_44.1khz_top10labels.h5'
trainx, trainy = load_data(train_data_path)
testx, testy = load_data(test_data_path)

tot = []
for i in trainy:
    bytes_obj = i
    string_obj = bytes_obj.decode('utf-8')
    label_list = [label for label in string_obj.split(',')]
    tot.append(label_list)

tott = []
for i in testy:
    bytes_obj = i
    string_obj = bytes_obj.decode('utf-8')
    label_list = [label for label in string_obj.split(',')]
    tott.append(label_list)
    
all_elements = []
all_elements2 =[]

# Concatenate all the lists into a single list
for lst in tot:
    all_elements.extend(lst)

for lst in tott:
    all_elements2.extend(lst)

# Convert the list into a set to remove duplicates
unique_elements = set(all_elements)
unique_elements2 = set(all_elements2)

# Convert the set back into a list
unique_list = list(unique_elements)
unique_list2 = list(unique_elements2)
   
    
lb = MultiLabelBinarizer()
trainy = lb.fit_transform(tot)
testy = lb.fit_transform(tott)

optimizer2 = keras.optimizers.Adam(learning_rate = 0.001)
nlabls = len(unique_list)
trainx = trainx[:,:,:,np.newaxis]
testx = testx[:,:,:, np.newaxis]
input_shape = (trainx.shape[1], trainx.shape[2], trainx.shape[3])
# print(input_shape)

def identity_block(X, f, filters, stage, block, dropout_rate = 0.2):
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    F1, F2, F3 = filters
    
    X_shortcut = X
        
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
#     X = Dropout(dropout_rate)(X)
        
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
#     X = Dropout(dropout_rate)(X)

    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Add shortcut value to main path
    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)
#     X = Dropout(dropout_rate)(X)
        
    return X

def convolutional_block(X, f, filters, stage, block, s = 2, d = 0.2):
        
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters
    X_shortcut = X
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
#     X = Dropout(d)(X)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
#     X = Dropout(d)(X)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)
    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)
#     X = Dropout(d)(X)
   
    return X



def ResNet50(input_shape = input_shape, classes = nlabls, p= 0.3):
    
    X_input = Input(input_shape)
    X = ZeroPadding2D((3, 3))(X_input)
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    # X = Dropout(p)(X)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')
    X = convolutional_block(X, f = 3, filters = [128,128,512], stage = 3, block='a', s = 2)
    # X = Dropout(p)(X)
    X = identity_block(X, 3, [128,128,512], stage=3, block='b')
    X = identity_block(X, 3, [128,128,512], stage=3, block='c')
    X = identity_block(X, 3, [128,128,512], stage=3, block='d')
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    # X = Dropout(p)(X)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    # X = Dropout(p)(X)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')
    X = AveragePooling2D(pool_size=(1, 1),name='avg_pool')(X)
    X = Flatten()(X)
    X = Dropout(p)(X)
    X = Dense(classes, activation='sigmoid', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
#     X = Dropout(p)(X)
    model = Model(inputs=X_input, outputs=X, name='ResNet50_MC')
    return model


from keras import backend as K
def mean_accuracy(y_true, y_pred):
    """
    Calculates the mean accuracy across all labels.
    y_true: true labels
    y_pred: predicted labels
    """
    # Calculate accuracy for each label
    accs = tf.cast(tf.equal(y_true, tf.round(y_pred)), tf.float32)
    # Calculate mean accuracy across all labels
    return K.mean(accs)

bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
model = ResNet50(input_shape = input_shape, classes = nlabls)
model.compile(optimizer='adam', loss=bce, metrics=[mean_accuracy])
model.summary()

model.load_weights('model_weights_top10_MC.h5')

y_pred = model.predict(testx)
yy_pred=[]
for sample in  y_pred:
    yy_pred.append([1 if i>=0.5 else 0 for i in sample ] )
yy_pred = np.array(yy_pred)

acc_mean,acc = mean_accuracy(testy, yy_pred)



def label_wise_accuracy(y_true, y_pred):
    """
    Calculate the average accuracy for each label in a multi-label classification problem.
    
    Parameters:
        y_true (numpy array): True labels of shape (n_samples, n_labels).
        y_pred (numpy array): Predicted labels of shape (n_samples, n_labels).
        
    Returns:
        numpy array: Average accuracy for each label of shape (n_labels,).
    """
    n_labels = y_true.shape[1]
    accuracies = np.zeros(n_labels)
    
    for label in range(n_labels):
        true_positives = np.sum(np.logical_and(y_true[:, label] == 1, y_pred[:, label] == 1))
        true_neg = np.sum(np.logical_and(y_true[:, label] == 0, y_pred[:, label] == 0))
        false_positives = np.sum(np.logical_and(y_true[:, label] == 0, y_pred[:, label] == 1))
        false_negatives = np.sum(np.logical_and(y_true[:, label] == 1, y_pred[:, label] == 0))
        
        accuracy = (true_positives + true_neg) / (true_positives +true_neg + false_positives + false_negatives + 1e-7)
        accuracies[label] = accuracy
        acc_mean = accuracies.mean()
        
    return accuracies, acc_mean

acc, acc_mean = label_wise_accuracy(testy, yy_pred)
result = pd.DataFrame({ 'Labels' : unique_list , ' Accuracy' : acc *100.0})

def predict_mc_dropout2(model, X, num_mc_samples):
    model(X,training=True)
    y_preds = []
    for i in range(num_mc_samples):
        y_pred = model.predict(X, verbose = 0)
        y_preds.append(y_pred)
    y_preds = np.array(y_preds)
        
    return y_preds

# Perform Monte Carlo dropout sampling
num_mc_samples = 50
confidences = predict_mc_dropout2(model, testx, num_mc_samples)

mean_probs = np.mean(confidences, axis=0)

# Calculate the overall confidence
overall_confidence = np.mean(np.max(mean_probs, axis=1))

# Calculate the confidence for each label
label_confidences = np.max(mean_probs, axis=0)

# Print the results
print('Overall confidence:', overall_confidence)

result['conf'] = label_confidences

mean_pred_prob = np.mean(confidences, axis=0)

import numpy as np
from itertools import tee
# from .numpy_metrics import accuracy

EPSILON = 1e-5

#From itertools recipes
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def classifier_calibration_error(y_pred, y_true, y_confidences, metric="mae", num_bins=10, weighted=False):
    """
        Estimates calibration error for a classifier for each label separately.
        y_pred are the class predictions of the model (integers), while y_true is the ground truth labels (integers),
        and y_confidences are confidences for each prediction (in the [0, 1] range).
        All three arrays must have equal number of samples.
    """

    bin_edges = np.linspace(0.0, 1.0 + EPSILON, num_bins + 1)

    label_errors = {}

    for label in range(y_confidences.shape[-1]):
        label_y_pred = y_pred[:, label]
        label_y_true = y_true[:, label]
        label_y_confidences = y_confidences[:, label]

        errors = []
        weights = []

        for start, end in pairwise(bin_edges):
            indices = np.where(np.logical_and(label_y_confidences >= start, label_y_confidences < end))
            filt_preds = label_y_pred[indices]
            filt_classes = label_y_true[indices]
            filt_confs = label_y_confidences[indices]

            if len(filt_confs) > 0:
                bin_acc = accuracy_score(filt_classes, filt_preds)
                bin_conf = np.mean(filt_confs)

                error = abs(bin_conf - bin_acc)
                weight = len(filt_confs)

                errors.append(error)            
                weights.append(weight)

        errors = np.array(errors)
        weights = np.array(weights) / sum(weights)

        if weighted:
            label_errors[label] = sum(errors * weights)
        else:
            label_errors[label] = np.mean(errors)

    return label_errors

err = classifier_calibration_error(yy_pred, testy, mean_pred_prob)

def classifier_calibration_curves(y_pred, y_true, y_confidences, metric="mae", num_bins=10):
    """
        Estimates the calibration plot for a classifier and returns the points in the plot.
        y_pred are the class predictions of the model (integers), while y_true is the ground truth labels (integers),
        and y_confidences are confidences for each prediction (in the [0, 1] range).
        All three arrays must have equal number of samples.
    """

    bin_edges = np.linspace(0.0, 1.0 + EPSILON, num_bins + 1)
    curves_conff = []
    curves_accc = []

    for label in range(y_confidences.shape[-1]):
        label_y_pred = y_pred[:, label]
        label_y_true = y_true[:, label]
        label_y_confidences = y_confidences[:, label]

        curve_conf = []
        curve_acc = []

        for start, end in pairwise(bin_edges):
            indices = np.where(np.logical_and(label_y_confidences >= start, label_y_confidences < end))
            filt_preds = label_y_pred[indices]
            filt_classes = label_y_true[indices]
            filt_confs = label_y_confidences[indices]

            if len(filt_confs) > 0:
                bin_acc = accuracy_score(filt_classes, filt_preds)
                bin_conf = np.mean(filt_confs)

                curve_conf.append(bin_conf)
                curve_acc.append(bin_acc)
            else:
                p = np.mean([start, end])
                curve_conf.append(p)
                curve_acc.append(p)

        curves_conff.append(curve_conf)
        curves_accc.append(curve_acc)

    return curves_conff, curves_accc


curve_conf, curve_acc = classifier_calibration_curves(yy_pred,testy, mean_pred_prob)


import matplotlib.pyplot as plt

curves_conf, curves_acc = classifier_calibration_curves(yy_pred,testy, mean_pred_prob)
for label in range(21):
    plt.plot(curves_conf[label], curves_acc[label], label=f"label {label}")
    
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("Confidence")
plt.ylabel("Accuracy")
# plt.legend()
plt.show()


