import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="1" # Select GPU
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import pathlib  # pathlib is in standard library
import numpy as np
#import cv2
import matplotlib as plt
from PIL import Image
import pickle

# Import csv file along with ground truth
root_path = "/mnt/DATA/"

directory = root_path

df = pd.read_csv(directory + "DRR_filenames_labels.csv", sep=";")
#df = pd.read_csv(directory + "DRR_filenames_labels_arbitrary_simple.csv", sep=";")
df.head()

#Extract info from csv
file_paths = df["file_name"].values
labels = df["label"].values
ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

images = []

# Creates tensorflow dataset from dataframe
def dataset_from_df(dataframe):
    file_paths = dataframe["file_name"].values
    labels = dataframe["label"].values
    images = []
    for i in range(0,len(labels)):
        image = Image.open(directory + file_paths[i])
        image = image.convert('RGB')
        image = np.array(image.resize((224,224)))
        image = np.reshape(image,[1,224,224,3]) # return the image with shaping that TF wants.
        if i == 1:
            print(np.shape(image))
        images.append(image)

    labels = np.asarray(labels).astype('float32').reshape((-1,1))
    ds = (
        tf.data.Dataset.from_tensor_slices(
            (
                tf.cast(images, tf.float64),
                tf.cast(labels, tf.int32)
            )
        )
    )
    return ds

# Create full dataset
ds = dataset_from_df(df)

import sklearn
import sklearn.model_selection

kfold = sklearn.model_selection.KFold(n_splits=10, shuffle=True, random_state=12)

import random
import math
tf.random.set_seed(0)
random.seed=0

# Data augmentation function (not used in the end)
class Brightness(tf.keras.layers.Layer):
    def __init__(self, brightness_level=0.05, **kwargs):
        self.brightness_level = brightness_level
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, p=0.5, training=None):
        brightness_value = random.uniform(1-self.brightness_level, 1+self.brightness_level)
        if  random.uniform(0,1) < p:
            return tf.clip_by_value(inputs * brightness_value,0, 255)
        else:
            return inputs
      
brightness_layer=Brightness()

 # Not used in the end
data_augmentation = tf.keras.Sequential([
#   tf.keras.layers.experimental.preprocessing.RandomRotation(0.04, seed=0), # 0.04 approx 15 degrees
   tf.keras.layers.experimental.preprocessing.RandomContrast(0.05, seed=0),
#  tf.keras.layers.experimental.preprocessing.RandomBrightness(0.05, value_range=(0, 1), seed=0),
   brightness_layer,
   #rotation_layer,
   tf.keras.layers.experimental.preprocessing.Resizing(height=224, width=224)
])

# Metrics for performance
metrics = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]  

i = 1
for train_index, val_index in kfold.split(np.zeros(len(labels))):
    df_train = df.iloc[train_index]
    df_val = df.iloc[val_index]
    
    ds_train = dataset_from_df(df_train)
    ds_val = dataset_from_df(df_val)

    base_model = tf.keras.applications.EfficientNetB7(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling=max,
        classifier_activation="softmax",
    )

    # We make sure that the base_model is running in inference mode here,
    # by setting `trainable=False`. This is important for fine-tuning
    base_model.trainable = False
    
    # Create new model on top
    initial = keras.Sequential([tf.keras.layers.Reshape((224,224,3), input_shape=(224,224,3))])
    inputs = keras.Input(shape=(224, 224, 3))
    x = initial(inputs)
    #x = data_augmentation(x)  # Apply random data augmentation
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(1,activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)

    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=metrics,
    )

    batch_size=5
    history = model.fit(ds_train, epochs=500, verbose=2, batch_size=batch_size)
    
    # Save train performance for each fold
    with open(root_path + 'DRR_EfficientNet-B7_trainHistoryDict_fold_' + str(i), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

    import matplotlib.pyplot as plt
    from tensorflow.keras.models import Sequential, save_model, load_model
    
    # Save the model
    filepath = root_path + '/DRR_EfficientNet-B7_model_simple_fold_' + str(i)
    save_model(model, filepath)

        # Evaluate the model on the test data using `evaluate`
    batch_size=5
    print("Evaluate on test data")
    results = model.evaluate(ds_val, batch_size=batch_size)
    print("test loss, test acc:", results)
    with open(root_path + 'DRR_EfficientNet-B7_testResultsDict_fold_' + str(i), 'wb') as file_pi:
            pickle.dump(results, file_pi)
    print("Generate predictions for all test samples")
    predictions = model.predict(ds_val)
    print("predictions shape:", predictions.shape)
    with open(root_path + 'DRR_EfficientNet-B7_testPredictions_fold_' + str(i), 'wb') as file_pi:
            pickle.dump(predictions, file_pi)

    # Finetuning part
    # Unfreeze the base model
    base_model.trainable = True

    # Optional code to unfreeze the last three layers
    #print(base_model.layers[-3].name)
    set_trainable = False
    for layer in base_model.layers:
        if layer.name == 'conv5_block3_3_bn': #name of 3rd last layer
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    # It's important to recompile your model after you make any changes
    # to the `trainable` attribute of any inner layer, so that your changes
    # are take into account
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=metrics,
    )

    # Train end-to-end. Be careful to stop before you overfit!
    history_finetuned = model.fit(ds_train, epochs=500, verbose=2, batch_size=batch_size)#,callbacks=[early_stopping])

    with open(root_path + 'DRR_EfficientNet-B7_finetuned_trainHistoryDict_fold_' + str(i), 'wb') as file_pi:
            pickle.dump(history_finetuned.history, file_pi)

    filepath = root_path + '/DRR_EfficientNet-B7_finetuned_fold_' + str(i)
    save_model(model, filepath)

    # Evaluate the model on the test data using `evaluate`
    batch_size=5
    print("Evaluate on test data")
    results = model.evaluate(ds_val, batch_size=batch_size)
    print("test loss, test acc:", results)
    with open(root_path + 'DRR_EfficientNet-B7_finetuned_testResultsDict_fold_' + str(i), 'wb') as file_pi:
            pickle.dump(results, file_pi)
    print("Generate predictions for all test samples")
    predictions = model.predict(ds_val)
    print("predictions shape:", predictions.shape)
    with open(root_path + 'DRR_EfficientNet-B7_finetuned_testPredictions_fold_' + str(i), 'wb') as file_pi:
            pickle.dump(predictions, file_pi)
            
    i = i + 1