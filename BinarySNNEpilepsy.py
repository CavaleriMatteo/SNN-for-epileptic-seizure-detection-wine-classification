import keras_tuner as kt
import keras_spiking
import numpy as np
import sklearn as sn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
import scipy.stats as st

from matplotlib.patches import Patch

from time import time

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Dense, Reshape, TimeDistributed
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, make_scorer
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier, plot_tree

from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, ShuffleSplit, LearningCurveDisplay

import tensorflow as tf
from tensorflow import keras

# Importing dataset
dataset = pd.read_csv('/your/local/path/dataset.csv') # change the path to your local path
dataset = dataset.drop('Unnamed', axis=1) # drop useless column

# Retrieving values
X = dataset.iloc[:, 1:178].values
y = dataset.iloc[:, 178:].values
y = y.flatten()


for i in range(len(y)):
    # 1 means seizure, any other values means no seizure then 0
    if y[i] != 1:
        y[i] = 0



# 70% training, 15% validation e 15% test
train_size = 0.7
validation_size = 0.15
test_size = 0.15

# First split: train+validation and test set
X_train_validation, X_test, y_train_validation, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Second split, training set and validation set
relative_validation_size = validation_size / (validation_size + train_size)
X_train, X_validation, y_train, y_validation = train_test_split(X_train_validation, y_train_validation, test_size=relative_validation_size, random_state=42)

# Standardizing features to avoid bias and allow a proper distribution
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_validation = sc.transform(X_validation)

#naive SNN
model = Sequential([
    # Adding input layer and first hidden layer with spiking neurons
    Reshape((1, 177), input_shape=(177,)),
    tf.keras.layers.TimeDistributed(Dense(177, activation="relu")),
    keras_spiking.SpikingActivation("relu", spiking_aware_training=True),


    # Adding output layer
    tf.keras.layers.GlobalAveragePooling1D(),
    Dense(1, activation="sigmoid")
])

# binary loss since the output is boolean
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history_naive = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_validation, y_validation))

print('Training loss:', history_naive.history['loss'][-1])
print('Training accuracy:', history_naive.history['accuracy'][-1])
nn_naive_train_accuracy = history_naive.history['accuracy'][-1]

# performance model (Accuracy and Loss)
score_naive = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score_naive[0])
print('Test accuracy:', score_naive[1])
nn_naive_test_accuracy = score_naive[1]

# Prediction on naive model
y_pred_naive = model.predict(X_test)
y_pred_naive = np.round(y_pred_naive)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_naive)

# Plotting confusion matrix
plt.figure(figsize=(6,6))
sns.heatmap(cm, square=True, annot=True, fmt='d', cbar=True,
                        xticklabels=['Seizure', 'No seizure'],
                        yticklabels=['Seizure', 'No seizure'])

plt.ylabel('Actual Label')
plt.xlabel('Predicted Label');
plt.show()

# Global performance

# Evaluating the model predictions
accuracy_train_test_naive = accuracy_score(y_test, y_pred_naive)
precision_naive = precision_score(y_test, y_pred_naive)
recall_naive = recall_score(y_test, y_pred_naive)
f1_naive = f1_score(y_test, y_pred_naive)

# printing naive model metrics
print('Accuracy:', accuracy_train_test_naive)
print('Precision:', precision_naive)
print('Recall:', recall_naive)
print('F1-score:', f1_naive)

nn_naive_global_report = {
    'accuracy': accuracy_train_test_naive,
    'precision': precision_naive,
    'recall': recall_naive,
    'f1-score': f1_naive
}

import matplotlib.pyplot as plt

# Plot loss and accuracy
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot loss
axes[0].plot(history_naive.history['loss'], label='train_loss')
axes[0].plot(history_naive.history['val_loss'], label='val_loss')
axes[0].set_title('Learning Curve (Loss)')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Loss')
axes[0].legend()

# Plot accuracy
axes[1].plot(history_naive.history['accuracy'], label='train_accuracy')
axes[1].plot(history_naive.history['val_accuracy'], label='val_accuracy')
axes[1].set_title('Learning Curve (Accuracy)')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

plt.tight_layout()
plt.show()

def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu', input_shape=(177,))),
    keras_spiking.SpikingActivation("relu", spiking_aware_training=True),
    # Looking for an optimal depth 
    for i in range(hp.Int('num_layers', 1, 4)):
        model.add(Dense(units=hp.Int(f'layer_{i}_units', min_value=32, max_value=512, step=32), activation='relu'))
        # Dropout layet to reduce overfitting
        keras_spiking.SpikingActivation("relu", spiking_aware_training=True),


        model.add(Dropout(rate=hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1)))
    tf.keras.layers.GlobalAveragePooling1D(),
    model.add(Dense(1, activation='sigmoid'))
    # Looking for an optimal learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    # Compiling the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Defining the tuner
tuner = kt.Hyperband(build_model,
                     objective='accuracy',
                     max_epochs=15,
                     factor=3,
                     directory='project',
                     overwrite=True, 
                     project_name='nn_model_hp')

# Early stopping 
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
# Finding the optmial hyperparameter configuration
tuner.search(X_train, y_train, epochs=15, validation_data=(X_validation, y_validation), callbacks=[stop_early])
# Retrieving the best parameteres
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Retrieving the best model
models = tuner.get_best_models(num_models=1)
model = models[0]

# Re-training the optimal model and finding the best epoch
history_hp = model.fit(X_train, y_train, epochs=15, validation_data=(X_validation, y_validation))

val_acc_per_epoch = history_hp.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1


models = tuner.get_best_models(num_models=1)
nn_model_hp  = models[0]

history_hp = nn_model_hp.fit(X_train, y_train, epochs=best_epoch, validation_data=(X_validation, y_validation))

# Training metrics

print('Training loss:', history_hp.history['loss'][-1])
print('Training accuracy:', history_hp.history['accuracy'][-1])
nn_hp_train_accuracy = history_hp.history['accuracy'][-1]

# Test metrics (Accuracy and Loss)
score_hp = nn_model_hp.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score_hp[0])
print('Test accuracy:', score_hp[1])
nn_hp_test_accuracy = score_hp[1]

# Predictions 
y_pred_hp = nn_model_hp.predict(X_test)
y_pred_hp = np.round(y_pred_hp)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_hp)

# Plotting the confusion matrix
plt.figure(figsize=(6,6))
sns.heatmap(cm, square=True, annot=True, fmt='d', cbar=True,
                        xticklabels=['No seizure', 'Seizure'],
                        yticklabels=['No seizure', 'Seizure'])

plt.ylabel('Actual Label')
plt.xlabel('Predict Label');
plt.show()

# Global performances

# Evaluating the optimal model
accuracy_train_test_hp = accuracy_score(y_test, y_pred_hp)
precision_hp = precision_score(y_test, y_pred_hp)
recall_hp = recall_score(y_test, y_pred_hp)
f1_hp = f1_score(y_test, y_pred_hp)

# printing the metrics
print('Accuracy:', accuracy_train_test_hp)
print('Precision:', precision_hp)
print('Recall:', recall_hp)
print('F1-score:', f1_hp)

nn_hp_global_report = {
    'accuracy': accuracy_train_test_hp,
    'precision': precision_hp,
    'recall': recall_hp,
    'f1-score': f1_hp
}

# Comparing the curves: naive vs advanced models
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plotting
axes[0].plot(history_naive.history['loss'], label='naive_train_loss')
axes[0].plot(history_naive.history['val_loss'], label='naive_val_loss')
axes[0].plot(history_hp.history['loss'], label='optimized_train_loss')
axes[0].plot(history_hp.history['val_loss'], label='optimized_val_loss')
axes[0].set_title('Learning Curve (Loss)')
axes[0].set_xlabel('Epochs')
axes[0].set_ylabel('Loss')
axes[0].legend()

# Plotting
axes[1].plot(history_naive.history['accuracy'], label='naive_train_accuracy')
axes[1].plot(history_naive.history['val_accuracy'], label='naive_val_accuracy')
axes[1].plot(history_hp.history['accuracy'], label='optimized_train_accuracy')
axes[1].plot(history_hp.history['val_accuracy'], label='optimized_val_accuracy')
axes[1].set_title('Learning Curve (Accuracy)')
axes[1].set_xlabel('Epochs')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

plt.tight_layout()
plt.show()

# Cross Validation to ensure robustness

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# Stratified K-fold Cross-Validation with 10 fold
n_fold = 10
kf = StratifiedKFold(n_splits=n_fold, shuffle=True)


scores = []


for train_index, test_index in kf.split(X, y):
    X_train_fold, X_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = np.array(y)[train_index], np.array(y)[test_index]

    # traning per each fold
    nn_model_hp.fit(X_train_fold, y_train_fold, epochs=best_epoch, batch_size=32, verbose=0)

    # predictions on the current fold
    y_pred_prob = nn_model_hp.predict(X_test_fold)
    y_pred = np.round(y_pred_prob)

    # Saving metrics of the current fold
    accuracy_fold = accuracy_score(y_test_fold, y_pred)
    precision_fold = precision_score(y_test_fold, y_pred)
    recall_fold = recall_score(y_test_fold, y_pred)
    f1_fold = f1_score(y_test_fold, y_pred)

    accuracy_scores.append(accuracy_fold)
    precision_scores.append(precision_fold)
    recall_scores.append(recall_fold)
    f1_scores.append(f1_fold)

# Average metrics
avg_accuracy = np.mean(accuracy_scores)
avg_precision = np.mean(precision_scores)
avg_recall = np.mean(recall_scores)
avg_f1 = np.mean(f1_scores)

# Printing metrics
print('Average Test accuracy:', avg_accuracy)
print('Average Precision:', avg_precision)
print('Average Recall:', avg_recall)
print('Average F1-score:', avg_f1)

