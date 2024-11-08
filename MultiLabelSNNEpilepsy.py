import pandas as pd
import keras_tuner as kt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, make_scorer, multilabel_confusion_matrix
import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Reshape, TimeDistributed, InputLayer, Dropout
from keras.utils import to_categorical
import keras_spiking
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, ShuffleSplit, LearningCurveDisplay
import seaborn as sns
import numpy as np

# Importing dataset
dataset = pd.read_csv('/your/local/path/dataset.csv') # change the path to your local path

dataset = dataset.drop('Unnamed', axis=1) # drop useless column

# Removing useless instances 
dataset = dataset[dataset.y != 4]
dataset = dataset[dataset.y != 5]

# Retrieving values
X = dataset.iloc[:, 1:178].values
y = dataset.iloc[:, 178:].values
y = y.flatten()

#Label encoding for multi-target
l_encode = LabelEncoder()
l_encode.fit(y)
y = l_encode.transform(y)
y = to_categorical(y)


# 70% training, 15% validation e 15% test
train_size = 0.7
validation_size = 0.15
test_size = 0.15

# First split: train+validation and test set
X_train_validation, X_test, y_train_validation, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

# Second split, training set and validation set
relative_validation_size = validation_size / (validation_size + train_size)
X_train, X_validation, y_train, y_validation = train_test_split(X_train_validation, y_train_validation, test_size=relative_validation_size, random_state=42)

# Standardizing features to avoid bias and allow a proper distribution
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_validation = sc.transform(X_validation)


# naive SNN
model = Sequential([
    # Adding input layer and first hidden layer with spiking neurons
    Reshape((1, 177), input_shape=(177,)),
    tf.keras.layers.TimeDistributed(Dense(177, activation="relu")),
    keras_spiking.SpikingActivation("relu", spiking_aware_training=True),

    # Adding output layer
    tf.keras.layers.GlobalAveragePooling1D(),
    Dense(3, activation="softmax")
])

# categorical loss since the output is multi-label
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

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
cm = multilabel_confusion_matrix(y_test, y_pred_naive)
print(cm)


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
    model.add(Dense(units=hp.Int('units', min_value=32, max_value=320, step=32), activation='relu', input_shape=(177,))),
    keras_spiking.SpikingActivation("relu", spiking_aware_training=True),
    # Looking for an optimal depth 
    for i in range(hp.Int('num_layers', 1, 5)):
        model.add(Dense(units=hp.Int(f'layer_{i}_units', min_value=32, max_value=320, step=32), activation='relu'))
        # Dropout layet to reduce overfitting
        keras_spiking.SpikingActivation("relu", spiking_aware_training=True),


        model.add(Dropout(rate=hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1)))
    tf.keras.layers.GlobalAveragePooling1D(),
    model.add(Dense(3, activation='softmax'))
    # Looking for an optimal learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    # Compiling the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Defining the tuner
tuner = kt.Hyperband(build_model,
                     objective='accuracy',
                     max_epochs=13,
                     factor=3,
                     directory='project',
                     overwrite=True, 
                     project_name='nn_model_hp')


# Early stopping
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
# Finding the optmial hyperparameter configuration
tuner.search(X_train, y_train, epochs=25, validation_data=(X_validation, y_validation), callbacks=[stop_early])
# Retrieving the best parameteres
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]


# Retrieving the best model
models = tuner.get_best_models(num_models=1)
model = models[0]

# Re-training the optimal model and finding the best epoch
history_hp = model.fit(X_train, y_train, epochs=20, validation_data=(X_validation, y_validation))

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
cm = multilabel_confusion_matrix(y_test, y_pred_naive)
print(cm)

# Global performances

# Evaluating the optimal model
accuracy_train_test_hp = accuracy_score(y_test, y_pred_hp)
precision_hp = precision_score(y_test, y_pred_hp, average='weighted')
recall_hp = recall_score(y_test, y_pred_hp, average='weighted')
f1_hp = f1_score(y_test, y_pred_hp, average='weighted')

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

for train_index, test_index in kf.split(X, y.argmax(axis=1)):  # Use integer labels for splitting
    X_train_fold, X_test_fold = X[train_index], X[test_index]
    
    # Ensure y is still in integer form for splitting
    y_train_fold_int, y_test_fold_int = y.argmax(axis=1)[train_index], y.argmax(axis=1)[test_index]
    
    # One-hot encode y after splitting
    y_train_fold = to_categorical(y_train_fold_int, num_classes=3)  # Adjust num_classes as needed
    y_test_fold = to_categorical(y_test_fold_int, num_classes=3)

    # Train the model for each fold
    nn_model_hp.fit(X_train_fold, y_train_fold, epochs=best_epoch, batch_size=32, verbose=0)

    # Make predictions on the current fold's test set
    y_pred_prob = nn_model_hp.predict(X_test_fold)
    y_pred = np.round(y_pred_prob)

    # Calculate and store evaluation metrics for the current fold
    accuracy_fold = accuracy_score(y_test_fold.argmax(axis=1), y_pred.argmax(axis=1))  # Convert back from one-hot
    precision_fold = precision_score(y_test_fold.argmax(axis=1), y_pred.argmax(axis=1), average='macro')  # Multi-class
    recall_fold = recall_score(y_test_fold.argmax(axis=1), y_pred.argmax(axis=1), average='macro')
    f1_fold = f1_score(y_test_fold.argmax(axis=1), y_pred.argmax(axis=1), average='macro')

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

