import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # Environment guard for Mac
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Lowers console logging noise

import pandas as pd
import numpy as np
import tensorflow as tf
import time

# Data Preprocessing
df = pd.read_csv(r'Churn_Modelling.csv')
X = df.iloc[:, 3:-1].values
y = df.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

optimizers = ['adam', 'adagrad', 'adadelta', 'adamax', 'rmsprop', 'sgd']
epoch_counts = [10, 50, 100]
results = []

print("Starting optimizer benchmarking suite (10, 50, 100 epochs per optimizer)...")

for epochs in epoch_counts:
    print(f"\n--- Testing for {epochs} Epochs ---")
    for opt in optimizers:
        print(f"Training with {opt}...", end=" ", flush=True)
        
        # Initialize a fresh model structure
        ann = tf.keras.models.Sequential()
        ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
        
        ann.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        
        start_time = time.time()
        history = ann.fit(X_train, y_train, batch_size=32, epochs=epochs, verbose=0)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # Extract metrics from history
        acc_list = history.history['accuracy']
        loss_list = history.history['loss']
        
        best_acc = max(acc_list)
        best_acc_epoch = acc_list.index(best_acc) + 1
        best_loss = min(loss_list)
        best_loss_epoch = loss_list.index(best_loss) + 1
        
        results.append({
            'Epochs_Target': epochs,
            'Optimizer': opt, 
            'Time': total_time,
            'BestAcc': best_acc, 
            'BestAccEpoch': best_acc_epoch,
            'BestLoss': best_loss, 
            'BestLossEpoch': best_loss_epoch,
            'FinalAcc': acc_list[-1], 
            'FinalLoss': loss_list[-1]
        })
        print(f"Done ({total_time:.2f}s)")

# Display results table
header = "{:<7} | {:<10} | {:<8} | {:<10} | {:<6} | {:<10} | {:<6} | {:<10} | {:<10}"
print("\n" + "="*128)
print(header.format('Epochs', 'Optimizer', 'Time(s)', 'Best Acc', 'Epoch', 'Best Loss', 'Epoch', 'Final Acc', 'Final Loss'))
print("-" * 128)
for r in results:
    print(header.format(
        r['Epochs_Target'],
        r['Optimizer'], 
        round(r['Time'], 2), 
        round(r['BestAcc'], 4), 
        r['BestAccEpoch'], 
        round(r['BestLoss'], 4), 
        r['BestLossEpoch'], 
        round(r['FinalAcc'], 4), 
        round(r['FinalLoss'], 4)
    ))
print("="*128)
