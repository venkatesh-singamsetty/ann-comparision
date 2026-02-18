import os
import pandas as pd
import numpy as np
import tensorflow as tf
import time

# --- Environment Setup ---
# Fix for potential library conflicts on macOS and silence TensorFlow info logs
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- 1. DATA PREPROCESSING ---
print("--- Loading and Preprocessing Data ---")
# Load the Bank Churn dataset
df = pd.read_csv(r'Churn_Modelling.csv')

# Features: Extract columns from 'CreditScore' (3) up to the one before 'Exited' (-1)
X = df.iloc[:, 3:-1].values
# Target: 'Exited' column (1 = Churned, 0 = Stayed)
y = df.iloc[:, -1].values

# Encode Categorical Data: 'Gender' is at index 2
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# Encode Categorical Data: 'Geography' is at index 1
# Using OneHotEncoding to handle France, Germany, and Spain as separate binary features
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Feature Scaling: Normalize all features to ensure the ANN treats them equally
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Split the dataset: 80% Training and 20% Testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# --- 2. OPTIMIZER BENCHMARKING CONFIGURATION ---
optimizers = ['adam', 'adagrad', 'adadelta', 'adamax', 'rmsprop', 'sgd']
epoch_counts = [10, 50, 100]
results = []

print("Starting optimizer benchmarking suite (10, 50, 100 epochs per optimizer)...")

# --- 3. TRAINING LOOP ---
for epochs in epoch_counts:
    print(f"\n--- Testing for {epochs} Epochs ---")
    for opt in optimizers:
        print(f"Training with {opt}...", end=" ", flush=True)
        
        # Initialize Artificial Neural Network (ANN) structure
        # A simple Sequential model with 2 hidden layers
        ann = tf.keras.models.Sequential()
        # Hidden Layer 1: 6 neurons
        ann.add(tf.keras.layers.Dense(units=6))
        # Hidden Layer 2: 6 neurons, ReLU activation
        ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
        # Output Layer: 1 neuron, Sigmoid activation (for binary classification probability)
        ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
        
        # Compile Model with the specific optimizer being tested
        ann.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train and track time
        start_time = time.time()
        history = ann.fit(X_train, y_train, batch_size=32, epochs=epochs, verbose=0)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # Retrieve training history metrics
        acc_list = history.history['accuracy']
        loss_list = history.history['loss']
        
        # Identify the best performance markers throughout the training
        best_acc = max(acc_list)
        best_acc_epoch = acc_list.index(best_acc) + 1
        best_loss = min(loss_list)
        best_loss_epoch = loss_list.index(best_loss) + 1
        
        # Store results for final comparison
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

# --- 4. DISPLAY COMPARISON RESULTS ---
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

print("Benchmark Suite Completed.")
