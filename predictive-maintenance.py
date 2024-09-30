# Import libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the four EPA datasets
file_42602 = pd.read_csv('/content/daily_42602_2024.csv')
file_42401 = pd.read_csv('/content/daily_42401_2024.csv')
file_42101 = pd.read_csv('/content/daily_42101_2024.csv')
file_44201 = pd.read_csv('/content/daily_44201_2024.csv')

# Define the pollutants to analyze (PM2.5 and Ozone)
pollutants = ['PM2.5', 'Ozone']

# Filter each dataset for relevant pollutants and columns
filtered_42602 = file_42602[file_42602['Parameter Name'].isin(pollutants)][['Date Local', 'Parameter Name', 'Arithmetic Mean', 'AQI']]
filtered_42401 = file_42401[file_42401['Parameter Name'].isin(pollutants)][['Date Local', 'Parameter Name', 'Arithmetic Mean', 'AQI']]
filtered_42101 = file_42101[file_42101['Parameter Name'].isin(pollutants)][['Date Local', 'Parameter Name', 'Arithmetic Mean', 'AQI']]
filtered_44201 = file_44201[file_44201['Parameter Name'].isin(pollutants)][['Date Local', 'Parameter Name', 'Arithmetic Mean', 'AQI']]

# Concatenate the filtered datasets into one dataframe
combined_pollutant_data = pd.concat([filtered_42602, filtered_42401, filtered_42101, filtered_44201])

# Convert 'Date Local' to datetime
combined_pollutant_data['Date Local'] = pd.to_datetime(combined_pollutant_data['Date Local'])

# One-hot encode the 'Parameter Name' (e.g., PM2.5 or Ozone)
combined_pollutant_data = pd.get_dummies(combined_pollutant_data, columns=['Parameter Name'])

# Handle missing pollutants in the dataset
columns_to_use = ['Arithmetic Mean', 'AQI']
if 'Parameter Name_Ozone' in combined_pollutant_data.columns:
    columns_to_use.append('Parameter Name_Ozone')
if 'Parameter Name_PM2.5' in combined_pollutant_data.columns:
    columns_to_use.append('Parameter Name_PM2.5')

# Select relevant columns
X = combined_pollutant_data[columns_to_use]

# Normalize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create binary labels based on AQI levels (e.g., AQI above 100 is considered an anomaly)
threshold = 100
y = (combined_pollutant_data['AQI'] > threshold).astype(int)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Reshape inputs for LSTM and CNN models
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# Early stopping callback to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Define the LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define the CNN model
def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(64, kernel_size=2, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the LSTM model with early stopping
lstm_model = create_lstm_model((X_train.shape[1], 1))
history_lstm = lstm_model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Train the CNN model with early stopping
cnn_model = create_cnn_model((X_train.shape[1], 1))
history_cnn = cnn_model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluate LSTM Model
lstm_accuracy = lstm_model.evaluate(X_test, y_test)[1]
print("LSTM Test Accuracy:", lstm_accuracy)

# Evaluate CNN Model
cnn_accuracy = cnn_model.evaluate(X_test, y_test)[1]
print("CNN Test Accuracy:", cnn_accuracy)

# Classification report for LSTM Model
y_pred_lstm = (lstm_model.predict(X_test) > 0.5).astype("int32")
print("LSTM Classification Report:")
print(classification_report(y_test, y_pred_lstm))

# Confusion matrix for LSTM Model
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lstm)
plt.title("LSTM Confusion Matrix")
plt.show()

# Classification report for CNN Model
y_pred_cnn = (cnn_model.predict(X_test) > 0.5).astype("int32")
print("CNN Classification Report:")
print(classification_report(y_test, y_pred_cnn))

# Confusion matrix for CNN Model
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_cnn)
plt.title("CNN Confusion Matrix")
plt.show()

# Plot training & validation accuracy for LSTM
plt.plot(history_lstm.history['accuracy'])
plt.plot(history_lstm.history['val_accuracy'])
plt.title('LSTM Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation accuracy for CNN
plt.plot(history_cnn.history['accuracy'])
plt.plot(history_cnn.history['val_accuracy'])
plt.title('CNN Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot ROC Curve for LSTM Model
fpr_lstm, tpr_lstm, _ = roc_curve(y_test, y_pred_lstm)
roc_auc_lstm = auc(fpr_lstm, tpr_lstm)
plt.plot(fpr_lstm, tpr_lstm, label=f'LSTM AUC = {roc_auc_lstm:.2f}')
plt.title("LSTM ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# Plot ROC Curve for CNN Model
fpr_cnn, tpr_cnn, _ = roc_curve(y_test, y_pred_cnn)
roc_auc_cnn = auc(fpr_cnn, tpr_cnn)
plt.plot(fpr_cnn, tpr_cnn, label=f'CNN AUC = {roc_auc_cnn:.2f}')
plt.title("CNN ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
