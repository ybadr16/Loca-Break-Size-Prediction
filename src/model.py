from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.regularizers import l1_l2
import numpy as np
import keras
from .custom_metrics import custom_loss
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def create_model(input_shape = (300, 10)):
    model = Sequential([
        LSTM(units=18, return_sequences=False, input_shape=input_shape, kernel_regularizer=l1_l2(0.013, 0.013)),
        Dense(8, activation='linear'),
        Dense(1, activation='linear', kernel_regularizer=l1_l2(0.013, 0.013))
    ])
    model.compile(loss=custom_loss, optimizer=Adam(learning_rate=0.001), metrics=['RootMeanSquaredError'])
    return model

def train_and_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, mean_y_train, std_y_train):
    model = create_model()
    cp = ModelCheckpoint('model/best_model.keras', save_best_only=True)
    model.fit(X_train, y_train, batch_size=32, validation_data=(X_val, y_val), epochs=2000, callbacks=[cp])
    model = keras.models.load_model('model/best_model.keras', custom_objects={'custom_loss': custom_loss})

    predictions = postprocess_output(model.predict(X_test).flatten(), mean_y_train, std_y_train)
    actual = postprocess_output(y_test.flatten(), mean_y_train, std_y_train)

    mae = mean_absolute_error(actual, predictions)
    mse = mean_squared_error(actual, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predictions)

    return mae, mse, rmse, r2, predictions, actual
