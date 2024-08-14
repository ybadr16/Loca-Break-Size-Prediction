import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data_processing import load_data, preprocess, preprocess_output, postprocess_output
from src.utils import percentage_within_tolerance
from src.model import train_and_evaluate_model


def main():
    path = './Data/WSC_Data/'
    sequences = load_data(path)
    y_data = [i for i in range(1, 101) if i not in {17, 19, 22, 23}]
    y_data += [i + 0.5 for i in range(0, 13)] + [i + 0.5 for i in range(
        20, 100) if i + 0.5 not in {21.5, 22.5, 27.5}]
    y_data.sort()

    sequences = np.array(sequences)
    y_data = np.array(y_data)

    num_samples = len(sequences)
    indices = np.arange(num_samples)
    train_size = int(0.7 * num_samples)
    val_size = int(0.2 * num_samples)

    results = []
    print(len(sequences), len(y_data))
    for run in range(100):
        print(f'This is run number {run + 1}')
        np.random.shuffle(indices)
        shuffled_data = sequences[indices]
        shuffled_targets = y_data[indices]

        X_train, y_train = shuffled_data[:train_size], shuffled_targets[:train_size]
        X_val, y_val = shuffled_data[train_size:train_size +
                                     val_size], shuffled_targets[train_size:train_size + val_size]
        X_test, y_test = shuffled_data[train_size +
                                       val_size:], shuffled_targets[train_size + val_size:]

        means = np.mean(X_train, axis=(0, 1))
        stds = np.std(X_train, axis=(0, 1))
        mean_y_train = np.mean(y_train)
        std_y_train = np.std(y_train)

        X_train = preprocess(X_train, means, stds)
        X_val = preprocess(X_val, means, stds)
        X_test = preprocess(X_test, means, stds)
        y_train = preprocess_output(y_train, mean_y_train, std_y_train)
        y_val = preprocess_output(y_val, mean_y_train, std_y_train)
        y_test = preprocess_output(y_test, mean_y_train, std_y_train)

        y_train = np.expand_dims(y_train, axis=-1)
        y_val = np.expand_dims(y_val, axis=-1)
        y_test = np.expand_dims(y_test, axis=-1)

        mae, mse, rmse, r2, predictions, actual = train_and_evaluate_model(
            X_train, y_train, X_val, y_val, X_test, y_test, mean_y_train, std_y_train
        )

        accuracy = percentage_within_tolerance(
            actual, predictions, tolerance=0.15)
        results.append([mae, mse, rmse, r2, accuracy])

        predictions_vs_actuals_df = pd.DataFrame(
            {'Actual': actual, 'Predictions': predictions})
        predictions_vs_actuals_df.to_csv(
            f'results/predictions_vs_actuals_run_{run + 1}.csv', index=False)

        with open('results/results.txt', 'w') as file:
            file.writelines(str(results))

    results_array = np.array(results)
    mean_results = np.mean(results_array, axis=0)
    mean_results_df = pd.DataFrame([mean_results], columns=[
                                   'MAE', 'MSE', 'RMSE', 'R2', 'Accuracy'])
    mean_results_df.to_csv('results/mean_results.csv', index=False)

    print("Mean Results:")
    print(mean_results_df)


if __name__ == "__main__":
    main()
