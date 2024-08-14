import numpy as np
import matplotlib.pyplot as plt
import ast

with open('results.txt', 'r') as file:
    arr = file.readlines()


# Step 3: Convert the string representation of the list to an actual list
arr = ast.literal_eval(arr[0])


print(np.mean(arr, axis = 0))


metrics = np.array(arr)
cumulative_means = np.cumsum(metrics, axis=0) / np.arange(1, metrics.shape[0] + 1)[:, None]

# Extract the cumulative means for each metric
cumulative_MAE = cumulative_means[:, 0]
cumulative_MSE = cumulative_means[:, 1]
cumulative_RMSE = cumulative_means[:, 2]
cumulative_R2 = cumulative_means[:, 3]
cumulative_Accuracy = cumulative_means[:, 4]



plt.figure(figsize=(8, 6))
plt.plot(cumulative_MAE)
plt.axhline(y=5.185992, color='r', linestyle='--')
plt.title('Cumulative Mean of MAE')
plt.xlabel('Iterations')
plt.ylabel('MAE')
plt.ylim(0, 10)
plt.savefig('cumulative_MAE.png')
plt.show()

# Plot cumulative mean of MSE
plt.figure(figsize=(8, 6))
plt.plot(cumulative_MSE)
plt.axhline(y=76.50258, color='r', linestyle='--')
plt.title('Cumulative Mean of MSE')
plt.xlabel('Iterations')
plt.ylabel('MSE')
plt.ylim(0, 150)
plt.savefig('cumulative_MSE.png')
plt.show()

# Plot cumulative mean of RMSE
plt.figure(figsize=(8, 6))
plt.plot(cumulative_RMSE)
plt.axhline(y=7.953842, color='r', linestyle='--')
plt.title('Cumulative Mean of RMSE')
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.ylim(0, 14)
plt.savefig('cumulative_RMSE.png')
plt.show()

# Plot cumulative mean of R2
plt.figure(figsize=(8, 6))
plt.plot(cumulative_R2)
plt.axhline(y=0.888586, color='r', linestyle='--')
plt.title('Cumulative Mean of R2')
plt.xlabel('Iterations')
plt.ylabel('R2')
plt.ylim(0, 1)
plt.savefig('cumulative_R2.png')
plt.show()

# Plot cumulative mean of Accuracy
plt.figure(figsize=(8, 6))
plt.plot(cumulative_Accuracy)
plt.axhline(y=80.684211, color='r', linestyle='--')
plt.title('Cumulative Mean of Accuracy')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.ylim(0, 100)
plt.savefig('cumulative_Accuracy.png')
plt.show()

