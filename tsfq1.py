import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load the dataset
url = "http://bit.ly/w-data"
data = pd.read_csv(url)
# print(data.head()) # print first 5 data columns

# Visualize the data
plt.subplot(1, 2, 1)
sns.scatterplot(x='Hours', y='Scores', data=data)
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
# plt.show()

# Prepare the data
X = data[['Hours']].values
y = data['Scores'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions
y_pred = regressor.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Visualize the regression line
plt.subplot(1, 2, 2)
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, regressor.predict(X_train), color='red')
plt.title('Hours vs Percentage (Training set)')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()

h = float(input('Enter number of hours:'))  # Convert input to float
hours = np.array([[h]])  # Ensure it's in the correct shape for prediction
predicted_score = regressor.predict(hours)

if(predicted_score[0] > 100):
    print(f'Predicted score for {h} hours/day: 100')
else:
    print(f'Predicted score for {h} hours/day: {predicted_score[0]}')

# In scikit-learn, the predict method returns a numpy array of predictions, even if there's only a single prediction. That's why predicted_score is an array, and you access the first (and only) element with predicted_score[0].

