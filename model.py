import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle


heart_data = pd.read_csv('Heart Disease data.csv')

heart_data.head()

heart_data.shape

heart_data.info()

heart_data.isnull().sum()

heart_data.describe()

heart_data['target'].value_counts()

x = heart_data.drop(columns='target', axis=1)
y = heart_data['target']

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

print(x.shape, x_train.shape, x_test.shape)

model = LogisticRegression()
model.fit(x_train, y_train)
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)

print("Accuracy on training data :", training_data_accuracy)

x_test_prediction = model.predict(x_test)
testing_data_accuracy = accuracy_score(x_test_prediction, y_test)

print("Accuracy on testing data :", testing_data_accuracy)

input_data = (41, 0, 130, 204, 0, 0, 172, 0, 1.4, 2, 0, 2, 6)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = model.predict(input_data_reshaped)

print(prediction)
pickle.dump(model, open("model.pkl", "wb"))
