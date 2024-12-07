import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Loaded the data from a pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Converted the data and labels to numpy arrays for further processing
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the data into training and testing sets (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Initialized the RandomForestClassifier model
model = RandomForestClassifier()

# Trained the model using the training data
model.fit(x_train, y_train)

# Predicted the labels for the test data
y_predict = model.predict(x_test)

# Calculated the accuracy score of the model
score = accuracy_score(y_test, y_predict)

# Printed the classification accuracy percentage
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Saved the trained model to a pickle file for future use
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)