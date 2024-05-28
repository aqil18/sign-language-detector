import pickle

from sklearn.ensemble import RandomForestClassifier # The classifier we will use
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb')) # Open up our dictionary dataset

data = np.asarray(data_dict['data']) # Must convert to np array to use with classifier
labels = np.asarray(data_dict['labels']) 

# Splitting the data
# Stratify labels make sure that there are equal proportions of each class in either split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels) 

# Call in model
model = RandomForestClassifier()

# Fit the training data onto model to train the classifier
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

# Collecting metrics of the classifier we have made
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

# Save the model for use later
file = open('model.p', 'wb')
pickle.dump({'model': model}, file)
file.close()
