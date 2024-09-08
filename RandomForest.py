"""
Obs: alterar a linha 65 com o caminho da pasta dos
executáveis do graphviz.

"""

from ucimlrepo import fetch_ucirepo # Dataset
# source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

# Modeling
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics # analyzing

import numpy as np # Data Processing

# Graphic Vizualisation
from sklearn.tree import export_graphviz
import pydotplus
import os



# fetch dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

# data (as pandas dataframes)
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets
Y = np.ravel(y)

# splitting the data into 70% for training and 30% for testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)

# creating a RandomForest (RF) classifier with some customized Hyperparameters
clf = RandomForestClassifier(n_estimators=100, min_samples_split=2, min_samples_leaf=1)

# training the model
clf.fit(X_train, y_train)

# performing predictions on the test dataset
y_pred = clf.predict(X_test)

# calculating and printing the accuracy based on the label dataset
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
# Precision result for on 'B' (Benign)
print("Precision (Benign): ", metrics.precision_score(y_test, y_pred, pos_label='B'))
# Precision result for on 'M' (Malign)
print("Precision (Malign):", metrics.precision_score(y_test, y_pred, pos_label='M'))
# False Positive ratio for 'Benign' (good recall = 1 and worst is 0)
print("Recall: ", metrics.recall_score(y_test, y_pred, pos_label='B'))

# Mapping 'B' = 1 and 'M' = 0 on the 'data_' lists to measure mean_squared_error
data_pred = []
for chr in y_pred:
    data_pred.append(1 if chr == 'B' else 0)
data_test = []
for chr in y_test:
    data_test.append(1 if chr == 'B' else 0)
print("Mean Squared Error: ", metrics.mean_squared_error(data_test, data_pred))

# Visualizing the first 3 trees from the RF (configure graphviz path)
classes_names = ["Malignant", "Benign"]
os.environ["PATH"] += os.pathsep + 'C:\\Bin\\Graphviz-12.1.0-win32\\bin'

for idx in range(3):
    DecisionTree = clf.estimators_[idx] # getting the tree
    # generating a DOT format graph
    dot_data = export_graphviz(DecisionTree, feature_names=X_train.columns, filled=True,
                                class_names=classes_names, max_depth=2, impurity=False,
                                proportion=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data) # generates the graph
    image_name = "arvore_" + str(idx) + ".png"
    graph.write_png(image_name) # creates the image in same directory
    