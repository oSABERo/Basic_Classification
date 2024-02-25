import pandas as pd

# Read the data
df = pd.read_csv("breast_cancer.csv")

x = df.iloc[:, 2:].values # df[['a', 'b', ...]]

y = df.iloc[:, 1].values
map_rule = {'M': 0, 'B': 1}
y = y.map(map_rule)


# Train and Test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

from sklearn.svm import SVC

clf = SVC()
clf.fit(x_train, y_train)

y_predict = clf.predict(x_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

acc = accuracy_score(y_test, y_predict)
precis = precision_score(y_test, y_predict)
recall = recall_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)

print("Accuracy score: ", acc)
print("Precision score:", precis)
print("Recall score:   ", recall)
print("F1 score:       ", f1)