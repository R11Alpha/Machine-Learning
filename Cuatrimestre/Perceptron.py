import pandas as pd

dataset = pd.read_csv(r"C:/Users/canch/Documents/Codigos/MachineLearning/ClassWork2/Social_Network_Ads.csv", sep = ',')

#print(dataset)

dataset = dataset.drop(columns = 'User ID')

#print(dataset)

dummy_columns = {
    'Gender': {
        'prefix' : 'Gender',
        'sep' : ','
    }
}

for column_name, dummy_data in dummy_columns.items():
   dummies = dataset[column_name].str.get_dummies(sep = dummy_data['sep'])

   dummies.columns = map(lambda col: f'{dummy_data["prefix"]}_{col}', dummies.columns)

   dataset = pd.concat([dataset, dummies], axis = 1)

dataset = dataset.drop(columns=dummy_columns.keys())

# print(dataset.head())

from sklearn.linear_model import Perceptron

xtrain = dataset.iloc[:319, 0:4]
ytrain = dataset.iloc[:319, 4]
xtest = dataset.iloc[320:, 0:4]
ytest = dataset.iloc[320:, 4]

# print(xtrain.head(1))
# print(ytrain.head(1))
# print(xtest.head(1))
# print(ytest.head(1))


# X, y = load_digits(return_X_y=True)
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(xtrain, ytrain)
Perceptron()
print(clf.score(xtest, ytest))