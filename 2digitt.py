from sklearn.ensemble import RandomForestClassifier
import numpy
import pandas

# create the training & test sets, skipping the header row with [1:]
dataset = pandas.read_csv("data/train.csv")
target = dataset[[0]].values.ravel()
train = dataset.iloc[:, 1:].values
test = pandas.read_csv("data/test.csv").values

# create and train the random forest
# multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
rf = RandomForestClassifier(n_estimators = 100)
rf.fit(train, target)
pred = rf.predict(test)

numpy.savetxt('submission_rand_forest.csv', numpy.c_[range(1, len(test) + 1), pred], delimiter = ',',
           header = 'ImageId,Label', comments = '', fmt = '%d')