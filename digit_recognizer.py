import os
from sklearn import datasets, svm, metrics, cross_validation
from sklearn.ensemble import RandomForestClassifier
import pandas
import numpy

def RandomForest():
    dataset = pandas.read_csv("data/train.csv")
    print "\ndataset:", dataset
    target = dataset[[0]].values.ravel()
    train = dataset.iloc[:, 1:].values
    test = pandas.read_csv("data/test.csv").values
    print "\ntestset:", test
    forest = RandomForestClassifier(n_estimators = 100)
    print "'\ntrain:", train, "\ntarget", target
    forest.fit(train, target)
    pred = forest.predict(test)
    numpy.savetxt('output.csv', numpy.c_[range(1, len(test) + 1), pred], delimiter = ',', comments = '', fmt = '%d')

def accuracy (predictions):
    count = 0.0
    for i in range(len(predictions)):
        if predictions[i] == train["label"][i]:
            count = count + 1.0

    accuracy = count / len(predictions)
    print "--- Accuracy value is " + str(accuracy)
    return accuracy

def RandomForest2():
    train = pandas.read_csv("data/train.csv")
    test = pandas.read_csv("data/test.csv")

    predictors = []
    for i in range(784):
        string = "pixel" + str(i)
        predictors.append(string)
    print "\npre"
    # Initialize our algorithm with the default paramters
    # n_estimators is the number of trees we want to make
    # min_samples_split is the minimum number of rows we need to make a split
    # min_samples_leaf is the minimum number of samples we can have at the place where a tree branch ends (the bottom points of the tree)
    alg = RandomForestClassifier(random_state = 1, n_estimators = 150, min_samples_split = 2, min_samples_leaf = 1)

    print "Using " + str(alg)+ " \n"
    # Compute the accuracy score for all the cross validation folds.
    scores = cross_validation.cross_val_score(alg, train[predictors], train["label"], cv = 3)

    # Take the mean of the scores (because we have one for each fold)
    print scores
    print"Cross validation scores = " + str(scores.mean())

    full_predictions = []
    # Fit the algorithm using the full training data.
    alg.fit(train[predictors], train["label"])
    # Predict using the test dataset.
    predictions = alg.predict_proba(train[predictors]).astype(float)
    predictions = predictions.argmax(axis = 1)

    submission = pandas.DataFrame({
        "true value": train["label"],
        "label"     : predictions
    })

    accuracyV = accuracy(predictions)

    # Compute accuracy by comparing to the training data.
    # accuracy = (sum(predictions[predictions == train["label"]])).astype(float) / len(predictions)
    # print accuracy

    filename = str('%0.5f' % accuracyV) + "_test_mnist.csv"
    submission.to_csv(filename, index = False)

    full_predictions = []
    # Fit the algorithm using the full training data.
    alg.fit(train[predictors], train["label"])
    # Predict using the test dataset.  We have to convert all the columns to floats to avoid an error.
    predictions = alg.predict_proba(test[predictors]).astype(float)

    predictions = predictions.argmax(axis = 1)
    ImageId = []
    for i in range(1, 28001):
        ImageId.append(i)

    submission = pandas.DataFrame({
        "ImageId": ImageId,
        "Label"  : predictions
    })

    submission.to_csv("kaggle_mnist.csv", index = False)

    # Score on kaggle mnist competition = 0.96614
    print "End of program"


RandomForest()
#RandomForest2()