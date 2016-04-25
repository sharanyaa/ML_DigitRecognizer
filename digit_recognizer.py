from sklearn import datasets, svm, metrics, cross_validation
from sklearn.ensemble import RandomForestClassifier
import pandas
import numpy

def calc_metrics(y_test, y_pred):
    acc = metrics.accuracy_score(y_test, y_pred)  # check accuracy with observed output and expected output
    print "Accuracy: ", acc
    f1 = metrics.f1_score(y_test, y_pred, average = "macro")
    print "F1 Score: ", f1
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print "Confusion Matrix:\n", conf_matrix
    report = metrics.classification_report(y_test, y_pred)
    print "Classification Report:\n", report

def write_output_file(fname, test_pred):
    numpy.savetxt(fname, numpy.c_[test_pred], comments = '', fmt = '%d')

def RandomForest(X_train, X_test, y_train, y_test, test):
    #print "\nxtrain: ", X_train, "\nxtest: ", X_test, "\nytrain:", y_train, "\nytest:", y_test
    forest = RandomForestClassifier(n_estimators = 100)
    forest.fit(X_train, y_train) # build a model
    y_pred= forest.predict(X_test) # test the model
    calc_metrics(y_test, y_pred)

    test_pred = forest.predict(test)
    #print "\n test_pred: ", test_pred
    write_output_file('forest_output.csv', test_pred)

def SVM_classifier(X_train, X_test, y_train, y_test, test):
    svc = svm.SVC(kernel = "linear", C = 1, gamma=0.1)
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    calc_metrics(y_test, y_pred)

    test_pred = svc.predict(test)
    # print "\n test_pred: ", test_pred
    write_output_file('svm_output.csv', test_pred)

# x_train: training input, y_train: labels for input/ground truth
# x_test: testing input, y_test: expected output/labels for tests
train = pandas.read_csv("train.csv")
features = train.columns[1:]
X = train[features]
y = train['label']
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X / 255., y, test_size = 0.20, random_state
= 0)
test = pandas.read_csv("data/test.csv").values
#print "test:\n ", test
# print "\n train: ", train
# print "\n features:",features
# print "\n X: ", X
# print "\n y: ", y
# print "\n X/255.: ", X/255.

print "\nSVM CLASSIFICATION... \n"
SVM_classifier(X_train, X_test, y_train, y_test, test)
#print "\nRANDOM FOREST CLASSIFICATION...\n"
#RandomForest(X_train, X_test, y_train, y_test, test)
