#Machine Learning Overview
#Machine Learning is a way of taking data and turning it into insights. We use computer power to analyze examples from the past to build a model that can predict the result for new examples.
#Pandas is used for reading data and data manipulation, 
#numpy is used for computations of numerical data, Numpy is a python library that allows fast and easy mathematical operations to be performed on arrays.
#matplotlib is used for graphing data,  
#scikit-learn is used for machine learning models.
#Supervised learning is when we have a known target based on past data (for example, predicting what price a house will sell for). 
#Supervised Learning means that we will have labeled historical data that we will use to inform our model. We call the label or thing we’re trying to predict, the target. 
#So in supervised learning, there is a known target for the historical data, and for unsupervised learning there is no known target.
#unsupervised learning is when there isn't a known past answer (for example, determining the topics discussed in restaurant reviews)
#Within supervised learning, there are classification and regression problems.
#Regression is predicting a numerical value (for example, predicting what price a house will sell for)
#classification is predicting what class something belongs to (for example, predicting if a borrower will default on their loan). Classification problems are where the target is a categorical value (often True or False, but can be multiple categories)

#Averages
#The mean is the most commonly known average.
#Add up all the values and divide by the number of values
#The median is the value in the middle
#In statistics, both the mean and the median are called averages. The layman’s average is the mean

#Percentiles
#The median can also be thought of as the 50th percentile

#Standard Deviation & Variance
#The standard deviation and variance are measures of how dispersed or spread out the data is.
#Even though data is never a perfect normal distribution, we can still use the standard deviation to gain insight about how the data is distributed.


#Selecting a Single Column
#To select a single column, we use the square brackets and the column name.
col = df['Fare']
print(col)
#The result is what we call a Pandas Series. A series is like a DataFrame, but it's just a single column

#Selecting Multiple Columns
small_df = df[['Age', 'Sex', 'Survived']]
print(small_df.head())
#When selecting a single column from a Pandas DataFrame, we use single square brackets. When selecting multiple columns, we use double square brackets.

#We create a Pandas Series that will be a series of Trues and Falses (True if the passenger is male and False if the passenger is female).
df['Sex'] == 'male'

#Now we want to create a column with this result. To create a new column, we use the same bracket syntax (df['male']) and then assign this new value to it.
df['male'] = df['Sex'] == 'male'

#Often our data isn’t in the ideal format. Luckily Pandas makes it easy for us to create new columns based on our data so we can format it appropriately.

#Then we use the values attribute to get the values as a numpy array.
df['Fare'].values
#The result is a 1-dimensional array. You can tell since there's only one set of brackets and it only expands across the page (not down as well)
#The values attribute of a Pandas Series give the data as a numpy array

df[['Pclass', 'Fare', 'Age']].values 
#The values attribute of a Pandas DataFrame give the data as a 2d numpy array.

#Masking
#We create what we call a mask first. This is an array of boolean values (True/False) of whether the passenger is a child or not.
mask = arr[:, 2] < 18
print(arr[mask])
print(arr[arr[:, 2] < 18])
#A mask is a boolean array (True/False values) that tells us which values from the array we’re interested in

#Summing and Counting
arr = df[['Pclass', 'Fare', 'Age']].values
mask = arr[:, 2] < 18
print(mask.sum())   #So we can just sum up the array and that’s equivalent to counting the number of true values.
print((arr[:, 2] < 18).sum())  #we don’t need to define the mask variable.
#Summing an array of boolean values gives the count of the number of True values.

#Scatter Plot
import matplotlib.pyplot as plt
#We use the scatter function to plot our data. The first argument of the scatter function is the x-axis (horizontal direction) and the second argument is the y-axis (vertical direction).
plt.scatter(df['Age'], df['Fare'])
#To make it easier to interpret, we can add x and y labels.
plt.xlabel('Age')
plt.ylabel('Fare')

#We can also use our data to color code our scatter plot. This will give each of the 3 classes a different color. We add the c parameter and give it a Pandas series. In this case, our Pandas series has 3 possible values (1st, 2nd, and 3rd class), so we'll see our datapoints each get one of three colors.
plt.scatter(df['Age'], df['Fare'], c=df['Pclass'])
#A scatter plot is used to show all the values from your data on a graph. In order to get a visual representation of our data, we have to limit our data to two features.

#eg. Write the code to create a scatter plot with Pclass on the y-axis and Fare on the x-axis. Color code it according to whether or not they survived. Add the labels “Fare” and “Pclass” on the x and y axes respectively.
plt.scatter(df['Fare'], df['Pclass'], c=df['survived'])
plt.xlabel('Fare')
plt.ylabel('Pclass')


#Line
#Now that we can put individual datapoints on a plot, let's see how to draw the line. The plot function does just that. The following draws a line to approximately separate the 1st class from the 2nd and 3rd class. From eyeballing, we’ll put the line from (0, 85) to (80, 5). Our syntax below has a list of the x values and a list of the y values.
plt.plot([0, 80], [85, 5])
#In matplotlib, we use the scatter function to create a scatter plot and the plot function for a line.


#Equation for the Line
#A line is defined by an equation in the following form:
0 = ax + by + c
#The values a, b, and c are the coefficients. Any three values will define a unique line.
#Let’s look at a specific example of a line where the coefficients are a=1, b=-1 and c=-30.
0 = (1)x + (-1)y + (-30)
#The three coefficients are: 1, -1, -30
#The coefficients of the line are what control where the line is.

#Making a Prediction Based on the Line
0 = (1)x + (-1)y - 30 

#If we take a passenger’s data, we can use this equation to determine which side of the line they fall on. For example, let’s say we have a passenger whose Fare is 100 and Age is 20.
#Let’s plug in these values to our equation:
(1)100 + (-1)20 - 30 = 100 - 20 - 30 = 50 

#Build a Logistic Regression Model with Sklearn
from sklearn.linear_model import LogisticRegression 
model = LogisticRegression()   #We first instantiate the class
X = df[['Fare', 'Age']].values
y = df['Survived'].values
model.fit(X, y)
print(model.coef_, model.intercept_)

#Make Predictions with the Model
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values
model = LogisticRegression()
model.fit(X, y)
model.predict(X)   #we can use the predict method to make predictions.
#Note that even with one datapoint, the predict method takes a 2-dimensional numpy array and returns a 1-dimensional numpy array.
#Let’s see what the model predicts for the first 5 rows of data and compare it to our target array. We get the first 5 rows of data with X[:5] and the first 5 values of the target with y[:5].
print(model.predict(X[:5])) 
# [0 1 1 1 0]
print(y[:5]) 
# [0 1 1 1 0]

#Score the model
#We can get a sense of how good our model is by counting the number of datapoints it predicts correctly. This is called the accuracy score
#Let’s create an array that has the predicted y values.
y_pred = model.predict(X)
y == y_pred   #we create an array of boolean values of whether or not our model predicted each passenger correctly
print((y == y_pred).sum() / y.shape[0]) 

#This is a common enough calculation, that sklearn has already implemented it for us. So we can get the same result by using the score method. The score method uses the model to make a prediction for X and counts what percent of them match y.
print(model.score(X, y))


#Introducing the Breast Cancer Dataset
from sklearn.datasets import load_breast_cancer
cancer_data = load_breast_cancer()
print(cancer_data.keys())
print(cancer_data['DESCR'])
#In the breast cancer dataset, there are several features that are calculated based on other columns. The process of figuring out what additional features to calculate is feature engineering.

#Loading the Data into Pandas
df = pd.DataFrame(cancer_data['data'])
columns = cancer_data['feature_names']
df['target'] = cancer_data['target']
print(df.head())

#Build a Logistic Regression Model
X = df[cancer_data.feature_names].values
y = df['target'].values
model = LogisticRegression() #we create a Logistic Regression object and use the fit method to build the model.
model.fit(X, y)
model = LogisticRegression(solver='liblinear')
model.fit(X, y)
model.predict([X[0]])  #the model predicts for the first datapoint in our dataset.
model.score(X, y)


#Machine Learning - Bob the Builder
#
#Building a Logistic Regression model.
#Task
#You are given a feature matrix and a single datapoint to predict. Your job will be to build a Logistic Regression model with the feature matrix and make a prediction (1 or 0) of the single datapoint.
#Input Format
#First line: Number of data points in the feature matrix (n)
#Next n lines: Values of the row in the feature matrix, separated by spaces
#Next line: Target values separated by spaces
#Final line: Values (separated by spaces) of a single datapoint without a target value
#
#Output Format
#Either 1 or 0
#
#Sample Input
#6
#1 3
#3 5
#5 7
#3 1
#5 3
#7 5
#1 1 1 0 0 0
#2 4
#
#Sample Output
1

import numpy as np
n = int(input())
X = []
for i in range(n):
    X.append([float(x) for x in input().split()])
y = [int(x) for x in input().split()]
datapoint = [float(x) for x in input().split()]

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X,y)
datapoint = np.array(datapoint).reshape(1,-1)
print(model.predict(datapoint[[0]])[0])


#Confusion Matrix
#We can see all the important values in what is called the Confusion Matrix (or Error Matrix or Table of Confusion).

#The Confusion Matrix is a table showing four values:
#• Datapoints we predicted positive that are actually positive
#• Datapoints we predicted positive that are actually negative
#• Datapoints we predicted negative that are actually positive
#• Datapoints we predicted negative that are actually negative

#The confusion matrix fully describes how a model performs on a dataset, though is difficult to use to compare models.
#We have names for each square of the confusion matrix.

#A true positive (TP) is a datapoint we predicted positively that we were correct about.
#A true negative (TN) is a datapoint we predicted negatively that we were correct about.
#A false positive (FP) is a datapoint we predicted positively that we were incorrect about.
#A false negative (FN) is a datapoint we predicted negatively that we were incorrect about.

#The four values of the confusion matrix (TP, TN, FP, FN) are used to compute several different metrics that we’ll use later on.

#Precision
#Two commonly used metrics for classification are precision and recall. 
#Conceptually, precision refers to the percentage of positive results which are relevant and recall to the percentage of positive cases correctly classified
#Precision is a measure of how precise the model is with its positive predictions.

#Recall
#Recall is the percent of positive cases that the model predicts correctly. Again, we will be using the confusion matrix to compute our result.
#Recall is a measure of how many of the positive cases the model can recall.

#Precision & Recall Trade-off
#We often will be in a situation of choosing between increasing the recall (while lowering the precision) or increasing the precision (and lowering the recall). 
#It will depend on the situation which we’ll want to maximize.
#There’s no hard and fast rule on what values of precision and recall you’re shooting for. It always depends on the dataset and the application.

#F1 Score
#Accuracy was an appealing metric because it was a single number. Precision and recall are two numbers so it’s not always obvious how to choose between two models if one has a higher precision and the other has a higher recall. 
#The F1 score is an average of precision and recall so that we have a single score for our model. The F1 score is the harmonic mean of the precision and recall values.

#Accuracy, Precision, Recall & F1 Score in Sklearn
#let’s start by recalling our code from the previous module to build a Logistic Regression model
df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values
model = LogisticRegression()
model.fit(X, y)
y_pred = model.predict(X)

#Let’s import them from scikit-learn.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print("accuracy:", accuracy_score(y, y_pred))
print("precision:", precision_score(y, y_pred))
print("recall:", recall_score(y, y_pred))
print("f1 score:", f1_score(y, y_pred))

#Confusion Matrix in Sklearn
#Scikit-learn has a confusion matrix function that we can use to get the four values in the confusion matrix (true positives, false positives, false negatives, and true negatives). Assuming y is our true target values and y_pred is the predicted values, we can use the confusion_matrix function as follows
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y, y_pred))
#Since negative target values correspond to 0 and positive to 1, scikit-learn has ordered them in this order. Make sure you double check that you are interpreting the values correctly!

#Overfitting
#This is artificially inflating our numbers since our model, in effect, got to see the answers to the quiz before we gave it the quiz. This can lead to what we call overfitting. Overfitting is when we perform well on the data the model has already seen, but we don’t perform well on new data.
#The more features we have in our dataset, the more prone we’ll be to overfitting.

#Training Set and Test Set
#To simulate making predictions on new unseen data, we can break our dataset into a training set and a test set. The training set is used for building the models.
#The test set is used for evaluating the models. We split our data before building the model, thus the model has no knowledge of the test set and we’ll be giving it a fair assessment.

#Training and Testing in Sklearn
#Assuming we have a 2-dimensional numpy array X of our features and a 1-dimensional numpy array y of the target, we can use the train_test_split function
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
print("whole dataset:", X.shape, y.shape)
print("training set:", X_train.shape, y_train.shape)
print("test set:", X_test.shape, y_test.shape)

#Building a Scikit-learn Model Using a Training Set
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

#Using a Random State - to ensure that we get the same random split every time the same code is run
from sklearn.model_selection import train_test_split
x = [[1,1], [2,2], [3,3], [4,4]]
y = [0, 0, 1, 1]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=27)
print('X_train', X_train)
print('X_test', 'X_test')
#The random state is also called a seed.

#Logistic Regression Threshold
#With a Logistic Regression model, we have an easy way of shifting between emphasizing precision and emphasizing recall. 
#The Logistic Regression model doesn’t just return a prediction, but it returns a probability value between 0 and 1.
#If we make the threshold higher, we’ll have fewer positive predictions, but our positive predictions are more likely to be correct. This means that the precision would be higher and the recall lower
#On the other hand, if we make the threshold lower, we’ll have more positive predictions, so we’re more likely to catch all the positive cases. This means that the recall would be higher and the precision lower.
#Each choice of a threshold is a different model. An ROC (Receiver operating characteristic) Curve is a graph showing all of the possible models and their performance.

#Sensitivity & Specificity
#An ROC Curve is a graph of the sensitivity vs. the specificity. These values demonstrate the same trade-off that precision and recall demonstrate.
#While we generally look at precision and recall values, for graphing the standard is to use the sensitivity and specificity. It is possible to build a precision-recall curve, but this isn’t commonly done

#Sensitivity & Specificity in Scikit-learn
from sklearn.metrics import recall_score
sensitivity_score = recall_score
print(sensitivity_score(y_test, y_pred)) 
# 0.6829268292682927
#Let’s look at the output of precision_recall_fscore_support.
from sklearn.metrics import precision_recall_fscore_support
print(precision_recall_fscore_support(y, y_pred))
#The first is the recall of the negative class and the second is the recall of the positive class. The second value is the standard recall or sensitivity value, and you can see the value matches what we got above. The first value is the specificity.
def specificity_score(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred)
    return r[0]
print(specificity_score(y_test, y_pred)) 
# 0.9214285714285714
#Sensitivity is the same as the recall (or recall of the positive class) and specificity is the recall of the negative class.

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_recall_fscore_support
sensitivity_score = recall_score
def specificity_score(y_true, y_pred): p, r, f, s = precision_recall_fscore_support(y_true, y_pred)
return r[0]

df = df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
y = df['Survived'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("sensitivity:", sensitivity_score(y_test, y_pred))
print("specificity:", specificity_score(y_test, y_pred))

#Adjusting the Logistic Regression Threshold in Sklearn
#We can use the predict_proba function to get them.
(model.predict_proba(X_test)
model.predict_proba(X_test)[:, 1]
y_pred = model.predict_proba(X_test)[:, 1] > 0.75
print("precision:", precision_score(y_test, y_pred))
print("recall:", recall_score(y_test, y_pred))

#How to Build an ROC Curve
#The ROC curve is a graph of the specificity vs the sensitivity.
#Let’s start by looking at the code to build the ROC curve. Scikit-learn has a roc_curve function we can use. 
#The function takes the true target values and the predicted probabilities from our model.
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred_proba = model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:,1])

plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('1 - specificity')
plt.ylabel('sensitivity')
plt.show()
#As we don’t use the threshold values to build the graph, the graph does not tell us what threshold would yield each of the possible models
#The closer the curve gets to the upper left corner, the better the performance. The line should never fall below the diagonal line as that would mean it performs worse than a random model.

#Area Under the Curve
#Let’s use scikit-learn to help us calculate the area under the curve. We can use the roc_auc_score function
(roc_auc_score(y_test, y_pred_proba[:,1]) 
#It’s important to note that this metric tells us how well in general a Logistic Regression model performs on our data. As an ROC curve shows the performance of multiple models, the AUC is not measuring the performance of a single model.

#Building and Evaluating with Multiple Training and Test Sets
#This process for creating multiple training and test sets is called k-fold cross validation. The k is the number of chunks we split our dataset into. The standard number is 5, as we did in our example above.
#Our goal in cross validation is to get accurate measures for our metrics (accuracy, precision, recall). We are building extra models in order to feel confident in the numbers we calculate and report.
#Computation power for building a model can be a concern when the dataset is large. In these cases, we just do a train test split.

#KFold Class
#Scikit-learn has already implemented the code to break the dataset into k chunks and create k training and test sets.
#For simplicity, let’s take a dataset with just 6 datapoints and 2 features and a 3-fold cross validation on the dataset. We’ll take the first 6 rows from the Titanic dataset and use just the Age and Fare columns.
X = df[['Age', 'Fare']].values[:6]
y = df['Survived'].values[:6]
kf = KFold(n_splits=3, shuffle=True)  #It takes two parameters: n_splits (this is k, the number of chunks to create) and shuffle (whether or not to randomize the order of the data).
list(kf.split(X))
#The split is done randomly, so expect to see different datapoints in the sets each time you run the code.


#Creating Training and Test Sets with the Folds
#First let’s pull out the first split.
splits = list(kf.split(X))
first_split = splits[0]
print(first_split)
# (array([0, 2, 3, 5]), array([1, 4]))
#The first array is the indices for the training set and the second is the indices for the test set. Let’s create these variables.
train_indices, test_indices = first_split
print("training set indices:", train_indices)
print("test set indices:", test_indices)
# training set indices: [0, 2, 3, 5]
# test set indices: [1, 4]
#Now we can create an X_train, y_train, X_test, and y_test based on these indices
X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

print("X_train")
print(X_train)
print("y_train", y_train)
print("X_test")
print(X_test)
print("y_test", y_test)
#At this point, we have training and test sets in the same format as we did using the train_test_split function.

#Loop Over All the Folds
scores = []
kf = KFold(n_splits=5, shuffle=True)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = LogisticRegression()
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))
print(scores)
# [0.75847, 0.83146, 0.85876, 0.76271, 0.74011]
print(np.mean(scores))
# 0.79029

#Expect to get slightly different values every time you run the code. 
#The KFold class is randomly splitting up the data each time, so a different split will result in different scores, though you should expect the average of the 5 scores to generally be about the same.

#Comparing Different Models
#Evaluation techniques are essential for deciding between multiple model options.

#Building the Models with Scikit-learn
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male'] = df['Sex'] == 'male'

kf = KFold(n_splits=5, shuffle=True)

X1 = df[['Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']].values
X2 = df[['Pclass', 'male', 'Age']].values
X3 = df[['Fare', 'Age']].values
y = df['Survived'].values

print("Logistic Regression with all features")
score_model(X1, y, kf)
print()
print("Logistic Regression with all features")
score_model(X2, y, kf)
print()
print("Logistic Regression with all features")
score_model(X3, y, kf)

#Expect to get slightly different results every time you run the code. 
#The k-fold splits are chosen randomly, so there will be a little variation depending on what split each datapoint ends up in

#Now that we’ve made a choice of a best model, we build a single final model using all of the data.
model = LogisticRegression()
model.fit(X1, y)


#Machine Learning - Welcome to the Matrix
#Calculating Evaluation Metrics using the Confusion Matrix.
#
#Task
#You will be given the values of the confusion matrix (true positives, false positives, false negatives, and true negatives). Your job is to compute the accuracy, precision, recall and f1 score and print the values rounded to 4 decimal places. To round, you can use round(x, 4).
#
#Input Format
#The values of tp, fp, fn, tn, in that order separated by spaces
#
#Output Format
#Each value on its own line, rounded to 4 decimal places, in this order:
#accuracy, precision, recall, f1 score
#
#Sample Input
#233 65 109 480
#
#Sample Output
#0.8038
#0.7819
#0.6813
#0.7281

tp, fp, fn, tn = [int(x) for x in input().split()]

total = tp+ fp+ fn+ tn

accuracy = (tp + tn )/(total)  #round(x, 4)
precision = tp/ (tp+fp)
recall = tp/ (tp+fn)
f1 = round(1/(((1/precision) + (1/recall)) / 2), 4)

print(round(accuracy, 4))
print(round(precision, 4))
print(round(recall, 4))
print(f1)

#A Nonparametric Machine Learning Algorithm
#These coefficients are called parameters. Since the model is defined by these parameters, Logistic Regression is a parametric machine learning algorithm.
#In this module, we’ll introduce Decision Trees, which are an example of a nonparametric machine learning algorithm. Decision Trees won’t be defined by a list of parameters as we’ll see in the upcoming lessons
#Every machine learning algorithm is either parametric or nonparametric.

#Tree Terminology
#Each of the rectangles is called a node. The nodes which have a feature to split on are called internal nodes. 
#The very first internal node at the top is called the root node. The final nodes where we make the predictions of survived/didn’t survive are called leaf nodes. 
#Internal nodes all have two nodes below them, which we call the node’s children
#Decision Trees are often favored if you have a non-technical audience since they can easily interpret the model.
#When building the Decision Tree, we don’t just randomly choose which feature to split on first. We want to start by choosing the feature with the most predictive power

#What makes a Good Split
#The mathematical term we’ll be measuring is called information gain. This will be a value from 0 to 1 where 0 is the information gain of a useless split and 1 is the information gain of a perfect split. 
#In the next couple parts we will define gini impurity and entropy which we will use to define information gain. First we will discuss the intuition of what makes a good split

#Gini Impurity
#Gini impurity is a measure of how pure a set is. We’ll later see how we can use the gini impurity to calculate the information gain.
#Entropy is another measure of purity.

#DecisionTreeClassifier Class
from sklearn.tree import DecisionTreeClassifier
#Now we can apply the same methods that we used with the LogisticRegression class: fit (to train the model), score (to calculate the accuracy score) and predict (to make predictions)
model = DecisionTreeClassifier()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22)
model.fit(X_train, y_train)
#Note that we have the same methods for a DecisionTreeClassifier as we did for a LogisticRegression object.

#Gini vs Entropy
dt = DecisionTreeClassifer(criterion='entropy')
#We first create a k-fold split since when we’re comparing two models we want them to use the same train/test splits to be fair
kf = KFold(n_splits=5, shuffle=True)
for criterion in ['gini', 'entropy']:
    print("Decision Tree - {}".format(criterion))
    accuracy = []
    precision = []
    recall = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        dt = DecisionTreeClassifier(criterion=criterion)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        precision.append(precision_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred))
    print("accuracy:", np.mean(accuracy))
    print("precision:", np.mean(precision))
    print("recall:", np.mean(recall))

#Visualizing Decision Trees
from sklearn.tree import export_graphviz
dot_file = export_graphviz(dt, feature_names=feature_names)
#We can then use the graphviz module to convert it to a png image format.
import graphviz
graph = graphviz.Source(dot_file)
graph.render(filename='tree', format='png', cleanup=True)

from sklearn.tree import export_graphviz
import graphviz
from IPython.display import Image

feature_names = ['Pclass', 'male']
X = df[feature_names].values
y = df['Survived'].values

dt = DecisionTreeClassifier()
dt.fit(X, y)

dot_file = export_graphviz(dt, feature_names=feature_names)
graph = graphviz.Source(dot_file)
graph.render(filename='tree', format='png', cleanup=True)


#Pruning
#In order to solve these issues, we do what’s called pruning the tree. This means we make the tree smaller with the goal of reducing overfitting.
#There are two types of pruning: pre-pruning & post-pruning.
#In pre-pruning, we have rules of when to stop building the tree, so we stop building before the tree is too big
#In post-pruning we build the whole tree and then we review the tree and decide which leaves to remove to make the tree smaller.
#The term pruning comes from the same term in farming. Farmers cut off branches of trees and we are doing the same to our decision tree.

#Pre-pruning
#We’re going to focus on pre-pruning techniques since they are easier to implement. We have a few options for how to limit the tree growth. Here are some commonly used pre-pruning techniques:
#• Max depth: Only grow the tree up to a certain depth, or height of the tree. If the max depth is 3, there will be at most 3 splits for each datapoint.
#• Leaf size: Don’t split a node if the number of samples at that node is under a threshold
#• Number of leaf nodes: Limit the total number of leaf nodes allowed in the tree
#Pruning is a balance. For example, if you set the max depth too small, you won’t have much of a tree and you won’t have any predictive power. This is called underfitting. Similarly if the leaf size is too large, or the number of leaf nodes too small, you’ll have an underfit model.

#Pre-pruning Parameters
#Prepruning Technique 1: Limiting the depth
#We use the max_depth parameter to limit the number of steps the tree can have between the root node and the leaf nodes.
#Prepruning Technique 2: Avoiding leaves with few datapoints
#We use the min_samples_leaf parameter to tell the model to stop building the tree early if the number of datapoints in a leaf will be below a threshold.
#Prepruning Technique 3: Limiting the number of leaf nodes
#We use max_leaf_nodes to set a limit on the number of leaf nodes in the tree.
#Here’s the code for creating a Decision Tree with the following properties:
#• max depth of 3
#• minimum samples per leaf of 2
#• maximum number of leaf nodes of 10

dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, max_leaf_nodes=10)

#Grid Search
from sklearn.model_selection import GridSearchCV
#GridSearchCV has four parameters that we’ll use:
#1. The model (in this case a DecisionTreeClassifier)
#2. Param grid: a dictionary of the parameters names and all the possible values
#3. What metric to use (default is accuracy)
#4. How many folds for k-fold cross validation
param_grid = {
    'max_depth': [5, 15, 25],
    'min_samples_leaf': [1, 3],
    'max_leaf_nodes': [10, 20, 35, 50]}
dt = DecisionTreeClassifier()
gs = GridSearchCV(dt, param_grid, scoring='f1', cv=5)
gs.fit(X, y)
print("best params:", gs.best_params_)
#best parameters: {'max_depth': 15, 'min_samples_leaf': 1, 'max_leaf_nodes': 35}
print("best score:", gs.best_score_)

#Computation
#When talking about how much computation is required for a machine learning algorithm, we separate it into two questions: how much computation is required to build the model and how much is required to predict.
#A decision tree is very computationally expensive to build. This is because at every node we are trying every single feature and threshold as a possible split.
#Predicting with a decision tree on the other hand, is computational very inexpensive. You just need to ask a series of yes/no questions about the datapoint to get to the prediction

#Performance
#Decision Trees can perform decently well depending on the data, though as we have discussed, they are prone to overfitting. Since a leaf node can have just one datapoint that lands there, it gives too much power to individual datapoints.
#To remedy the overfitting issues, decision trees generally require some tuning to get the best possible model. Pruning techniques are used to limit the size of the tree and they help mitigate overfitting.

#Interpretability
#The biggest reason that people like choosing decision trees is because they are easily interpretable. Depending on what you’re building a model for, you might need to give a reason why you made a certain prediction. 
#A non-technical person can interpret a Decision Tree so it’s easy to give an explanation of a prediction


#Machine Learning - Split to Achieve Gain
#Calculate Information Gain.
#Task
#Given a dataset and a split of the dataset, calculate the information gain using the gini impurity.
#
#The first line of the input is a list of the target values in the initial dataset. The second line is the target values of the left split and the third line is the target values of the right split.
#
#Round your result to 5 decimal places. You can use round(x, 5).
#
#Input Format
#Three lines of 1's and 0's separated by spaces
#
#Output Format
#Float (rounded to 5 decimal places)
#
#Sample Input
#1 0 1 0 1 0
#1 1 1
#0 0 0
#
#Sample Output
#0.5

S = [int(x) for x in input().split()]
A = [int(x) for x in input().split()]
B = [int(x) for x in input().split()]

def p(a):
    x=sum(a)/len(a)
    return x

def log2(a):
    a=math.log2(a)
    return a

def gini(a):
    h=2*p(a)*(1-p(a))
    return h

infogain=gini(S)-(len(A)/len(S))*gini(A)-(len(B)/len(S))*gini(B)
print(round(infogain,5))

#Improving on Decision Trees
#Decision Trees are very susceptible to random idiosyncrasies in the training dataset. We say that Decision Trees have high variance since if you randomly change the training dataset, you may end up with a very different looking tree
#One of the advantages of decision trees over a model like logistic regression is that they make no assumptions about how the data is structured.
#We will be learning about random forests in this module, which as you may guess from the name, is a model built with multiple trees
#The goal of random forests is to take the advantages of decision trees while mitigating the variance issues.
#A random forest is an example of an ensemble because it uses multiple machine learning models to create a single model.

#Bootstrapping
#A bootstrapped sample is a random sample of datapoints where we randomly select with replacement datapoints from our original dataset to create a dataset of the same size. Randomly selecting with replacement means that we can choose the same datapoint multiple times. 
#This means that in a bootstrapped sample, some datapoints from the original dataset will appear multiple times and some will not appear at all


#Bagging Decision Trees
#Bootstrap Aggregation (or Bagging) is a technique for reducing the variance in an individual model by creating an ensemble from multiple models built on bootstrapped samples
#Bagging Decision Trees is a way of reducing the variance in the model.


#Decorrelate the Trees
#With bagged decision trees, the trees may still be too similar to have fully created the ideal model. They are built on different resamples, but they all have access to the same features. 
#Thus we will add some restrictions to the model when building each decision tree so the trees have more variation. We call this decorrelating the trees.
#A standard choice for the number of features to consider at each split is the square root of the number of features. So if we have 9 features, we will consider 3 of them at each node (randomly chosen).
#If we bag these decision trees, we get a random forest.
#Each decision tree within a random forest is probably worse than a standard decision tree. But when we average them we get a very strong model!


#Random Forest with Sklearn
#The syntax for building and using a Random Forest model is the same as it was for Logistic Regression and Decision Trees. The builders of scikit-learn intentionally made it so that it would be easy to switch between and compare different models.
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101)
#We have added the random state parameter here so that it will do the same split every time we run the code.
#Then we create the RandomForestClassifier object and use the fit method to build the model on the training set
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

first_row = X_test[0]
print("prediction:", rf.predict([first_row]))
print("true value:", y_test[0])
#We can use the score method to calculate the accuracy over the whole test set
print("random forest accuracy:", rf.score(X_test, y_test))

#We can see how this compares to the Decision Tree model.
dt = DecisionTreeClassifier()
df.fit(X_train, y_train)
print("decision tree accuracy:", dt.score(X_test, y_test))
#Note how similar the scikit-learn code is to the code for Logistic Regression and Decision Trees. This makes it very easy to try and compare different models

#Random Forest Parameters
#When you look at the scikit-learn docs for the RandomForestClassifier, you will see quite a few parameters that you can control.
#Since a random forest is made up of decision trees, we have all the same tuning parameters for prepruning as we did for decision trees: max_depth, min_samples_leaf, and max_leaf_nodes. With random forests, it is generally not important to tune these as overfitting is generally not an issue.
#We will look at two new tuning parameters: n_estimators (the number of trees) and max_features (the number of features to consider at each split).
#The default for the max features is the square root of p, where p is the number of features (or predictors). The default is generally a good choice for max features and we usually will not need to change it, but you can set it to a fixed number with the following code
rf = RandomForestClassifier(max_features=5)

rf = RandomForestClassifier(n_estimators=15)
#One of the big advantages of Random Forests is that they rarely require much tuning. The default values will work well on most datasets

#Grid Search
#If you recall from the Decision Tree module, scikit-learn has built in a Grid Search class to help us find the optimal choice of parameters
#Recall that we need to define the parameter grid of the parameters we want to vary and give a list of the values to try
 param_grid = {
    'n_estimators': [10, 25, 50, 75, 100],
}
#Now we can create a Random Forest Classifier and a Grid Search.
rf = RandomForestClassifier()
gs = GridSearchCV(rf, param_grid, cv=5)
#Now we use the fit method to run the grid search. 
gs.fit(X, y)
print("best params:", gs.best_params_)

#Elbow Graph
#With a parameter like the number of trees in a random forest, increasing the number of trees will never hurt performance. Increasing the number trees will increase performance until a point where it levels out. 
#The more trees, however, the more complicated the algorithm. A more complicated algorithm is more resource intensive to use. Generally it is worth adding complexity to the model if it improves performance but we do not want to unnecessarily add complexity.
#We can use what is called an Elbow Graph to find the sweet spot. Elbow Graph is a model that optimizes performance without adding unnecessary complexity.
n_estimators = list(range(1, 101))
param_grid = {
    'n_estimators': n_estimators,
}
rf = RandomForestClassifier()
gs = GridSearchCV(rf, param_grid, cv=5)
gs.fit(X, y)

#The values are located in the cv_results_ attribute. This is a dictionary with a lot of data, however, we will only need one of the keys: mean_test_score. Let’s pull out these values and store them as a variable.
scores = gs.cv_results_['mean_test_score']
# [0.91564148, 0.90685413, ...]

import matplotlib.pyplot as plt

scores = gs.cv_results_['mean_test_score']
plt.plot(n_estimators, scores)
plt.xlabel("n_estimators")
plt.ylabel("accuracy")
plt.xlim(0, 100)
plt.ylim(0.9, 1)
plt.show()

#Now we can build our random forest model with the optimal number of trees.
rf = RandomForestClassifier(n_estimators=10)
rf.fit(X, y)


#Feature Importances
#There are 30 features in the cancer dataset. Does every feature contribute equally to building a model? If not, which subset of features should we use? This is a matter of feature selection
#Random forests provide a straightforward method for feature selection: mean decrease impurity. Recall that a random forest consists of many decision trees, and that for each tree, the node is chosen to split the dataset based on maximum decrease in impurity, typically either Gini impurity or entropy in classification.
#Thus for a tree, it can be computed how much impurity each feature decreases in a tree. And then for a forest, the impurity decrease from each feature can be averaged. Consider this measure a metric of importance of each feature, we then can rank and select the features according to feature importance.
#Scikit-learn provides a feature_importances_ variable with the model, which shows the relative importance of each feature.
rf = RandomForestClassifier(n_estimators=10, random_state=111)
fr.fit(X_train, y_train)
ft_imp = pd.Series(rf.feature_importances_, index=cancer_data.feature_names).sort_values(ascending=False)
ft_imp.head(10)

#From the output, we can see that among all features, worst radius is most important (0.31), followed by mean concave points and worst concave points.
#In regression, we calculate the feature importance using variance instead.

#New Model on Selected Features
#Why should we perform feature selection? Top reasons are: it enables us to train a model faster; it reduces the complexity of a model thus makes it easier to interpret. And if the right subset is chosen, it can improve the accuracy of a model. Choosing the right subset often relies on domain knowledge, some art, and a bit of luck.
X_worst = df[worst_cols]
X_train, X_test, y_train, y_test = train_test_split(X_worst, y, random_state=101)

#There is no best feature selection method, at least not universally. Instead, we must discover what works best for the specific problem and leverage the domain expertise to build a good model.
#Scikit-learn provides an easy way to discover the feature importances.

#Performance
#Probably the biggest advantage of Random Forests is that they generally perform well without any tuning. They will also perform decently well on almost every dataset.
#A linear model, for example, cannot perform well on a dataset that cannot be split with a line. It is not possible to split the following dataset with a line without manipulating the features. However, a random forest will perform just fine on this dataset.
#We can see this by looking at the code to generate the fake dataset above and comparing a Logistic Regression model with a Random Forest model. The function make_circles makes a classification dataset with concentric circles.
#We use kfold cross validation to compare the accuracy scores and see that the Logistic Regression model performs worse than random guessing but the Random Forest model performs quite well
#When looking to get a benchmark for a new classification problem, it is common practice to start by building a Logistic Regression model and a Random Forest model as these two models both have potential to perform well without any tuning. This will give you values for your metrics to try to beat. Oftentimes it is almost impossible to do better than these benchmarks.

#Interpretability
#Random Forests, despite being made up of Decision Trees, are not easy to interpret. A random forest has several decision trees, each of which is not a very good model, but when averaged, create an excellent model. Thus Random Forests are not a good choice when looking for interpretability.
#In most cases, interpretability is not important.

#Computation
#Random Forests can be a little slow to build, especially if you have a lot of trees in the random forest. Building a random forest involves building 10-100 (usually) decision trees. Each of the decision trees is faster to build than a standard decision tree because of how we do not compare every feature at every split, however given the quantity of decision trees it is often slow to build.
#Similarly, predicting with a Random Forest will be slower than a Decision Tree since we have to do a prediction with each of the 10-100 decision trees in order to get our final prediction
#Random Forests are not the fastest model, but generally this is not a problem since the computational power of computers is a lot.


#Machine Learning - A Forest of Trees
#Build a Random Forest model.
#Task
#You will be given a feature matrix X and target array y. Your task is to split the data into training and test sets, build a Random Forest model with the training set, and make predictions for the test set. Give the random forest 5 trees.
#
#You will be given an integer to be used as the random state. Make sure to use it in both the train test split and the Random Forest model.
#
#Input Format
#First line: integer (random state to use)
#Second line: integer (number of datapoints)
#Next n lines: Values of the row in the feature matrix, separated by spaces
#Last line: Target values separated by spaces
#
#Output Format
#Numpy array of 1's and 0's
#
#Sample Input
#1
#10
#-1.53 -2.86
#-4.42 0.71
#-1.55 1.04
#-0.6 -2.01
#-3.43 1.5
#1.45 -1.15
#-1.6 -1.52
#0.79 0.55
#1.37 -0.23
#1.23 1.72
#0 1 1 0 1 0 0 1 0 1
#
#Sample Output
#[1 0 0]

x_train, x_test, y_train, y_test = train_test_split(X,y,random_state=random_state)

#print(x_train,x_test,y_train,y_test,sep='\n \n')
model = RandomForestClassifier(n_estimators=5,random_state=random_state)

model.fit(x_train,y_train)

pred =model.predict(x_test)
print(pred)


#Neural Network Use Cases
#Neural Networks are incredibly popular and powerful machine learning models. They often perform well in cases where we have a lot of features as they automatically do feature engineering without requiring domain knowledge to restructure the features.
#In this module we will be using image data. Since each pixel in the image is a feature, we can have a really large feature set. They are all commonly used in text data as it has a large feature set as well. Voice recognition is another example where neural networks often shine
#Neural networks often work well without you needing to use domain knowledge to do any feature engineering.

#Biological Neural Network
#A more accurate term for Neural Networks is Artificial Neural Networks (ANN). They were inspired by how biological neural networks work in human brains.
#A brain’s neural network is made up of around 86 billion neurons. The neurons are connected by what are called synapses. There are about 100 trillion synapses in the human brain. The neurons send signals to each other through the synapses

#What's a Neuron?
#An artificial neuron (often called a node) is modeled after a biological neuron. It is a simple object that can take input, do some calculations with the input, and produce an output.
#Each neuron is only capable of a small computation, but when working together they become capable of solving large and complicated problems

#Neuron Computations - wx + wx + b
#Inside the neuron, to do the computation to produce the output, we first put the inputs into the following equation (just like in logistic regression)
#In neural networks, we refer to w1 and w2 as the weights, and b as the bias.
#We plug this value into what is called an activation function. The above equation can have a result of any real number. The activation function condenses it into a fixed range (often between 0 and 1).
#A commonly used activation function is the sigmoid function, the same function we used in logistic regression.
#The weights, w1 and w2, and the bias, b, control what the neuron does. We call these values (w1, w2, b) the parameters

#Activation Functions
#There are three commonly used activation functions: sigmoid (from the previous part), tanh, and ReLU.
#Tanh has a similar form to sigmoid, though ranges from -1 to 1 instead of 0 to 1. Tanh is the hyperbolic tan function
#ReLU stands for Rectified Linear Unit.
#A neuron by itself does not have much power, but when we build a network of neurons, we can see how powerful they are together.

#Multi-Layer Perceptron
#To create a neural network we combine neurons together so that the outputs of some neurons are inputs of other neurons. We will be working with feed forward neural networks which means that the neurons only send signals in one direction. In particular, we will be working with what is called a Multi-Layer Perceptron (MLP)
#A multi-layer perceptron will always have one input layer, with a neuron (or node) for each input. In the neural network above, there are two inputs and thus two input nodes. It will have one output layer, with a node for each output.
#Above there is 1 output node for a single output value. It can have any number of hidden layers and each hidden layer can have any number of nodes. Above there is one hidden layer with 5 nodes.
#The nodes in the input layer take a single input value and pass it forward. The nodes in the hidden layers as well as the output layer can take multiple inputs but they always produce a single output. Sometimes the nodes need to pass their output to multiple nodes.
#A single-layer perceptron is a neural network without any hidden layers. These are rarely used. Most neural networks are multi-layer perceptrons, generally with one or two hidden layers.

#Example Neural Network
#We have a neural network with two inputs, a single hidden layer with two nodes and one output. The weights and bias are given in the nodes below. All the nodes use the sigmoid activation function.
#To change how the neural network performs, we can change the weights and bias values.


#More Than 2 Target Values
#A nice benefit of an MLP classifier is that it easily extends to problems that have more than 2 target values. In the previous modules, we have dealt with predicting 0 or 1 (true or false, survived or not, cancerous or not, ). In some cases, we will be choosing among 3 or more possible outputs. A neural network does this naturally. We just need to add more nodes to the output layer.
#We can use any classifier for a multi-class problem, but neural networks generalize naturally.

#Loss
#In order to train a neural network, we need to define a loss function. This is a measure of how far off our neural network is from being perfect. When we train the neural network, we are optimizing a loss function.
#We will use cross entropy as our loss function. This is the same as the likelihood we used in logistic regression but is called by a different name in this context
#Just like we did with the likelihood function in logistic regression, we use the loss function to find the best possible model.

#Backpropagation
#A neural network has a lot of parameters that we can control. There are several coefficients for each node and there can be a lot of nodes! The process for updating these values to converge on the best possible model is quite complicated. The neural network works backwards from the output node iteratively updating the coefficients of the nodes. This process of moving backwards through the neural network is called backpropagation or backprop.
#We won't go through all the details here as it involves calculating partial derivatives, but the idea is that we initialize all the coefficient values and iteratively change the values so that at every iteration we see improvement in the loss function. Eventually we cannot improve the loss function anymore and then we have found our optimal model.
#Before we create a neural network we fix the number of nodes and number of layers. Then we use backprop to iteratively update all the coefficient values until we converge on an optimal neural network.

#Creating Artificial Dataset
#Sometimes in order to test models, it is helpful to create an artificial dataset. We can create a dataset of the size and complexity needed. Thus we can make a dataset that is easier to work with than a real life dataset. This can help us understand how models work before we apply them to messy real world data.
#We will use the make_classification function in scikit-learn. It generates a feature matrix X and target array y. We will give it these parameters:
#
#• n_samples: number of datapoints
#• n_features: number of features
#• n_informative: number of informative features
#• n_redundant: number of redundant features
#• random_state: random state to guarantee same result every time

#Here is the code to generate a dataset.
from sklearn.datasets import make_classification
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=3)
#Here’s the code to plot the data so we can look at it visually.
from matplotlib import pyplot as plt
plt.scatter(X[y==0][:, 0], X[y==0][:, 1], s=100, edgecolors='k')
plt.scatter(X[y==1][:, 0], X[y==1][:, 1], s=100, edgecolors='k', marker='^')
plt.show()
#Scikit-learn has a couple other functions besides make_classification for making classification datasets with different properties. Look at make_circles and make_moons if you want to play around with more artificial datasets.

#MLPClassifier
#Scikit-learn has an MLPClassifier class which is a multi-layer perceptron for classification. We can import the class from scikit-learn, create an MLPClassifier object and use the fit method to train.
mlp = MLPClassifier()
mlp.fit(X_train, y_train)
#Output: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet. % self.max_iter, ConvergenceWarning)
#You will notice that we get a ConvergenceWarning. This means that the neural network needs more iterations to converge on the optimal coefficients.
mlp = MLPClassifier(max_iter=1000)
#Neural networks are incredibly complicated, but scikit-learn makes them very approachable to use!

#Parameters for MLPClassifier
#There are a couple of parameters that you may find yourself needing to change in the MLPClassifier.
#You can configure the number of hidden layers and how many nodes in each layer. The default MLPClassifier will have a single hidden layer of 100 nodes. This often works really well, but we can experiment with different values. This will create an MLPCLassifier with two hidden layers, one of 100 nodes and one of 50 nodes
mlp = MLPClassifier(max_iter=1000, hidden_layer_sizes=(100, 50))
#We saw max_iter in the previous part. This is the number of iterations. In general, the more data you have, the fewer iterations you need to converge. If the value is too large, it will take too long to run the code. If the value is too small, the neural network will not converge on the optimal solution.
#We also sometimes need to change alpha, which is the step size. This is how much the neural network changes the coefficients at each iteration. If the value is too small, you may never converge on the optimal solution. If the value is too large, you may miss the optimal solution. Initially you can leave this at the default. The default value of alpha is 0.0001. Note that decreasing alpha often requires an increase in max_iter.
#Sometimes you will want to change the solver. This is what algorithm is used to find the optimal solution. All the solvers will work, but you may find for your dataset that a different solver finds the optimal solution faster. The options for solver are 'lbfgs', 'sgd' and 'adam'.
#Run this code in the playground and try changing the parameters for the MLPClassifier. The code uses a random_state to ensure that every time you run the code with the same parameters you will get the same output.

from sklearn.model_selection import train_test_split
from sklearn.neutral_network import MLPClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)
mlp = MLPClassifier(max_iter=1000, hidden_layer_sizes=(100,50), alpha=0.0001, solver='adam', random_state=3)
mlp.fit(X_train, y_train)
print("accuracy:", mlp.score(X_test, y_test))


#The MNIST Dataset
#NIST is the National Institute of Standards and Technology and the M stands for Modified.
#This is a database of images of handwritten digits. We will build a classifier to determine which digit is in the image.
#In scikit-learn we can load the dataset using the load_digits function. To simplify the problem, we will initially only be working with two digits (0 and 1), so we use the n_class parameter to limit the number of target values to 2.
from sklearn.datasets import load_digits
X, y = load_digits(n_class=2, return_X_y=True)
print(X.shape, y.shape)
print(X[0])
print(y[0])
print(X[0].reshape(8, 8))

#Drawing the Digits
#We use the matplotlib function matshow to draw the image. The cmap parameter is used to indicate that the image should be in a grayscale rather than colored.
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

X, y = load_digits(n_class=2, return_X_y=True)
plt.matshow(X[0].reshape(8, 8), cmap=plt.cm.gray)
plt.xticks(())  # remove x tick marks
plt.yticks(())  # remove y tick marks
plt.show()

#MLP for MNIST Dataset
#We will do a train/test split and train an MLPClassifier on the training set.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
mlp = MLPClassifier()
mlp.fit(X_train, y_train)
# We use matplotlib to draw the images and then show the model’s prediction.
x = X_test[0]
plt.matshow(x.reshape(8, 8), cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.show()
print(mlp.predict([x]))
# 0

#Similarly, let’s look at the second datapoint.
x = X_test[1]
plt.matshow(x.reshape(8, 8), cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.show()
print(mlp.predict([x]))
# 1

#Classifying all 10 Digits
#Since neural networks easily generalize to handle multiple outputs, we can just use the same code to build a classifier to distinguish between all ten digits.
y_pred = mlp.predict(X_test)
incorrect = X_test[y_pred != y_test]
incorrect_true = y_test[y_pred != y_test]
incorrect_pred = y_pred[y_pred != y_test]
#Let’s look at the first image that we got wrong and what our prediction was.
j = 0
plt.matshow(incorrect[j].reshape(8, 8), cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.show()
print("true value:", incorrect_true[j])
print("predicted value:", incorrect_pred[j])

#Open ML
from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

print(X.shape, y.shape)
print(np.min(X), np.max(X))
print(y[0:5])

#We will be modifying some of the default parameters in the MLPClassifier to build the model. Since our goal will be to visualize the weights of the hidden layer, we will use only 6 nodes in the hidden layer so that we can look at all of them. We will use 'sgd' (stochastic gradient descent) as our solver which requires us to decrease alpha (the learning rate)
mlp=MLPClassifier(
  hidden_layer_sizes=(6,), 
  max_iter=200, alpha=1e-4,
  solver='sgd', random_state=2)

mlp.fit(X5, y5)
#Since this dataset is quite large, you will want to work with it on your computer rather than the code playground.

#MLPClassifier Coefficients
print(mlp.coefs_)
print(len(mlp.coefs_))
#The two elements in the list correspond to the two layers: the hidden layer and the output layer. We have an array of coefficients for each of these layers.
print(mlp.coefs_[0].shape)
#Output:(784, 6)

#Visualizing the Hidden Layer
fig, axes = plt.subplots(2, 3, figsize=(5, 4))
for i, ax in enumerate(axes.ravel()):
    coef = mlp.coefs_[0][:, i]
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(i + 1)
plt.show()
#If you change the random state in the MLPClassifier, you will likely get different results. There are many equivalently optimal neural networks that work differently.

#Interpretability
#While we can visualize the nodes in the hidden layer to understand on a high level what the neural network is doing, it is impossible to answer the question "Why did datapoint x get prediction y?"
#Since there are so many nodes, each with their own coefficients, it is not feasible to get a simple explanation of what the neural network is doing. This makes it a difficult model to interpret and use in certain business use cases.
#Neural Networks are not a good option for interpretability.


#Computation
#Neural networks can take a decent amount of time to train. Each node has its own coefficients and to train they are iteratively updated, so this can be time consuming. However, they are parallelizable, so it is possible to throw computer power at them to make them train faster.
#Once they are built, neural networks are not slow to make predictions, however, they are not as fast as some of the other models.

#Performance
#The main draw to neural networks is their performance. On many problems, their performance simply cannot be beat by other models. They can take some tuning of parameters to find the optimal performance, but they benefit from needing minimal feature engineering prior to building the model
#A lot of simpler problems, you can achieve equivalent performance with a simpler model like logistic regression, but with large unstructured datasets, neural networks outperform other models
#The key advantage of neural networks is their performance capabilities.


#Machine Learning - The Sigmoid Function
#Calculate Node Output.
#Task
#You are given the values for w1, w2, b, x1 and x2 and you must compute the output for the node. Use the sigmoid as the activation function.
#
#Input Format
#w1, w2, b, x1 and x2 on one line separated by spaces
#
#Output Format
#Float rounded to 4 decimal places
#
#Sample Input
#0 1 2 1 2
#
#Sample Output
#0.9820

w1, w2, b, x1, x2 = [float(x) for x in input().split()]
import math
output = w1*x1 + w2*x2 + b
output = round(1/(1 + math.exp(output*-1)), 4)
print(output)

