#Machine learning, a subset of data science, is the scientific study of computational algorithms and statistical models to perform specific tasks through patterns and inference instead of explicit instructions
#Machine learning can be described as a set of tools to build models on data. Data scientists explore data, select and build models (machine), tune parameters such that a model fits observations (learning), then use the model to predict and understand aspects of new unseen data.
#Machine learning is a set of tools used to build models on data. Building models to understand data and make predictions is an important part of a data scientists' job.

#Supervised and Unsupervised Learning
#Supervised learning is when we have a known target (also called label) based on past data (for example, predicting what price a house will sell for)
#unsupervised learning is when there isn’t a known past answer (for example, determining the topics discussed in restaurant reviews).
#In this module we will explore Linear Regression, a supervised machine learning algorithm
#In the modules to come we will also explore another supervised machine learning algorithm, classification, 
#as well as an unsupervised machine learning algorithm, clustering.

#Scikit-learn
from sklearn.datasets import load_boston  #boston data
#Scikit-learn, one of the best known machine learning libraries in python for machine learning, implements a large number of commonly used algorithms
#Regardless of the type of algorithm, the syntax follows the same workflow: import > instantiate > fit > predict.
#Once the basic use and syntax of Scikit-learn is understood for one model, switching to a new algorithm is straightforward

#Linear Regression
#Linear regression fits a straight line to data, mathematically:
y = b + m*x
#where b is the intercept and m is the slope, x is a feature or an input, whereas y is label or an output. 
#Our job is to find m and b such that the errors are minimized
#Linear regression models are popular because they can perform a fit quickly, and are easily interpreted. Predicting a continuous value with linear regression is a good starting point
#So linear regression essentially is finding the line where it minimizes the sum of the squared residuals

from sklearn.datasets import load_boston
boston_datadet = load_boston()

#For easier manipulations later, we create a pandas DataFrame from the numpy ndarrays stored in boston_dataset.data as follows
import pandas as pd
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
#We then add the target into the DataFrame:
boston['MEDV'] = boston_dataset.target

#Head
#It is useful for quickly testing if the DataFrame has the right type of data in it. To see the first few rows of a DataFrame, use .head(n), where you can specify n for the number of rows to be selected
#Selecting the columns we want to see in the top5
boston.[['CHAS', 'RM', 'AGE', 'RAD', 'MEDV']]head()


#Summary Statistics
boston.describe().round(2)  #round to the second decimal place for better display
#If the DataFrame contains more than just numeric values, by default, describe() outputs the descriptive statistics for the numeric columns. 
#To show the summary statistics of all column, specify include = 'all' in the method


#Visualization
#It’s a good practice to visualize and inspect the distribution column by column.
boston.hist(column='CHAS')
plt.show()

boston.hist(column='RM', bins=20)
plt.show()
#Informative data visualization not only reveals insights, but they are invaluable to communicate findings to stakeholders.


#Correlation Matrix
#To understand the relationship among features (columns), a correlation matrix is very useful in the exploratory data analysis. 
#Correlation measures linear relationships between variables. We can construct a correlation matrix to show correlation coefficients between variables. 
#It is symmetric where each element is a correlation coefficient ranging from -1 and 1. A value near 1 (resp. -1) indicates a strong positive (resp. negative) correlation between variables. 
#We can create a correlation matrix using the "corr" function
corr_matrix = boston.corr().round(2)
#Understanding data using exploratory data analysis is an essential step before building a model. 
#From sample size and distribution to the correlations between features and target, we gather more understanding at each step aiding in feature and algorithm selection

#Data Preparation - Feature Selection
boston.plot(kind='scatter', x='RM', y="MEDV", figsize=(8,6));
#Recall that the single bracket outputs a Pandas Series, while a double bracket outputs a Pandas DataFrame, and the model expects the feature matrix X to be a 2darray.
#Feature selection is used for several reasons, including simplification of models to make them easier to interpret, shorter training time, reducing overfitting, etc.

#Instantiating the Model
#In scikit-learn, every class of model is represented by a class in python. A class of model is not the same as an instance of a model. Recall that instance is an individual object of a certain class. 
#Thus, we first import the linear regression class, then instantiate the model, that is to create an instance of the class LinearRegression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
#Now the model is instantiated, but not yet applied to the data.
#Scikit-learn makes the distinction between choice of model and application of model to data very clear.

#Train-Test Split
#Next we split the data into training and testing sets. Why? To assess the performance of the model on newly unseen data. 
#We train the model using a training set, and save the testing set for evaluation.
#A good rule of thumb is to split data 70-30, that is, 70% of data is used for training and 30% for testing. 
#We use train_test_split function inside scikit-learn’s module model_selection to split the data into two random subsets. 
#Set random_state so that the results are reproducible
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=1)
#We check the dimensions to ensure the same number of rows.
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
#To get an objective assessment on model’s predictive power, it’s important to keep the testing data unseen to the built model.

#Fitting the Model
#In short, fitting is equal to training. It fits the model to the training data and finds the coefficients specified in the linear regression model, i.e., intercept and slope. After it is trained, the model can be used to make predictions.
#Now let us apply the model to data. Remember, we save the testing data to report the model performance and only use the training set to build the model. The syntax is:
model.fit(X_train, Y_train)
#The fit() command triggers the computations and the results are stored in the model object.
#Fitting is how well the machine learning model measures against the data upon which it was trained.

#Parameter Estimates
#The linear regression model has been fitted, what it means is that both parameters, the intercept and the slope, have been learned. What are they? 
#In Scikit-learn, by convention all model parameters have trailing underscores, for example to access the estimated intercept from the model, rounded to the 2nd decimal place for better display:
model.intercept_.round(2)  #output: -30.57
#Similarly, the estimated coefficient of feature RM is:
model.coef_.round(2)  #output: [8.46]
#The two parameters represent the intercept and slope of the line fit to the data. Our fitted model is MEDV = -30.57 + 8.46 * RM. For one unit increase in RM, the median home price would go up by $8460.
#The full code to fit the model is:
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X = boston[['RM']]
Y = boston[['MEDV']]
X_train, X_test, Y_train, Y_test = train_test_split(x, Y,
test_size = 0.3,
random_state=1)
model = LinearRegression()
model.fit(X_train, Y_train)
model.intercept_.round(2)
model.coef_.round(2)  #to access the slope of the fitted line in object model
#You did it! You just built the first linear regression model in scikit-learn: from import the class to instantiate the model, to fit the model to the data, and done!

#Prediction
#Once the model is trained, supervised machine learning will evaluate test data based on previous predictions for the unseen data. We can make a prediction using the predict() method.
#Eg. When the average number of rooms per dwelling is 6.5, the model predicts a home value of $24,426.06.
new_RM = np.array([6.5]).reshape(-1,1) #make sure it is 2 dimensional
model.predict(new_RM)
#output: [24.42606323]
#Note that the input has to be 2-dimensional, either a 2darray or DataFrame will work in this case
#This value is the same as we plug in the line b + m*x where b is the estimated intercept from the model, and m is the estimated slope.
model.intercept_+model.coef_*6.5
#In addition, we can feed the testing set and get predictions for all homes.
y_test_predicted = model.predict(X_test)
y_test_predicted.shape
type(y_test_predicted)
#The output is a 1darray, same shape as the Y_test.
Y_test.shape
#The predict() method estimates the median home value by computing model.intercept_ + model.coef_*RM.

plt.scatter(X_test, Y_test, label='testing data');
plt.plot(X_test, Y_predicted, label='predition', linewidth=3)
plt.xlabel('RM'); plt.ylabel('MEDV')
plt.legend(loc='upper left')
plt.show()


#Residuals
#Some points sit on the line, but some are away from it. We can measure the distance between a point to the line along the vertical line, and this distance is referred to as residual or error. 
#A residual is the difference between the observed value of the target and the predicted value. The closer the residual is to 0, the better job our model is doing
#We can calculate a residual and represent it in a scatter plot.
residuals = Y_test - y_test_predicted
plt.scatter(X_test, residuals)  #plot the residuals
plt.hlines(y = 0, xmin = X_test.min(), xmax=X_test.max(), linestyle='--')   #plot a horizontal line at y=0
plt.xlim((4,9))  #set xlim
plt.xlabel('RM'); plt.ylabel('residuals')
plt.show()
#Residuals are scattered around the horizontal line, y = 0, with no particular pattern. This seemingly random distribution is a sign that the model is working. 
#Ideally the residuals should be symmetrically and randomly spaced around the horizontal axis; if the residual plot shows some pattern, linear or nonlinear, that’s an indication that our model has room for improvement.
#Residual plots can reveal bias from the model and statistical measures indicate goodness-of-fit.


#Mean Squared Error
#Previously, we learned that when each residual is near 0 it suggests a good fit. For example, the first five residuals in our model:
residuals[:5]
#Those are individual data points, how about the model performance for all data points? We need a way to aggregate the residuals and just report one number as the metric. 
#It is natural to take the average of all residuals:
residuals.mean() #output: -0.236450
#-0.24 is quite close to 0, but there’s a problem: residuals can be positive or negative so taking the average cancels them out. That’s not an accurate metric. 
#To solve this, we take a square of each residual, then take the mean of squares. This is called mean squared error (MSE):
(residual**2).mean()
#We can also use the mean_squared_error() method under scikit-learn metrics module to output the same result:
from sklearn.metrics import mean_squared_error
mean_squared_error(Y_test, y_test_predicted)

#In general, the smaller the MSE, the better, yet there is no absolute good or bad threshold. We can define it based on the dependent variable, i.e., MEDV in the test set. 
#Y_test ranges from 6.3 to 50 with a variance 92.26. Compared to the total variance, a MSE of 36.52 is not bad
#To make the scale of errors to be the same as the scale of targets, root mean squared error (RMSE) is often used. It is the square root of MSE.


#R-squared
#Another common metric to evaluate the model performance is called R-squared; one can calculate it via model.score():
model.score(X_test, Y_test)  
#It is the proportion of total variation explained by the model. Here, around 60% of variability in the testing data is explained by our model.
#The total variation is calculated as the sum of squares of the difference between the response and the mean of response, in the example of testing data:
((Y_test-Y_test.mean())**2).sum()  #a
#Whereas the variation that the model fails to capture is computed as the sum of squares of residuals:
(residuals**2).sum() #b
#Then the proportion of total variation from the data is:
1-b/a
#A perfect model explains all the variation in the data. Note R-squared is between 0 and 100%: 0% indicates that the model explains none of the variability of the response data around its mean while 100% indicates that the model explains all of it
#Evaluating R-squared values in conjunction with residual plots quantifies model performance

#We can add the feature and build a multivariate linear regression model where the home price depends on both RM and LSTAT linearly:
#MEDV = b0 + b1 * RM + b2 * LSTAT
#To find intercept b0, and coefficients b1 and b2, all steps are the same except for the data preparation part, we are now dealing with two features:
X2 = boston[['RM', 'LSTAT']]  #data prep
Y = boston['MEDV']    #data prep
X2_train, X2_test, Y_train, Y_test = train_test_split(X2, Y
test_size = 0.3, random_state=1) #train test split and same random_state to ensure the same splits
model2 = LinearRegression()
model2.fit(X2_train, Y_train)
#We can access the parameters after model2 is fitted
model2.intercept_
model2.coef_
#Note the coefficients are stored in a 1darray of shape (2,). The second model then is
#MEDV = 5.32 + 4.13 * RM + (-0.68) * LSTAT.
#Allowing for predictions:
y_test_predicted2 = model2.predict(X2_test)
#The extension from univariate to multivariate linear regression is straightforward in scikit-learn. The model instantiation, fitting, and predictions are identical, the only difference being the data preparation

#Comparing Models
#An easy metric for linear regression is the mean squared error (MSE) on the testing data. Better models have lower MSEs. Recall the MSE of the first model on testing data is:
mean_squared_error(Y_test, y_test_predicted).round(2)
#The MSE of the second model is:
mean_squared_error(Y_test, y_test_predicted2).round(2)
#The second model has a lower MSE, specifically a 21% reduction (36.52-28.93)/36.52 = 21%); thus it does a better job predicting the median home values than the univariate model.
#In general, the more features the model includes the lower the MSE would be. Yet be careful about including too many features. Some features could be random noise, thus hurt the interpretability of the model.


#Ordinary least squares for linear regression.
#Ordinary least squares (OLS) is a method to estimate the parameters β in a simple linear regression, Xβ = y, where X is the feature matrix and y is the dependent variable (or target), by minimizing the sum of the squares of the differences between the observed dependent variable in the given dataset and those predicted by the linear function. Mathematically, the solution is given by the formula in the image, where the superscript T means the transpose of a matrix, and the superscript -1 means it is an inverse of a matrix.
#Task
#Given a 2D array feature matrix X and a vector y, return the coefficient vector; see the formula.
#Input Format
#First line: two integers separated by spaces, the first indicates the rows of the feature matrix X (n) and the second indicates the columns of X (p)
#Next n lines: values of the row in the feature matrix
#Last line: p values of target y
n, p = [int(x) for x in input().split()]
X = []
for i in range(n):
    X.append([float(x) for x in input().split()])

y = [float(x) for x in input().split()]

import numpy as np
coef_mat = np.linalg.lstsq(X ,y, rcond= None)[0].round(2)
print(coef_mat)


#Discrete Values
#Discrete data are only able to have certain values, while continuous data can take on any value.
#Examples of classification problems involving discrete data values are:
#• to predict whether a breast cancer is benign or malignant given a set of features
#• to classify an image as containing cats or dogs or horses
#• to predict whether an email is spam or not from a given email address
#In each of the examples, the labels come in categorical form and represent a finite number of classes.
#Discrete data values can be numeric, like the number of students in a class, or it can be categorical, like red, blue or yellow.

#Binary and Multi-class Classification
#There are two types of classification: binary and multi-class. If there are two classes to predict, that is a binary classification problem, for example, a benign or malignant tumor.
#When there are more than two classes, the task is a multi-classification problem. For example, classifying the species of iris, which can be versicolor, virqinica, or setosa, based on their sepal and petal characteristics.
#Common algorithms for classification include logistic regression, k nearest neighbors, decision trees, naive bayes, support vector machines, neural networks, etc. Here we will learn how to use k nearest neighbors to classify iris species.
#Supervised learning problems are grouped into regression and classification problems. Both problems have as a goal the construction of a mapping function from input variables (X) to an output variable (y). The difference is that the output variable is continuous in regression and categorical for classification

#Iris Dataset
import pandas as pd
iris = pd.read_csv('./data/iris.csv')
iris.shape() #Now inspect the dimensions and first few rows:
iris.head()  #We use the .head() function to view the first 5 rows:

#The column id is the row index, not really informative, so we can drop it from the dataset using drop() function
iris.drop('id', axis=1, inplace=True)
iris.head()
#When we are learning about machine learning algorithms, using simple, well-behaved data such as iris flower dataset, decreases the learning curve and makes it easier to understand and debug.

#eg. complete the code to remove the column labeled email and check to see the first 5 rows
import pandas as pd
user = pd.read_csv('./data/user.csv')
user.drop('Email', axis=1, inplace=True)
user.head()

#The method value_counts() is a great utility for quickly understanding the distribution of the data. When used on the categorical data, it counts the number of unique values in the column of interest.

#To see the interactions between attributes we use scatter plots
#Therefore, we define a color code for each species to differentiate species visually:
inv_name_dict = {'iris-sertosa': 0,
'iris-versicolor': 1,
'iris-virginica': 2}  #build a dict mapping species to an integer code
colors = [inv_name_dict[item] for item in iris['species']]  #build integer color code 0/1/2
scatter = plt.scatter(iris['sepal_len'], iris['sepal_wd'], c=colors)  #scatter plot
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.legend(handles=scatter.legend_elements()[0], labels = inv_name_dict.keys())  #add legend
plt.show()
#To see scatter plots of all pairs of features, use pandas.plotting.scatter_matrix(). Besides the histograms of individual variables along the diagonal, it will show the scatter plots of all pairs of attributes to help spot structured relationships between features

#K nearest neighbors
#K nearest neighbors (knn) is a supervised machine learning model that takes a data point, looks at its 'k' closest labeled data points, and assigns the label by a majority vote
#Here we see that changing k could affect the output of the model. In knn, k is a hyperparameter.
#A hyperparameter in machine learning is a parameter whose value is set before the learning process begins. We will learn how to tune the hyperparameter later.
from sklearn.neighbors import KNeighborsClassifier
#K nearest neighbors can also be used for regression problems. The difference lies in prediction. Instead of a majority vote, knn for regression makes a prediction using the mean labels of the k closest data points.

#Data Preparation
X = iris[['petal_len', 'petal_wd']]
y = iris['species']

#we set aside some portion of the data as a test set to mimic the unknown data the model will be presented with in the future. As done in the previous module, we use train_test_split in sklearn.model_selection.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1, stratify=y)

#We use a 70-30 split, i.e., 70% of the data is for training and 30% for testing. Note that we specified the split was stratified by label (y). This is done to ensure that the distribution of labels remains similar in both train and test sets:
y_train.value_counts()
y_test.value_counts()

#Modeling
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
#Use the data X_train and y_train to train the model:
knn.fit(X_train, y_train)
#It outputs the trained model. We use most the default values for the parameters, e.g., metric = 'minkowski' and p = 2 together defines that the distance is euclidean distance


#Label Prediction
#To make a prediction in scikit learn, we can call the method predict().
#We are trying to predict the species of iris using given features in feature matrix X.
#Let’s make the predictions on the test data set and save the output in pred for later review:
pred = knn.predict(X_test)
#Let’s review the first five predictions:
pred[:5]
#predict() returns an array of predicted class labels for the predictor data.


#Probability Prediction
#Of all classification algorithms implemented in scikit learn, there is an additional method 'predict_prob'.
#Instead of splitting the label, it outputs the probability for the target in array form. Let’s take a look at what the predicted probabilities are for the 11th and 12th flowers:
y_pred_prob = knn.predict_proba(X_test)
y_pred_prob[10:12]
#To see the corresponding predictions:
y_pred[10:12]
#In classification tasks, soft prediction returns the predicted probabilities of data points belonging to each of the classes while hard prediction outputs the labels only.

#Accuracy
#In classification the most straightforward metric is accuracy. 
#It calculates the proportion of data points whose predicted labels exactly match the observed labels.
(y_pred==y_test.values).sum()
y_test.size

#The classifier made one mistake. Thus, the accuracy is 44/45:
(y_pred==y_test.values).sum()/y_test.size

#Same as:
knn.score(X_test, y_test)
#Under the module sklearn.metrics, function accuracy_score(y_true, y_pred) does the same calculation.

#Confusion Matrix
#Classification accuracy alone can be misleading if there is an unequal number of observations in each class or if there are more than two classes in the dataset
#Calculating a confusion matrix will provide a better idea of what the classification is getting right and what types of errors it is making.
#What is a confusion matrix? It is a summary of the counts of correct and incorrect predictions, broken down by each class.
#In classifying the iris, we can use confusion_matrix() under module sklearn.metrics:
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred, labels=['iris-sentosa', 'iris-versicolor', 'iris-virginica'])
#We can visualize the confusion matrix:
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(knn, X_test, y_test, cmap=plt.cm.Blues);
#Here we specified the labels in order. Each column of the matrix corresponds to a predicted class, and each row corresponds to an actual class. So the row sums up to the total number of instances of the class.
#A confusion matrix is a table that is often used to describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known.

#K-fold Cross Validation
#Previously we made train-test split before fitting the model so that we can report the model performance on the test data. This is a simple kind of cross validation technique, also known as the holdout method
#However, the split is random, as a result, model performance can be sensitive to how the data is split. To overcome this, we introduce k-fold cross validation.
#In k fold cross validation, the data is divided into k subsets. Then the holdout method is repeated k times, such that each time, one of the k subsets is used as the test set and the other k-1 subsets are combined to train the model.
#Then the accuracy is averaged over k trials to provide total effectiveness of the model. In this way, all data points are used; and there are more metrics so we don’t rely on one test data for model performance evaluation.
#The simplest way to use k-fold cross-validation in scikit-learn is to call the cross_val_score function on the model and the dataset:
from sklearn.model_selection import cross_val_score
# create a new KNN model
knn_cv = KNeighborsClassifier(n_neighbors=3)
#Note that now we are fitting a 3nn model.
# train model with 5-fold cv
cv_scores = cross_val_score(knn_cv, X, y, cv=5)
#Each of the holdout set contains 20% of the original data.
print(cv_scores)
cv_scores.mean()  #find average
#We can not rely on one single train-test split, rather we report that the 3nn model has an accuracy of 95.33% based on a 5-fold cross validation.
#As a general rule, 5-fold or 10-fold cross validation is preferred; but there is no formal rule.
#As k gets larger, the difference in size between the training set and the resampling subsets gets smaller. As this difference decreases, the bias of the technique becomes smaller.

#Grid Search
#Finding the optimal k is called tuning the hyperparameter. A handy tool is grid search. 
#In scikit-learn, we use GridSearchCV, which trains our model multiple times on a range of values specified with the param_grid parameter and computes cross validation score, so that we can check which of our values for the tested hyperparameter performed the best.
from sklearn.model_selection import GridSearchCV
# create new a knn model
knn2 = KNeighborsClassifier()
# create a dict of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(2, 10)}
# use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
#fit model to data
knn_gscv.fit(X, y)

#To check the top performing n_neighbors value:
knn_gscv.best_params_   #{'n_neighbors': 4}
#What is the accuracy of the model when k is 4?
knn_gscv.best_score_
#By using grid search to find the optimal hyperparameter for our model, it improves the model accuracy by over 1%.
knn_final = KNeighborsClassifier(n_neighbors=knn_gscv.best_params_['n_neighbors'])
knn_final.fit(x, y)
y_pred = knn_final.predict(X)
knn_final.score(X,y) #output = 0.97333 We can report that our final model, 4nn, has an accuracy of 97.3% in predicting the species of iris!
#The techniques of k-fold cross validation and tuning parameters with grid search is applicable to both classification and regression problems.


#Label Prediction with New Data
#Now we are ready to deploy the model knn_final'. We take some measurements of an iris and record that the length and width of its sepal are 5.84 cm and 3.06 cm, respectively, and the length and width of its petal are 3.76 cm and 1.20 cm, respectively. How do we make a prediction using the built model?
#Use model.predict. Since the model was trained on the length and width of petals, that’s the data we will need to make a prediction. Let’s put the petal length and petal width into a numpy array:
new_data = np.array([3.76, 1.20])
#If we feed it to the model:
knn_final.predict(np.array(new_data))
#value error: expected 2d array, got 1d array instead
#Wait, what just happened? When we trained the model, the data is 2D DataFrame, so the model was expecting a 2D array, which could be numpy array or pandas DataFrame. Now new_data is a 1D array, we need to make it 2D as the error message suggested:
new_data = new_data.reshape(1, -1)
#Now we are ready to make a label prediction:
knn_final.predict(new_data)
#['iris-versicolor']
#Model.predict can also take a 2D list. For example, knn_final.predict([[3.76, 1.2]]) will output the same result as shown in the lesson.


#Probability Prediction with New Data
#Let's collect more data: three plants of iris share the same petal width, 2.25cm, but are different in the length of the petal: 5.03 cm, 3.85 cm, and 1.77 cm, respectively. We store the new data into a 2D array as follows:
new_data = np.array([[3.76, 1.2], [5.25, 1.2], [1.58, 1.2]])
#We learned from the previous part that we can make predictions using knn_final.predict():
knn_final.predict(new_data)
#(['iris-versicolor', 'iris-virginica', 'iris-sentosa']

#Recall that in classifications, it is more common to predict the probability of each data point being assigned to each label:
knn_final.predict_proba(new_data)
#For classification algorithms in scikit learn, function predict_proba takes a new data point and outputs a probability for each class as a value between 0 and 1.

new_data = np.array([[3.76, 1.2], [5.25, 1.2], [1.58, 1.2]])
knn_final.predict_proba(new_data)

#classify iris and output the predicted probabilities on the testing data set
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
new_predictions =knn.predict_proba(x_test)

#Given y_true and y_pred below, fill in the blanks in the output. Note that the labels are to index the matrix.
import numpy as np
from sklearn.metrics import confusion_matrix 
y_true = np.array(['cat', 'dog', 'dog', 'cat', 'fish', 'dog', 'fish'])
y_pred = np.array(['cat', 'cat', 'cat', 'cat', 'fish', 'dog', 'fish'])
confusion_matrix(y_true, y_pred, labels=['cat', 'dog', 'fish'])
#array([[2,0,0], [2,1,0], [0,0,2]])


#Data Science - Binary Disorder
#Confusion matrix of binary classification.
#For binary classifications, a confusion matrix is a two-by-two matrix to visualize the performance of an algorithm. Each row of the matrix represents the instances in a predicted class while each column represents the instances in an actual class.
#
#Task
#Given two lists of 1s and 0s (1 represents the true label, and 0 represents the false false) of the same length, output a 2darrary of counts, each cell is defined as follows
#
#Top left: Predicted true and actually true (True positive)
#Top right: Predicted true but actually false (False positive)
#Bottom left: Predicted false but actually true (False negative)
#Bottom right: Predicted false and actually false (True negative)
#
#Input Format
#First line: a list of 1s and 0s, separated by space. They are the actual binary labels.
#Second line: a list of 1s and 0s, the length is the same as the first line. They represented the predicted labels.
#
#Output Format
#A numpy 2darray of two rows and two columns, the first row contains counts of true positives and false positives and the second row contains counts of false negatives and true negatives.
#
#Sample Input
#1 1 0 0
#1 0 0 0
#
#Sample Output
#[[1., 0.],
#[1., 2.]]

y_true = [int(x) for x in input().split()]
y_pred =  [int(x) for x in input().split()]

from sklearn.metrics import confusion_matrix
import numpy as np

cm = confusion_matrix(y_pred, y_true, labels=[1,0])

print(np.array(cm, dtype='f'))


#Clustering
#Clustering is a type of unsupervised learning that allows us to find groups of similar objects, objects that are more related to each other than to the objects in other groups. This is often used when we don’t have access to the ground truth, in other words, the labels are missing.
#Examples of business use cases include the grouping of documents, music, and movies based on their contents, or finding customer segments based on purchase behavior as a basis for recommendation engines.
#The goal of clustering is to separate the data into groups, or clusters, with more similar traits to each other than to the data in the other clusters.

#Different Types of Clustering Algorithms
#There are more than 100 clustering algorithms known, 12 of them have been implemented in scikit-learn, but few gained popularity.
#In general, there are four types:
#Centroid based models - each cluster is represented by a single mean vector (e.g., k-means),
#Connectivity based models - built based on distance connectivity (e.g., hierarchical clustering)
#Distribution based models - built using statistical distributions (e.g., Gaussian mixtures)
#Density based models - clusters are defined as dense areas (e.g., DBSCAN)
#In this module, we will explore the simple and widely-used clustering algorithm, k-means, to reveal subgroups of wines based on the chemical analysis reports.

#K-means
#One of the most popular clustering algorithms is k-means. Assuming that there are n data points, the algorithm works as follows:
#Step 1:initialization - pick k random points as cluster centers, called centroids
#Step 2:cluster assignment - assign each data point to its nearest centroid based on its distance to each centroid, and that forms k clusters
#Step 3:centroid updating - for each new cluster, calculate its centroid by taking the average of all the points assigned to the cluster
#Step 4:repeat steps 2 and 3 until none of cluster assignments change, or it reaches the maximum number of iterations
#The k-means algorithm has been implemented in module sklearn.cluster, to access it:
from sklearn.cluster import KMeans
#The algorithm has gained great popularity because it is easy to implement and scales well to large datasets. However, it is difficult to predict the number of clusters, it can get stuck in local optimums, and it can perform poorly when the clusters are of varying sizes and density.


#Distance Metric
#How do we calculate the distance in k-means algorithm? One way is the euclidean distance, a straight line between two data points as shown below.
#in numpy we can calculate the distance as follows:
import numpy as np
x1 = np.array([0, 1])
x2 = np.array([2, 0])
print(np.sqrt(((x1-x2)**2).sum()))
print(np.sqrt(5))
#There are other distance metrics, such as Manhattan distance, cosine distance, etc. The choice of the distance metric depends on the data.

#Wine Data
#In this module, we analyze the result of a chemical analysis of wines grown in a particular region in Italy. And the goal is to try to group similar observations together and determine the number of possible clusters. This would help us make predictions and reduce dimensionality. 
#As we will see there are 13 features for each wine, and if we could group all the wines into, say 3 groups, then it is reducing the 13-dimensional space to a 3-dimensional space. More specifically we can represent each of our original data points in terms of how far it is from each of these three cluster centers.
#The analysis reported the quantities of 13 constituents from 178 wines: alcohol, malic acid, ash, alcalinity of ash, magnesium, total phenols, flavanoids, nonflavanoid phenols, proanthocyanins, color intensity, hue, od280/od315 of diluted wines, and proline.
#The data is loaded in a dataframe 'wine'.
wine.shape
wine.columns
#For the ease of display, we show the basic statistics of the first 3 features:
wine.iloc[:, :3].describe()
#Another way to check for column names and the datatype of each column is to use .info()

#Plotting the Data
#The summary statistics provide some of the information, while visualization offers a more direct view showing the distribution and the relationship between features.
#Here we introduce a plotting function to display histograms along the diagonal and the scatter plots for every pair of attributes off the diagonal, 'scatter_matrix', for the ease of display, let’s show just two features:

import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
scatter_matrix(wine.iloc[:, [0,5]])
plt.show()
#No matter whether it is a supervised or unsupervised learning problem, exploratory data analysis (EDA) is essential and strongly recommended before one dives into modeling.


#Pre-processing: Standardization
#After examining all the pairs of scatter plot, we pick two features to better illustrate the algorithm: alcohol and total_phenols, whose scatterplot also suggests three subclusters
X = wine[['alcohol', 'total_phenols']]
#Unlike any supervised learning models, in general, unsupervised machine learning models do not require to split data into training and testing sets since there is no ground truth to validate the model. 
#However, centroid-based algorithms require one pre-processing step because k-means works better on data where each attribute is of similar scales. One way to achieve this is to standardize the data; mathematically:
#z = (x - mean) / std
#where x is the raw data, mean and std are the average and standard deviation of x, and z is the scaled x such that it is centered at 0 and it has a unit standard deviation. StandardScaler under the sklearn.preprocessing makes it easy:
from sklearn.preprocessing import StandardScaler
# instantiate the scaler
scale = StandardScaler()
# compute the mean and std to be used later for scaling
scale.fit(X)
# StandardScaler(copy=True, with_mean=True, with_std=True)

#We can look into the object scale, extract the calculated mean and std:
scale.mean_
scale.scale_
#we can fit to the training data, and transform it
X_scaled = scale.transform(X)


#K-means Modeling
#Just like linear regression and k nearest neighbours, or any machine learning algorithms in scikit-learn, to do the modeling, we follow instantiate / fit / predict workflow. 
#There are other arguments in KMeans, such as method to initialize the centroids, stopping criteria, etc., yet we focus on the number of clusters, n_clusters, and allow other parameters to take the default values. Here we specify 3 clusters:
from sklearn.cluster import KMeans  #instantiate the model
kmeans = KMeans(n_clusters=3)   #fit the model
kmeans.fit(X_scaled)   #make predictiond
y_pred = kmeans.predict(X_scaled)
print(y_pred)

kmeans.cluster_centers_  #To inspect the coordinates of the three centroids:

import matplotlib.pyplot as plt  #plot the scaled data
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=y_pred)  #identify the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="*", s= 250, c =[0,1,2], edgecolors='k')
plt.xlabel('alcohol'); plt.ylabel('total phenols')
plt.title('k-means (k=3)')
plt.show()

#First we need to put the new data into a 2d array:
X_new = np.array([[13, 2.5]])

#Next, we need to standardize the new data:
X_new_scaled = scale.transform(X_new)
print(X_new_scaled)
# [[-0.00076337  0.32829793]]

kmeans.predict(X_new_scaled)#Now we are ready to predict the cluster:
#One major shortcoming of k-means is that the random initial guess for the centroids can result in bad clustering, and k-means++ algorithm addresses this obstacle by specifying a procedure to initialize the centroids before proceeding with the standard k-means algorithm. 
#In scikit-learn, the initialization mechanism is set to k-means++, by default.

#Intuitively, k-means problem partitions n data points into k tight sets such that the data points are closer to each other than to the data points in the other clusters. And the tightness can be measured as the sum of squares of the distance from data point to its nearest centroid, or inertia.
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_scaled)
kmeans.inertia_

import numpy as mp 
#calculate distortion for a range for a number of cluster
inertia = []
for i in np.arange(1, 11)
km = KMeans(n_clusters=i)
km.fit(X_scaled)
inertia.append(km.inertia_)
#plot
plt.plot(np.arange(1, 11), inertia, marker="o")
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()
#For example, k=3 seems to be optimal, as we increase the number of clusters from 3 to 4, the decrease in inertia slows down significantly, compared to that from 2 to 3. This approach is called elbow method (can you see why?). It is a useful graphical tool to estimate the optimal k in k-means.
#One single inertia alone is not suitable to determine the optimal k because the larger k is, the lower the inertia will be.

#Modeling With More Features
#However, can we use more features
X = wine
#Don’t forget to standardize each feature.
scale = StandardScaler() 
scale.fit(X)
X_scaled = scale.transform(X)
#Plot the inertia for a range of k to determine the optimal k via elbow method:
inertia = []
for i in np.arange(1, 11)
km = KMeans(n_clusters=i)
km.fit(X_scaled)
inertia.append(km.inertia_)
#plot
plt.plot(np.arange(1, 11), inertia, marker="o")
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

#Similarly we spot that the inertia no longer decreases as rapidly after k = 3. We then finalize the model by setting n_clusters = 3 and obtain the predictions.
k_opt = 3
kmeans = KMeans(k_opt)
kmeans.fit(X_scaled)
y_pred = kmeans.predict(X_scaled)
print(y_pred)
#It is natural to ask, which model is better? Recall that clustering is an unsupervised learning method, which indicates that we don’t know the ground truth of the labels. 
#Thus it is difficult, if not impossible, to determine that the model with 2 features is more accurate in grouping wines than the one with all 13 features, or vice versa.
#In practice, the features are often chosen by the collaboration between data scientists and domain knowledge experts.


#Data Science - Pandas Pandas Pandas
#
#Finding the next centroid
#Unsupervised learning algorithm clustering involves updating the centroid of each cluster. Here we find the next centroids for given data points and initial centroids.
#Task
#Assume that there are two clusters among the given two-dimensional data points and two random points (0, 0), and (2, 2) are the initial cluster centroids. Calculate the euclidean distance between each data point and each of the centroid, assign each data point to its nearest centroid, then calculate the new centroid. If there's a tie, assign the data point to the cluster with centroid (0, 0). If none of the data points were assigned to the given centroid, return None.
#
#Input Format
#First line: an integer to indicate the number of data points (n)
#Next n lines: two numeric values per each line to represent a data point in two dimensional space.
#
#Output Format
#Two lists for two centroids. Numbers are rounded to the second decimal place.
#
#Sample Input
#3
#1 0
#0 .5
#4 0
#
#Sample Output
#[0.5 0.25]
#[4. 0.]

