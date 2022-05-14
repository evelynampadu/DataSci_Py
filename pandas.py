#Pandas is one of the most popular data science libraries in Python. Easy to use, it is built on top of NumPy and shares many functions and properties.
#With Pandas, you can read and extract data from files, transform and analyze it, calculate statistics and correlations, and much more!
#pd is a common short name used when importing the library.
#Pandas is derived from the term "panel data", an econometrics term for data sets that include observations over multiple time periods for the same individuals.
#As numpy ndarrays are homogeneous, pandas relaxes this requirement and allows for various dtypes in its data structures.
import pandas as pd 

#The two primary components of pandas are the Series and the DataFrame.
#A Series is essentially a column, and a DataFrame is a multi-dimensional table made up of a collection of Series
#You can think of a Series as a one-dimensional array, while a DataFrame is a multi-dimensional array.
#Before working with real data, let's first create a DataFrame manually to explore its functions.
#The easiest way to create a DataFrame is using a dictionary:
data = {
   'ages': [14, 18, 24, 42],
   'heights': [165, 180, 176, 184]
} 
#Each key is a column, while the value is an array representing the data for that column.

import pandas as pd
data = {
   'ages': [14, 18, 24, 42],
   'heights': [165, 180, 176, 184]
} 
df = pd.DataFrame(data)
print(df)


import pandas as pd
x = {
'a': [1, 2],
'b': [3, 4],
'c': [5, 6]
}
df = pd.DataFrame(x)
print(df)


#The DataFrame automatically creates a numeric index for each row.
#We can specify a custom index, when creating the DataFrame:
df = pd.DataFrame(data, index=['James', 'Bob', 'Amy', 'Dave'])

#Now we can access a row using its index and the loc[] function:
print(df.loc["Bob"])
#Note, that loc uses square brackets to specify the index.
#.loc[ ] allows us to select data by label or by a conditional statement.

#If we wanted to access columns order, age, and height, we can do it with .loc. .loc allows us to access any of the columns. For example, if we wanted to access columns from order through height for the first three presidents
presidents_df.loc[:, 'order':'height'].head(n=3)
#The index in pandas makes retrieving information from rows or columns convenient and easy, especially when the data set is large or there are many columns. Therefore, we don’t have to memorize the integer positions of each row or column


#indexing
print(df["ages"])
#The result is a Series object

#If we want to select multiple columns, we can specify a list of column names:
print(df[["ages", "height"]])

#slicing 
#iloc follows the same rules as slicing does with Python lists.
#Pandas uses the iloc function to select data based on its numeric index.
#It works the same way indexing lists does in Python
print(df.iloc[2])  #third row
print(df.iloc[:3])  #first 3 rows
print(df.iloc[1:3])  #rows 2 to 3
#If we do know the integer position(s), we can use .iloc to access the row(s).

#Both .loc[ ] and .iloc[ ] may be used with a boolean array to subset the data.

#Conditions
#We can also select the data based on a condition.
#For example, let's select all rows where age is greater than 18 and height is greater than 180
df[(df['ages']>18) & (df['heights']<18)] 
#Similarly, the or | operator can be used to combine conditions.


#eg: fill in the blanks to select to all rows which have rank>10 or type equal to 42.
df[(df['rank']) > 10 | (df['type']==42)]

#get the rows of the dataframe where the 'rank' column is larger than 100
df[df['rank']>100]

#Reading Data into a dataframe
#It is quite common for data to come in a file format. One of the most popular formats is the CSV (comma-separated values).
#Pandas supports reading data from a CSV file directly into a DataFrame.
#For our examples, we will use a CSV file that contains the COVID-19 infection data in California for the year 2020, called 'ca-covid.csv'.
#The read_csv() function reads the data of a CSV file into a DataFrame:
df = pd.read_csv("ca-covid.csv")
pd.options.display.max_columns = 6
#We need to provide the file path to the read_csv() function.
#Pandas also supports reading from JSON files, as well as SQL databases

#Once we have the data in a DataFrame, we can start exploring it.
#We can get the first rows of the data using the head() function of the DataFrame:
print(df.head())
#By default it returns the first 5 rows. 
#You can instruct it to return the number of rows you would like as an argument (for example, df.head(10) will return the first 10 rows).
#Similarly, you can get the last rows using the tail() function.



#The info() function is used to get essential information about your dataset, such as number of rows, columns, data types
df.info()

#We also see that Pandas has added an auto generated index.
#We can set our own index column by using the set_index() function
df.set_index("date", inplace=True)

#The date column is a good choice for our index, as there is one row for each date
#The inplace=True argument specifies that the change will be applied to our DataFrame, without the need to assign it to a new DataFrame variable
#Pandas automatically generates an index for the DataFrame, if none is specified



#Dropping a Column
df.drop('state', axis=1, inplace=True)
#drop() deletes rows and columns.
#axis=1 specifies that we want to drop a column.
#axis=0 will drop a row.
df.drop('rating', axis=1, inplace=True)


#Creating Columns
#Pandas allows us to create our own columns.
#eg. For example, we can add a month column based on the date column
df['month'] = pd.to_datetime(df['date'], format="%d.%m.%y").dt.month_name()
#We do this by converting the date column to datetime and extracting the month name from it, assigning the value to our new month column
#Our date is in DD.MM.YY format, which is why we need to specify the format attribute


#eg. add a new column called area to the dataframe, which should be the product of the height and width column values
df['area'] = df['width']*df['height']



#Summary Statistics
#The describe() function returns the summary statistics for all the numeric columns:
df.describe()
#This function will show main statistics for the numeric columns, such as std, mean, min, max values, etc.
#We can also get the summary stats for a single column, for example:
df['cases'].describe()


data = {
    'height': [133, 120, 180, 100],
    'age': [9, 7, 16, 4]
}
df = pd.DataFrame(data)
print(df['age'].mean())



#Grouping
#Since we have a month column, we can see how many values each month has, by using the value_counts() functions
df['month'].value_counts()
#value_counts() returns how many times a value appears in the dataset, also called the frequency of the values.

#fill in the blanks to get the frequency of the 'name' column
df['name'].value_counts()

#et's determine the number of total infections in each month.
#To do this, we need to group our data by the month column and then calculate the sum of the cases column for each month:
df.groupby('month')['cases'].sum()
#The groupby() function is used to group our dataset by the given column
#We can also calculate the number of total cases in the entire year
df['casses'].sum()
#Similarly, we can use min(), max(), mean(), etc. to find the corresponding values for each group.


#eg. get the maximum age for each name
df.groupby('name')['age'].max()


import pandas as pd
data = {
    'a': [1, 2, 3],
    'b': [5, 8, 4]
}
df = pd.DataFrame(data)
df['c'] = df['a']+df['b']
print(df.iloc[2]['c'])
#result  7


#eg. the dataframe df includes movie data with cloumns name,length, genre
#select the maximum length of movies for each genre
df.groupby('genre')['length'].max()


#You are working with the COVID dataset for California, which includes the number of cases and deaths for each day of 2020.
#Find the day when the deaths/cases ratio was largest.
#To do this, you need to first calculate the deaths/cases ratio and add it as a column to the DataFrame with the name 'ratio', then find the row that corresponds to the largest value.
#The output should be a DataFrame, containing all of the columns of the dataset for the corresponding row
import pandas as pd

df = pd.read_csv("/usercode/files/ca-covid.csv")

df.drop('state', axis=1, inplace=True)
df.set_index('date', inplace=True)
df['ratio'] = df['deaths']/df['cases']

print(df[ df['ratio']==df['ratio'].max() ])


#Inspect a DataFrame - Info
#Use .info() to get an overview of the DataFrame. Its output includes index, column names, count of non-null values, dtypes, and memory usage


#Columns
#We can retrieve an entire column from presidents_df by name. First we access all the column names, Which returns an index object containing all column names
presidents_df.columns

#Then we can access the column called height by:
president_df['height']
president_df['height'].shape
#Which returns a Series containing heights from all U.S. presidents.

#To select multiple columns, we pass the names in a list, resulting in a DataFrame. Remember, we can use .head() to access the first 3 rows as shown below
president_df[['height', 'age']].head(n=3)



#Min / Max / Mean
#Summary statistics include measures of location and measures of spread. 
#Measures of location are quantities that represent the average value of a variable
#measures of spread represent how similar or dissimilar the values of a variable are.

#Measures of Location - Minimum, Maximum, Mean
#Measures of Spread - Range, Variance, Standard Deviation
presidents_df.min()
presidents_df.max()
presidents_df.mean()

#Once the minimum and maximum are known, we can determine the range, a measure of spread.
#These methods work on Series as well. For example, 'presidents_df['age'].mean()' also results in 54.71


#Quantiles
#Quantiles are cut points dividing the range of the data into continuous intervals with an equal number of observations. 
#Median is the only cut point in 2-quantiles, such that 50% of the data is below the median with the other half above it
#Quartiles let us quickly divide a set of data into four groups, making it easy to see which of the four groups a particular data point is in. 
#Quartiles are then 4-quantiles, that is, 25% of the data are between the minimum and first quartile, the next is 25% between the first quartile and median, the next 25% is between the median and the third quartile, and the last 25% of the data lies between the third quartile and the maximum
presidents_df['age'].quantile([0.25, 0.5, 0.75, 1])

presidents_df['age'].mean()
presidents_df['age'].median()
#Both .quantile(0.5) and .median() result in the same output.


#Variance and Standard Deviation
#variance is the mean squared deviation of each data point from the mean of the entire dataset.
#You can think of it as how far apart a set of numbers are spread out from their average value. 
#Standard deviation (std) is the square root of variance. A high std implies a large spread, and a low std indicates a small spread, or most points are close to the mean
#In one extreme example, the data consists of all constant 2, there is no variation, thus the variation is 0.0, so is its std
const = pd.Series([2, 2, 2])
const.var()
const.std()

#eg
[2, 3, 4] 
#The mean of [2,3,4] is (2+3+4)/3 = 3.0, and its variation is (2-3)^2 + (3-3)^2 + (4-3)^2 = 1+0+1 = 2. Note that in Python, .var() will return the variance divided by N-1 where N is the length of the data, the output is then 2/(3-1) = 1
dat = pd.Series([2,3,4])
dat.mean()
dat.var()
#Outputs: 3.0 and 1.0

#And the std is just the square root of variance:
dat.std()
#Outputs: 1.0

presidents_df['age'].var()  #ans: 43.5
presidents_df['age'].std()  #ans: 6.595354

#We can apply std on the entire DataFrame to get column-wise standard deviation.
presidents_df.std()  #will output for each column
#We can apply min, max, quantile, and var on the entire DataFrame in the same way.


#describe()
#describe() prints out almost all of the summary statistics mentioned previously except for the variance. In addition, it counts all non-null values of each column
presidents_df['age'].describe()
presidents_df.describe()

#From the output we can see that there are 45 non-null data points of ages, with a mean 55 and std 6.60. The ages range from 42 to 70 with a median 55. 
#Its first and third quartiles are 51 and 58, respectively. Now we have an overall description of all age data. 
#In addition to being applied to a series, describe() can be applied to a DataFrame with multiple columns.
#.describe() ignores the null values, such as `NaN` (Not a Number) and generates the descriptive statistics that summarize the central tendency (i.e., mean), dispersion (i.e., standard deviation), and shape (i.e., min, max, and quantiles) of a dataset’s distribution.


#Categorical Variable
#The fourth column 'party' was omitted in the output of .describe() because it is a categorical variable. A categorical variable is one that takes on a single value from a limited set of categories. 
#It doesn’t make sense to calculate the mean of democratic, republican, federalist, and other parties. We can check the unique values and corresponding frequency by using .value_counts():
presidents_df['party'].value_counts()
presidents_df['party'].describe()
#Summary statistics provides us with a large amount of information put as simply as possible. The measure of location, median, is more robust than mean, for continuous variables as the latter is sensitive to outliers, e.g., extremely large values.

#Groupby
#To find the value based on a condition, we can use the groupby operation.
#The split step breaks the DataFrame into multiple DataFrames based on the value of the specified key; the apply step is to perform the operation inside each smaller DataFrame; the last step combines the pieces back into the larger DataFrame.
presidents_df.groupby('party')
#The .groupby("party") returns a DataFrameGroupBy object, not a set of DataFrames. To produce a result, apply an aggregate (.mean()) to this DataFrameGroupBy object
presidents_df.groupby('party').mean()
#The mean() method is one of many possibilities, you can apply any pandas or numpy aggregation function, or any DataFrame operation, as we demonstrate through this course

#Eg. fill in blanks to produce median height by party:
presidents_df.groupby('party')['height'].median()


#Aggregation
#We can also perform multiple operations on the groupby object using .agg() method
#It takes a string, a function, or a list thereof. For example, we would like to obtain the min, median, and max values of heights grouped by party:
presidents_df.groupby('party')['height'].agg(['min', np.median, max])
#From the output we can see, the heights of the democratic presidents range from 168 cm to 193 cm, with a median at 180 cm

#Often time we are interested in different summary statistics for multiple columns. 
#For instance, we would like to check the median and mean of heights, but minimum and maximum for ages, grouped by party. In this case, we can pass a dict with key indicate the column name, and value indicate the functions
presidents_df.groupby('party').agg({'height': [np.median, np.mean], 'age': [min, max]})
presidents_df.groupby('party').agg([min, np.median])
#Using groupby and agg provides us the flexibility and therefore the power to look into various perspectives of a variable or column conditioned on categories
presidents_df.groupby('party')['age'].agg([np.median, max])


#check the number of rows in a dtaframe
df.size
df.shape[0] #for rows
df.shape[1] #for columns


