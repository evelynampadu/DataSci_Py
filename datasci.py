vac_nums = [0,0,0,0,0,
            1,1,1,1,1,1,1,1,
            2,2,2,2,
            3,3,3
            ]
#your code goes here
sum = 0
for i in vac_nums:
    sum += i

mean = sum/len(vac_nums)
print(mean)



import numpy as np

mean = np.mean(players)
standard_deviation = np.std(players)
variance = np.var(players)
result = 0
for i in players:
    if i >= mean-standard_deviation and i <= mean+standard_deviation:
        result += 1
print(result)


#Imputing missing values.
#In the real world, you will often need to handle missing values. One way to impute (i.e., fill) the numerical column is to replace the null values with its mean.
#Task
#Given a list of numbers including some missing values, turn it into a pandas series, impute the missing values with the mean, and finally return the series.
#Input Format
#A list of numbers including one or more string "nan" to indicate a missing value.
#
#Output Format
#A list of imputed values where all values are rounded to its first decimal place.
#
#Sample Input
#3 4 5 3 4 4 nan
#
#Sample Output
#0 3.0
#1 4.0
#2 5.0
#3 3.0
#4 4.0
#5 4.0
#6 3.8
#dtype: float64

import numpy as np
import pandas as pd

lst = [float(x) if x != 'nan' else np.NaN for x in input().split()]

df = pd.DataFrame(lst)
df = df.fillna(df.mean().round(1))

print(df[0].to_string())
print('dtype: '+str(df[0].dtypes))