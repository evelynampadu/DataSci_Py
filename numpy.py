#NumPy (Numerical Python) is a Python library used to work with numerical data.
#NumPy includes functions and data structures that can perform a wide variety of mathematical operations.
#To start using NumPy, we first need to import it: import numpy as np 
#NumPy arrays are often called ndarrays, which stands for "N-dimensional array", because they can have multiple dimensions.
#Another characteristic about numpy array is that it is homogeneous, meaning each element must be of the same data type
#To check the data type, use numpy.ndarray.dtype
import numpy as np

x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(x.ndim) # 2
print(x.size) # 9
print(x.shape) # (3, 3)

#This will create a 2-dimensional array, which has 3 columns and 3 rows, and output the value at the 2nd row and 3rd column.
#If we reshape the heights_arr to (45,1), the same as 'ages_arr', we can stack them horizontally (by column) to get a 2darray using 'hstack'
arr3 = np.hstack((arr1, arr2))
#or
arr3 = np.concatenate((arr1, arr2), axis=1  #You can use np.hstack to concatenate arrays ONLY if they have the same number of rows.)
#Similarly, if we want to combine the arrays vertically (by row), we can use 'vstack
#To combine arr1 of shape(10,2) and arr(5,2) into a new array of (15,2)
arr3 = np.vstack((arr1, arr2))

#or
arr3 = np.concatenate((arr1, arr2), axis=0)

#Arrays have properties, which can be accessed using a dot.
#ndim returns the number of dimensions of the array.
#size returns the total number of elements of the array.
#shape returns a tuple of integers that indicate the number of elements stored along each dimension of the array.


#We can add, remove and sort an array using the np.append(), np.delete() and np.sort() functions

import numpy as np

x = np.array([2, 1, 3])

x = np.append(x, 4)
x = np.delete(x, 0)
x = np.sort(x)
print(x)


#np.arange() allows you to create an array that contains a range of evenly spaced intervals (similar to a Python range)
import numpy as np

x = np.arange(2, 10, 3)
print(x)


#NumPy allows us to change the shape of our arrays using the reshape() function. For example, we can change our 1-dimensional array to an array with 3 rows and 2 columns
x = np.arange(1, 7)
z = x.reshape(3, 2)
#Reshape - Numpy can calculate the shape (dimension) for us if we indicate the unknown dimension as -1. For example, given a 2darray `arr` of shape (3,4), arr.reshape(-1) would output a 1darray of shape (12,), while arr.reshape((-1,2)) would generate a 2darray of shape (6,2)

#Reshape can also do the opposite: take a 2-dimensional array and make a 1-dimensional array from it:
#The same result can be achieved using the flatten() function.
x = np.array([1, 2], [3, 4], [5, 6])
z = x.reshape(6)
#The result is a flat array that contains 6 elements.

#NumPy arrays can be indexed and sliced the same way that Python lists are
x = np.arange(1, 10)
print(x[0:3])
print(x[5:])
print(x[:2])
print(x[-3:])
#Negative indexes count from the end of the array, so, [-3:] will result in the last 3 elements

import numpy as np
x = np.arange(1, 10)
print(x[x<4])


#It is easy to perform basic mathematical operations with arrays.
#For example, to find the sum of all elements, we use the sum() function:
import numpy as np
x = np.arange(1, 10)
print(x.sum())

#Similarly, min() and max() can be used to get the smallest and largest elements.

x = np.arange(1, 10)
y = x*2
print(y)

#NumPy understands that the given operation should be performed with each element. This is called broadcasting.


import numpy as np
x = np.array([14, 18, 19, 24, 26, 33, 42, 55, 67])
print(np.mean(x))
print(np.median(x))
print(np.std(x))
print(np.var(x))


x = np.arange(3, 9)
z = x.reshape(2, 3)
print(z[1][1])
#result is 7


import numpy as np
x = np.arange(1, 5)
x = x*2
print(x[:3].sum())
#result is 12


#You are given an array that represents house prices.
#Calculate and output the percentage of houses that are within one standard deviation from the mean.
import numpy as np

data = np.array([150000, 125000, 320000, 540000, 200000, 120000, 160000, 230000, 280000, 290000, 300000, 500000, 420000, 100000, 150000, 280000])
dmean = np.mean(data)
dstd = np.std(data)
a = dmean + dstd
b = dmean - dstd
one = data[(data <= a) & (data >= b)]
units = np.size(data)
print(np.size(one)/np.size(data) * 100)



import numpy as np 
a = np.array([0,0,0,0,0,
            1,1,1,1,1,1,1,1,
            2,2,2,2,
            3,3,3
            ])
#your code goes here


mean = np.sum(a)/a.size
v = np.sum((a-mean)**2)/a.size
print(v)

#In order to sum all heights and sum all ages separately, we can specify axis=0 to calculate the sum across the rows, that is, it computes the sum for each column, or column sum. On the other hand, to obtain the row sums specify axis=1
arr.sum(axis=0)


#Masking (works like filters)
#Masking is used to extract, modify, count, or otherwise manipulate values in an array based on some criterion. In our example, the criteria was height of 182cm or taller.
mask = height[:,0] >= 182  #index means whole row, 1st column
mask.sum()
#Then pass it to the first axis of `height_age_arr` to filter presidents who don’t meet the criteria
tall = height[mask, ]

#eg. Obtain a subset of presidents who started their presidency under age 50
mask = age[:, 1] < 50  #index means whole row of 2nd column
young = age[mask, ]

#We can create a mask satisfying more than one criteria
#For example, an a 2dimensional array of height & age, in addition to height, we want to find those presidents that were 50 years old or younger at the start of their presidency. To achieve this, we use & to separate the conditions and each condition is encapsulated with parentheses "()"
mask = (heightagearr[:, 0]) >= 182 & (heightagearr[:, 1] <50)
heightagearr[mask,]



#In a matrix, or 2-d array X, the averages (or means) of the elements of rows is called row means.
#Task
#Given a 2D array, return the rowmeans.
#Input Format
#First line: two integers separated by spaces, the first indicates the rows of matrix X (n) and the second indicates the columns of X (p)
#Next n lines: values of the row in X
#Output Format
#An numpy 1d array of values rounded to the second decimal

import numpy as np
n, p = [int(x) for x in input().split()]
x= [] # array for the list

for i in range(n): # taking input for each row
    x.append(input().split()) # taking input and spliting the data into columnwise input

arr = np.array(x) # making this numpy array
arr = arr.astype(np.float16)#making data type into float

mn = arr.mean(axis = 1) # having mean value in rowwise  with axis = 1
mean= mn.round(2) #as output need to be two value after point value
print(mean)

#The split step breaks the DataFrame into multiple DataFrames based on the value of the specified key; 
#the apply step is to perform the operation inside each smaller DataFrame; 
#the combine step combines the pieces back into the larger DataFrame.



#Task
#Given a list of numbers and the number of rows (r), reshape the list into a 2-dimensional array. Note that r divides the length of the list evenly.
#
#Input Format
#First line: an integer (r) indicating the number of rows of the 2-dimensional array
#Next line: numbers separated by the space
#
#Output Format
#An numpy 2d array of values rounded to the second decimal.
#
#Sample Input
#2
#1.2 0 0.5 -1
#
#Sample Output
#[[ 1.2 0. ]
#[ 0.5 -1. ]]
import numpy as np
r = int(input()) 
lst = [float(x) for x in input().split()]
arr = np.array(lst).reshape(r, int(len(lst)/r))
print(arr)