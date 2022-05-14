#Matplotlib is a library used to create graphs, charts, and figures. It also provides functions to customize your figures by changing the colors, labels, etc.
import matplotlib.pyplot as plt 
#pyplot is the module we will be using to create our plots.
#plt is a common name used for importing this module.
#matplotlib.pyplot is a collection of functions that make plotting in python work like MATLAB.
#Each function makes some change to a figure, e.g., creates a figure, creates a plotting area in a figure, plots lines, annotates the plots with labels, etc. as we will see in the following lessons.

#Matplotlib works really well with Pandas!

s = pd.Series([18, 42, 9, 32, 81, 64, 3])
s.plot(kind='bar')
plot.savefig(plot.png)

#The data from the series is using the Y axis, while the index is plotted on the X axis.
#As we have not provided a custom index for our data, the default numeric index is used
#plt.savefig('plot.png') is used to save and display the chart in our Code Playground.
#In most environments this step is not needed, as calling the plot() function automatically displays the chart.

#For all matplotlib plots, first create a figure and an axes object; to show the plot, call “plt.show()”. 
#The figure contains all the objects, including axes, graphics, texts, and labels. The axes is a bounding box with ticks and labels. Think of axes as an individual plot.
import matplotlib.pyplot as plt 
fig = plt.figure()
ax = plt.axes()
plt.show()

#LinePlot
#Let’s start with a beautiful wave function, sine function, sin(x), where x ranges from 0 to 10. 
#We need to generate the sequence along the x-axis, an evenly spaced array, via linspace()
import numpy as np
x = np.linespace(0, 10, 1000)  #x is a 1000 evenly spaced
numbers from 0 to 10
#The second line generates an evenly spaced sequence of 1000 numbers o 0 to 10. 
#You can view it as tides go up and down over time and the height of the tides are obtained by the sin function
y = np.sin(x)
#To plot, as before, we first create the figure and axes objects.
import matplotlib.pyplot as plt 
fig = plt.figure()
ax = plt.axes()
#We now make a plot directly from the Axes, "ax.plot()"; by default, it generates a Line2D object. To show the plot, we need to call show().
ax.plot(x, y)
plt.show()
#A line plot displays data along a number line, a useful tool to track changes over short and long periods of time

#eg. plot the cosine function from x= 0 to 10
x = np.linspace(0, 10, 1000)
plt.plot(x, np.cos(x))

#Labels and Titles
#The job of the title is to accurately communicate what the figure is about. In addition, axes need titles, or more commonly referred to as axis labels. 
#The axis labels explain what the plotted data values are. We can specify the x and y axis labels and a title using plt.xlabel(), plt.ylabel() and plt.title().
x = np.linspace(0, 10, 1000)
y = np.sin(x)
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('function sin(x)')
plt.show
#One can also set the limits of x- and y-axis using plt.xlim() and plt.ylim(), respectively.
#eg.
plt.plot(x, y)
plt.ylabel('sin(x)')

#Multiple Lines
#We can plot multiple lines on the same figure. 
#Say, the sin function capture the tides on the east coast and cos function capture the tides on the west coast at the same time, we can plot them both on the same figure by calling the .plot() function multiple times.
x = np.linspace(0, 10, 1000)  #1d array of length 1000
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
plt.show()
#Colors and line styles can be specified to differentiate lines:
x = np.linspace(0, 10, 1000)
plt.plot(x, np.sin(x), color='k')
plt.plot(x, np.cos(x), color='r', linestyle='--')
plt.show()
#Note that we specified basic colors using a single letter, that is, k for black and r for red. More examples include b for blue, g for green, c for cyan, etc. 
#It is sometimes helpful to compare different datasets visually on the same figure and matplotlib makes it easy
#Eg. plot y=x**2 in black and y2 = x in blue as x ranges from 0 to 2
x = np.linspace(0, 2, 100)
y1 = x**2
y2 = x
plt.plot(x, y1, color='k')
plt.plot(x, y2, color='b')
plt.show()

#Legend/Key
#When there are multiple lines on a single axes, it’s often useful to create a plot legend labeling each line. 
#We can use the method plt.legend(), in conjunction with specifying labels in the plt.plot()
x = np.linspace(0, 10, 1000)
plt.plot(x, np.sin(x), 'k:', label='sin(x)')
plt.plot(x, np.cos(x), 'r--', label='cos(x)')
plt.legend()
plt.show()
#Note here we use 'k:' to indicate the line of sin function to be black (indicated by k) and dotted (indicated by :). 
#Line style and color codes can be combined into a single non-keyword argument in the plt.plot() function
#The color, line style (e.g., solid, dashed, etc.), and position, size, and style of labels can be modified using optional arguments to the function

#Matplotlib supports the creation of different chart types.
#Let's start with the most basic one -- a line chart
#To create a line chart we simply need to call the plot() function on our DataFrame.
df[df['month']==12]['cases'].plot()  #12 is for the month of Dec
df[df['month']==6]['deatgs'].plot()   #6 is for the month of June


#We can also include multiple lines in our chart.
#For example, let's also include the deaths column in our DataFrame:
(df[df['month']==12])[['cases', 'deaths']].plot()
#matplotlib automatically added a legend to show the colors of the lines for the columns


#Barplot
#The plot() function can take a kind argument, specifying the type of the plot we want to produce.
#For bar plots, provide kind="bar".
#Let's make a bar plot for the monthly infection cases

(df.groupby('month')['cases'].sum().plot(kind="bar"))
#We first group the data by the month column, then calculate the sum of the cases in that month

#calculate monthly deaths and create a bar chart
df = df.groupby('month')
df['deaths'].sum().plot(kind='bar')


#We can also plot multiple columns.
#The stacked property can be used to specify if the bars should be stacked on top of each other
df = df.groupby('monthly')[['cases', 'deaths']].sum()
df.plot(kind="bar", stacked=True)
#kind="barh" can be used to create a horizontal bar chart.

#create a horizontal bar plot showing the average age for each gender. Set the title of the plot to "Average Age"
df.groupby('gender')['age'].mean().plot(kind="barh")

#read the dataframe and create a bar chart showing the average height for each age
df = pd.read_csv('people.csv')
df.set_index('age', inplace=True)  #puts in on the x-axis
df = df.groupby('age')['heignt'].mean()
df.plot(kind="bar")

#eg
party_cnt = presidents_df['party'].value_counts()
plt.style.use('ggplot')
party_cnt.plot(kind ='bar')
plt.show()
#Bar plots are commonly confused with a histogram. Histogram presents numerical data whereas bar plot shows categorical data


#Boxplot
#A box plot is used to visualize the distribution of values in a column, basically visualizing the result of the describe() function.
#For example, let's create a box plot for the cases in June:
df[df["month"]==6]["cases"].plot(kind="box")
#The green line shows the median value.
#The box shows the upper and lower quartiles (25% of the data is greater or less than these values).
#The circles show the outliers, while the black lines show the min/max values excluding the outliers.
plt.style.use('classic')
presidents_df.boxplot(column='height');
#The box-and-whisker plot doesn’t show frequency, and it doesn’t display each individual statistic, but it clearly shows where the middle of the data lies and whether the data is skewed

#Histogram
#Similar to box plots, histograms show the distribution of data.
#Visually histograms are similar to bar charts, however, histograms display frequencies for a group of data rather than an individual data point; therefore, no spaces are present between the bars.
df[df["month"]==6]["cases"].plot(kind="hist")
#You can manually specify the number of bins to use using the bins attribute: plot(kind="hist", bins = 10)
#A histogram is a diagram that consists of rectangles with width equal to the interval and area proportional to the frequency of a variable
presidents_df['height'].plot(kind='hist', title = 'height', bins=5)
plt.show()

#or
plt.hist(presidents_df['height'], bins=5)
plt.show()
#In addition, plt.hist() outputs a 1darray with frequency, and bin end points.
#A histogram shows the underlying frequency distribution of a set of continuous data, allowing for inspection of the data for its shape, outliers, skewness, etc.


#Area Plot
#kind='area' creates an Area plot
df[df["month"]==6][["cases", "deaths"]].plot(kind="area", stacked=False)
#Area plots are stacked by default, which is why we provided stacked=False explicitly.


#Scatter Plot
#A scatter plot is used to show the relationship between two variables.
#For example, we can visualize how the cases/deaths are related
df[df["month"]==6][["cases", "deaths"]].plot(kind="scatter", x='cases', y='deaths')
#We need to specify the x and y columns to be used for the plot.
#points are represented individually with a dot, or another shape. 
#Pass 'o' in the plt.plot(), or use plt.scatter() to show the relationship of heights and ages:
plt.scatter(presidents_df['height']),
presidents_df['age']
plt.show()
#By default, each data point is a full circle, and there is a good collection of other shapes. 
#For example, we can pass '<' to draw a triangle pointing to the left, in addition we specify the color to blue:
plt.scatter(presidents_df['height'],
presidents_df['age'],
marker='<', #or '^'
color='b')
plt.xlabel('height');
plt.ylabel('age')
plt.title('U.S. presidents')
plt.show()
#Scatter plot is a useful tool to display the relationship between two features; whereas a line plot puts more emphasis on the change between points as its slope depicts the rate of change.

#Plotting with Pandas
#A great thing about pandas is that it integrates well with matplotlib, so we can plot directly from DataFrames and Series
#We specify the kind of plot as 'scatter', 'height' along x-axis, and 'age' along y-axis, and then give it a title:
presidents_df.plot(kind='scatter', x='height', y='age', title='U.S. presidents')
plt.show()
#As we specified the x-axis and y-axis with column names from the DataFrame, the labels were also annotated on the axes


#Pie Chart
#We can create a pie chart using kind="pie".
#Let's create one for cases by month:
df.groupby('month')['cases'].sum().plot(kind="pie")

#create a pie chart showing the number of cases for each week day
df = df.groupby('weeksday')
df['cases'].sum().plot(kind="pie")

#eg
data = {
    'sport': ["soccer", "tennis", "soccer", "hockey"]
    'players': [5,4,8,20]
}
df = pd.DataFrame(data)
df.groupby('sport')['players'].sum().plot(kind="pie")


#Plot Formatting
#Matplotlib provides a number of arguments to customize your plot.
#The legend argument specifies whether or not to show the legend.
#You can also change the labels of the axis by setting the xlabel and ylabel arguments:
df[['cases', 'deaths']].plot(kind="line", legend=True)
plt.xlabel('Days in June')
plt.ylabel('Number')
#By default, pandas select the index name as xlabel, while leaving it empty for ylabel. By default, the vertical axis has no label

#The suptitle() function can be used to set a plot title:
plt.suptitle("COVID-19 in June")

#We can also change the colors used in the plot by setting the color attribute. It accepts a list of color hexes.
#For example, let's set the cases to blue, deaths to red colors:
df[['cases', 'deaths']].plot(kind="area", legend=True, stacked=False, color=['#1970E7', '#E73E19'])

#These attributes work for almost all chart types.

