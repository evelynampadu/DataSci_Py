#Basic SQL commands
#The SQL SHOW statement displays information contained in the database and its tables
#the SHOW DATABASES command lists the databases managed by the server.
from collections import _OrderedDictKeysView
from concurrent.futures import ProcessPoolExecutor
from ctypes import Union
from email.errors import FirstHeaderLineIsContinuationDefect
from symtable import SymbolTableFactory
from types import CoroutineType
from typing import ItemsView


SHOW DATABASES

#The SHOW TABLES command is used to display all of the tables in the currently selected MySQL database.
#SHOW COLUMNS displays information about the columns in a given table.
#The following example displays the columns in our customers table:
SHOW COLUMNS FROM customers

#SELECT Statement
#The SELECT statement is used to select data from a database.
#The result is stored in a result table, which is called the result-set.

#A query may retrieve information from selected columns or from all columns in the table.
#To create a simple SELECT statement, specify the name(s) of the column(s) you need from the table

SELECT column_list
FROM table_name

#- column_list includes one or more columns from which data is retrieved
#- table-name is the name of the table from which the information is retrieved

SELECT FirstName FROM customers;
SELECT City FROM customers;
#SQL is case insensitive.

#Syntax Rules
#A single SQL statement can be placed on one or more text lines. In addition, multiple SQL statements can be combined on a single text line.
#White spaces and multiple lines are ignored in SQL
#However, it is recommended to avoid unnecessary white spaces and lines.
#Combined with proper spacing and indenting, breaking up the commands into logical lines will make your SQL statements much easier to read and maintain.

#Selecting Multiple Columns
SELECT FirstName, LastName, City FROM customers;

#Selecting All Columns
#To retrieve all of the information contained in your table, place an asterisk (*) sign after the SELECT command, rather than typing in each column names separately
SELECT * FROM customers;
#In SQL, the asterisk means all.


#The DISTINCT Keyword
#In situations in which you have multiple duplicate records in a table, it might make more sense to return only unique records, instead of fetching the duplicates.
#The SQL DISTINCT keyword is used in conjunction with SELECT to eliminate all duplicate records and return only unique ones
SELECT DISTINCT column_name1, column_name2
FROM table_name;

#eg. there are duplicate City names. The following SQL statement selects only distinct values from the City column
SELECT DISTINT City FROM customers;
#The DISTINCT keyword only fetches the unique values.


#The LIMIT Keyword
SELECT column list
FROM table_name
LIMIT [number of records];
#we can retrieve the first five records from the customers table
SELECT ID, FirstName, LastName, City FROM customers LIMIT 5;

#You can also pick up a set of records from a particular offset.
#In the following example, we pick up four records, starting from the third
SELECT ID, FirstName, LastName, City FROM customers OFFSET 3 LIMIT 4;
SELECT ID, FirstName FROM customers LIMIT 4, 12  #show firts 12 records starting from the fifth column
#The reason that it produces results starting from ID number four, and not three, is that MySQL starts counting from zero, meaning that the offset of the first row is 0, not 1.

#Fully Qualified Names
#In SQL, you can provide the table name prior to the column name, by separating them with a dot.
SELECT City FROM customers;
SELECT customers.City FROM customers;
#The term for the above-mentioned syntax is called the "fully qualified name" of that column.

#Order By
#ORDER BY is used with SELECT to sort the returned data. The following example sorts our customers table by the FirstName column
SELECT * FROM customers
ORDER BY FirstName;
#By default, the ORDER BY keyword sorts the results in ascending order

#Sorting Multiple Columns
#ORDER BY can sort retrieved data by multiple columns. When using ORDER BY with more than one column, separate the list of columns to follow ORDER BY with commas.
SELECT * FROM customers
ORDER BY LastName, Age;
#The ORDER BY command starts ordering in the same sequence as the columns. It will order by the first column listed, then by the second, and so on

SELECT * FROM CAKES
ORDER BY CALORIES
LIMIT 3;


#The WHERE Statement
#The WHERE clause is used to extract only those records that fulfill a specified criterion
SELECT column_list 
FROM table_name
WHERE condition;

SELECT * FROM CUSTOMERS
WHERE ID = 7;


#SQL Operators
#Comparison Operators and Logical Operators are used in the WHERE clause to filter the data to be selected.
SELECT * FROM CUSTOMERS
WHERE ID != 5;  #the record with ID=5 is excluded from the list.


#The BETWEEN Operator
#The BETWEEN operator selects values within a range. The first value must be lower bound and the second value, the upper bound
SELECT column_name(s)
FROM table_name
WHERE column_name BETWEEN value1 AND value2;

SELECT * FROM CUSTOMERS
WHERE ID  BETWEEN 3 AND 7;


#Text Values
#When working with text columns, surround any text that appears in the statement with single quotation marks (')
SELECT * FROM CUSTOMERS
WHERE City = 'New York';
#If your text contains an apostrophe (single quote), you should use two single quote characters to escape the apostrophe. For example: 'Can"t'.

#Logical Operators
#Logical operators can be used to combine two Boolean values and return a result of true, false, or null.
#The following operators can be used: AND, OR, IN, NOT
SELECT ID, FirstName, LastName, Age 
FROM CUSTOMERS
WHERE Age >= 30 AND Age <= 40;
#You can combine as many conditions as needed to return the desired results.

#OR
#If you want to select rows that satisfy at least one of the given conditions, you can use the logical OR operator
SELECT * FROM CUSTOMERS
WHERE City = 'New York' OR City = 'Chicago';
#You can OR two or more conditions.

Combining AND & OR


#The SQL AND and OR conditions may be combined to test multiple conditions in a query.
#These two operators are called conjunctive operators.
#When combining these conditions, it is important to use parentheses, so that the order to evaluate each condition is known
#The statement below selects all customers from the city "New York" AND with the age equal to "30" OR “35"
SELECT * FROM CUSTOMERS
WHERE City = 'New York'
AND (Age=30 OR Age=35);
#You can nest as many conditions as you need.

#The IN Operator
#The IN operator is used when you want to compare a column with more than one value.
#You can achieve the same result with a single IN condition, instead of the multiple OR conditions
SELECT * FROM customers
WHERE City IN ('New York', 'Los Angeles', 'Chicago')
#Note the use of parentheses in the syntax.

#The NOT IN Operator
#The NOT IN operator allows you to exclude a list of specific values from the result set.
#If we add the NOT keyword before IN in our previous query, customers living in those cities will be excluded
SELECT * FROM customers 
WHERE City NOT IN ('New York', 'Los Angeles', 'Chicago');


#The CONCAT Function
#The CONCAT function is used to concatenate two or more text values and returns the concatenating string.
SELECT CONCAT(FirstName, ',', City) FROM customers;
#The CONCAT() function takes two or more parameters.

#The AS Keyword
#A concatenation results in a new column. The default column name will be the CONCAT function.
#You can assign a custom name to the resulting column using the AS keyword:
SELECT CONCAT(FirstName, ',', City) AS new_column
FROM customers;


#Arithmetic Operators
#Arithmetic operators perform arithmetical operations on numeric operands. The Arithmetic operators include addition (+), subtraction (-), multiplication (*) and division (/)
SELECT ID, FirstName, LastName, Salary+500 AS salary  
FROM employees;
#The example below adds 500 to each employee's salary and selects the result

#The UPPER Function
#The UPPER function converts all letters in the specified string to uppercase.
#The LOWER function converts the string to lowercase.
SELECT ID, FirstName, UPPER(LastName) AS LastName 
FROM employees;
#If there are characters in the string that are not letters, this function will have no effect on them

#SQRT and AVG
#The SQRT function returns the square root of given value in the argument.
SELECT Salary, SQRT(Salary)
FROM employees;

#Similarly, the AVG function returns the average value of a numeric column
SELECT Avg(Salary) FROM employees;
#Another way to do the SQRT is to use POWER with the 1/2 exponent. However, SQRT seems to work faster than POWER in this case

#The SUM function
#The SUM function is used to calculate the sum for a column's values.
SELECT SUM(Salary) FROM employees;

#Subqueries
#A subquery is a query within another query.
#Let's consider an example. We might need the list of all employees whose salaries are greater than the average.
SELECT AVG(Salary) FROM employees;
#As we already know the average, we can use a simple WHERE to list the salaries that are greater than that number
SELECT FirstName, Salary FROM employees;
WHERE Salary > 3100
ORDER BY Salary DESC;
#The DESC keyword sorts results in descending order.
#Similarly, ASC sorts the results in ascending order.

#Subqueries
SELECT FirstName, Salary FROM employees
WHERE Salary > (SELECT Avg(Salary) FROM employees)
ORDER BY Salary DESC;
#Enclose the subquery in parentheses.
#Also, note that there is no semicolon at the end of the subquery, as it is part of our single query.

#The Like Operator
#The LIKE keyword is useful when specifying a search condition within your WHERE clause.
SELECT column_name(s)
FROM table_name
WHERE column_name LIKE pattern;

#SQL pattern matching enables you to use "_" to match any single character and "%" to match an arbitrary number of characters (including zero characters).
#For example, to select employees whose FirstNames begin with the letter A, you would use the following query:
SELECT * FROM employees
WHERE FirstName LIKE 'A%';
#the following SQL query selects all employees with a LastName ending with the letter "s":
SELECT * FROM employees
WHERE LastName LIKE '%s';
#The % wildcard can be used multiple times within the same pattern.

#The MIN Function
#The MIN function is used to return the minimum value of an expression in a SELECT statement.
ELECT MIN(Salary) AS salary FROM employees;
#All of the SQL functions can be combined together to create a single expression


#Joining Tables
#All of the queries shown up until now have selected from just one table at a time.
#One of the most beneficial features of SQL is the ability to combine data from two or more tables.
#In SQL, "joining tables" means combining data from two or more tables. A table join creates a temporary table showing the data from the joined tables.

SELECT customers.ID, customers.Name, orders.Name, orders.Amount
FROM customers, orders 
WHERE customers.ID=orders.Customer_ID
ORDER BY customers.ID;
#Each table contains "ID" and "Name" columns, so in order to select the correct ID and Name, fully qualified names are used.
#Note that the WHERE clause "joins" the tables on the condition that the ID from the customers table should be equal to the customer_ID of the orders table
#Specify multiple table names in the FROM by comma-separating them.

SELECT ct.ID, ct.Name, ord.Name, ord.Amount
FROM customers AS ct, orders AS ord
WHERE ct.ID=ord.Customer_ID
ORDER BY ct.ID;
#As you can see, we shortened the table names as we used them in our query

#Types of Join
#The following are the types of JOIN that can be used in MySQL:
#- INNER JOIN
#- LEFT JOIN
#- RIGHT JOIN
#INNER JOIN is equivalent to JOIN. It returns rows when there is a match between the tables.
SELECT column_name(s)
FROM table1 INNER JOIN table2 
ON table1.column_name=table2.column_name;
#Note the ON keyword for specifying the inner join condition.
#Only the records matching the join condition are returned.


#LEFT JOIN
#The LEFT JOIN returns all rows from the left table, even if there are no matches in the right table.
#This means that if there are no matches for the ON clause in the table on the right, the join will still return the rows from the first table in the result.
SELECT table1.column1, table2.column2...
FROM table1 LEFT OUTER JOIN table2
ON table1.column_name = table2.column_name;
#The OUTER keyword is optional, and can be omitted.

SELECT customers.Name, items.Name
FROM customers LEFT JOIN Items
ON customers.ID=items.Seller_id;
#The result set contains all the rows from the left table and matching data from the right table
#If no match is found for a particular row, NULL is returned.

#RIGHT JOIN
#The RIGHT JOIN returns all rows from the right table, even if there are no matches in the left table.
SELECT table1.column1, table2.column2...
FROM table1 RIGHT OUTER JOIN table2
ON table1.column_name = table2.column_name;
#Again, the OUTER keyword is optional, and can be omitted.

SELECT customers.Name, items.Name 
FROM customers RIGHT JOIN items 
ON customersID=items.seller_ID
#The RIGHT JOIN returns all the rows from the right table (items), even if there are no matches in the left table (customers).
#There are other types of joins in the SQL language, but they are not supported by MySQL.


#Set Operation
#Occasionally, you might need to combine data from multiple tables into one comprehensive dataset. This may be for tables with similar data within the same database or maybe there is a need to combine similar data across databases or even across servers.
#To accomplish this, use the UNION and UNION ALL operators.
#UNION combines multiple datasets into a single dataset, and removes any existing duplicates.
#UNION ALL combines multiple datasets into one dataset, but does not remove duplicate rows
#UNION ALL is faster than UNION, as it does not perform the duplicate removal operation over the data set

#UNION
#The UNION operator is used to combine the result-sets of two or more SELECT statements.
#All SELECT statements within the UNION must have the same number of columns. The columns must also have the same data types. Also, the columns in each SELECT statement must be in the same order.
SELECT column_name(s) FROM table1
UNION
SELECT column_name(s) FROM table2;


SELECT ID, FirstName, LastName, City FROM First 
UNION 
SELECT ID, FirstName, LastName, City FROM First 
#As you can see, the duplicates have been removed.
#If your columns don't match exactly across all queries, you can use a NULL (or any other) value such as
SELECT FirstName, LastName, Company FROM businessContacts
UNION
SELECT FirstName, LastName, NULL FROM otherContacts;
#The UNION operator is used to combine the result-sets of two or more SELECT statements.

#UNION ALL
#UNION ALL selects all rows from each table and combines them into a single table.

SELECT ID, FirstName, LastName, City FROM First 
UNION ALL 
SELECT ID, FirstName, LastName, City FROM First
#As you can see, the result set includes the duplicate rows as well.

#Inserting Data
#SQL tables store data in rows, one row after another. The INSERT INTO statement is used to add new rows of data to a table in the database.
INSERT INTO table_name
VALUES (value1, value2, value3,...);
#Make sure the order of the values is in the same order as the columns in the table.

INSERT INTO Employees
VALUES (8, 'Anthony', 'Young', 35)
#When inserting records into a table using the SQL INSERT statement, you must provide a value for every column that does not have a default value, or does not support NULL.

#Inserting Data
#Alternatively, you can specify the table's column names in the INSERT INTO statement:
INSERT INTO table_name (column1, column2, column3, ...,columnN)  
VALUES (value1, value2, value3,...valueN);

INSERT INTO Employees (ID, FirstName, LastName, Age)
VALUES (8, 'Anthony', 'Young', 35)
#You can specify your own column order, as long as the values are specified in the same order as the columns.

#Updating Data
#The UPDATE statement allows us to alter data in the table.
#The basic syntax of an UPDATE query with a WHERE clause is as follows
UPDATE table_name
SET column1=value1, column2=value2, ...
WHERE condition;
#You specify the column and its new value in a comma-separated list after the SET keyword
#If you omit the WHERE clause, all records in the table will be updated!

UPDATE Employees
SET Salary=5000
WHERE ID=1;

#Updating Multiple Columns
UPDATE Employees
SET Salary=5000, FirstName='Robert'
WHERE ID=1;
#You can specify the column order any way you like in the SET clause.


#Deleting Data
#The DELETE statement is used to remove data from your table. DELETE queries work much like UPDATE queries.
DELETE FROM table_name
WHERE condition; 

DELETE FROM Employees
WHERE ID=1;
#If you omit the WHERE clause, all records in the table will be deleted!
#The DELETE statement removes the data from the table permanently


#SQL Tables
#A single database can house hundreds of tables, each playing its own unique role in the database schema.
#SQL tables are comprised of table rows and columns. Table columns are responsible for storing many different types of data, including numbers, texts, dates, and even files.
#The CREATE TABLE statement is used to create a new table
#Creating a basic table involves naming the table and defining its columns and each column's data type.
CREATE TABLE table_name
(
column_name1 data_type(size),
column_name2 data_type(size),
column_name3 data_type(size),
....
columnN data_type(size)
);

#- The column_names specify the names of the columns we want to create.
#- The data_type parameter specifies what type of data the column can hold. For example, use int for whole numbers.
#- The size parameter specifies the maximum length of the table's column.
#Note the parentheses in the syntax.

#Creating a Table
#Assume that you want to create a table called "Users" that consists of four columns: UserID, LastName, FirstName, and City.
#Use the following CREATE TABLE statement:
CREATE TABLE Users
(
   UserID int,
   FirstName varchar(100), 
   LastName varchar(100),
   City varchar(100)
);
#varchar is the datatype that stores characters. You specify the number of characters in the parentheses after the type. So in the example above, our fields can hold max 100 characters long text.


#Data Types
#Data types specify the type of data for a particular column.
#If a column called "LastName" is going to hold names, then that particular column should have a "varchar" (variable-length character) data type.
#The most common data types:
#Numeric
#INT -A normal-sized integer that can be signed or unsigned.
#FLOAT(M,D) - A floating-point number that cannot be unsigned. You can optionally define the display length (M) and the number of decimals (D).
#DOUBLE(M,D) - A double precision floating-point number that cannot be unsigned. You can optionally define the display length (M) and the number of decimals (D).
#
#Date and Time
#DATE - A date in YYYY-MM-DD format.
#DATETIME - A date and time combination in YYYY-MM-DD HH:MM:SS format.
#TIMESTAMP - A timestamp, calculated from midnight, January 1, 1970
#TIME - Stores the time in HH:MM:SS format.
#
#String Type
#CHAR(M) - Fixed-length character string. Size is specified in parenthesis. Max 255 bytes.
#VARCHAR(M) - Variable-length character string. Max size is specified in parenthesis.
#BLOB - "Binary Large Objects" and are used to store large amounts of binary data, such as images or other types of files.
#TEXT - Large amount of text data.
#
#Choosing the correct data type for your columns is the key to good database design.


#Primary Key
#The UserID is the best choice for our Users table's primary key.
#Define it as a primary key during table creation, using the PRIMARY KEY keyword.
CREATE TABLE Users
(
   UserID int,
   FirstName varchar(100),
   LastName varchar(100),
   City varchar(100),
   PRIMARY KEY(UserID)
);
#Specify the column name in the parentheses of the PRIMARY KEY keyword.

#SQL Constraints
#SQL constraints are used to specify rules for table data.
#The following are commonly used SQL constraints:
#NOT NULL - Indicates that a column cannot contain any NULL value.
#UNIQUE - Does not allow to insert a duplicate value in a column. The UNIQUE constraint maintains the uniqueness of a column in a table. More than one UNIQUE column can be used in a table.
#PRIMARY KEY - Enforces the table to accept unique data for a specific column and this constraint create a unique index for accessing the table faster.
#CHECK - Determines whether the value is valid or not from a logical expression.
#DEFAULT - While inserting data into a table, if no value is supplied to a column, then the column gets the value set as DEFAULT.
#For example, the following means that the name column disallows NULL values.
name varchar(100) NOT NULL
#During table creation, specify column level constraint(s) after the data type of that column.


#AUTO INCREMENT
#Auto-increment allows a unique number to be generated when a new record is inserted into a table.
#Often, we would like the value of the primary key field to be created automatically every time a new record is inserted.
#By default, the starting value for AUTO_INCREMENT is 1, and it will increment by 1 for each new record.
#Let's set the UserID field to be a primary key that automatically generates a new value:
UserID int NOT NULL AUTO_INCREMENT,
PRIMARY KEY (UserID)
#Auto-increment allows a unique number to be generated when a new record is inserted into a table.

#Using Constraints
CREATE TABLE Users (
id int NOT NULL AUTO_INCREMENT,
username varchar(40) NOT NULL, 
password varchar(10) NOT NULL,
PRIMARY KEY(id)
);
#The following SQL enforces that the "id", "username", and "password" columns do not accept NULL values. We also define the "id" column to be an auto-increment primary key field.
#When inserting a new record into the Users table, it's not necessary to specify a value for the id column; a unique new value will be added automatically.


#ALTER TABLE
#The ALTER TABLE command is used to add, delete, or modify columns in an existing table.
#You would also use the ALTER TABLE command to add and drop various constraints on an existing table.
ALTER TABLE People ADD DateOfBirth date;
#All rows will have the default value in the newly added column, which, in this case, is NULL.

ALTER TABLE People 
DROP COLUMN DateOfBirth;
#The column, along with all of its data, will be completely removed from the table.

DROP TABLE People;
#Be careful when dropping a table. Deleting a table will result in the complete loss of the information stored in the table!

#Renaming
ALTER TABLE People 
RENAME FirstName TO name;
#This query will rename the column called FirstName to name.

RENAME TABLE People TO Users;
#You can rename the entire table using the RENAME command


#Views
#In SQL, a VIEW is a virtual table that is based on the result-set of an SQL statement.
#A view contains rows and columns, just like a real table. The fields in a view are fields from one or more real tables in the database.
#Views allow us to:
#- Structure data in a way that users or classes of users find natural or intuitive.
#- Restrict access to the data in such a way that a user can see and (sometimes) modify exactly what they need and no more.
#- Summarize data from various tables and use it to generate reports.
CREATE VIEW view_name AS
SELECT column_name(s)
FROM table_name
WHERE condition;
#The SELECT query can be as complex as you need it to be. It can contain multiple JOINS and other commands.

CREATE VIEW List AS
SELECT FirstName, Salary
FROM  Employees;
#create a view that displays each employee's FirstName and Salary.
#Now, you can query the List view as you would query an actual table.
SELECT * FROM List; 
#A view always shows up-to-date data! The database engine uses the view's SQL statement to recreate the data each time a user queries a view.

#Updating a View
CREATE OR REPLACE VIEW view_name AS
SELECT column_name(s)
FROM table_name
WHERE condition;

#The example below updates our List view to select also the LastName:
CREATE OR REPLACE VIEW LIST AS 
SELECT FirstName, LastName, Salary 
FROM Employees;

#You can delete a view with the DROP VIEW command.
DROP VIEW List;
#It is sometimes easier to drop a table and recreate it instead of using the ALTER TABLE statement to change the table’s definition.


#You manage a zoo. Each animal in the zoo comes from a different country. Here are the tables you have:Animals
#1) A new animal has come in, with the following details:
#name - "Slim", type - "Giraffe", country_id - 1
#Add him to the Animals table.
#2) You want to make a complete list of the animals for the zoo’s visitors. Write a query to output a new table with each animal's name, type and country fields, sorted by countries.

INSERT INTO Animals (name, type, country_id) 
VALUES ('Slim', 'Giraffe', 1);

SELECT animals.name, animals.type, countries.country 
FROM animals, countries
WHERE animals.country_id=countries.id 
order by animals.country_id desc;