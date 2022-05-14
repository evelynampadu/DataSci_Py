def do_calculation():
    print("lets " + command + " some numbers")
    input1 = input("Number 1> ")
    input2 = input("Number 2> ")
    number1 = int(input1)
    number2 = int(input2)
    if command == "add":
        result = number1 + number2
        operator = " + "
    elif command == "subtract":
        result = number1 - number2
        operator = " - "
    output = str(result)
    print(input1 + operator + input2 + " = " + output)

    do_calculation() #call function to run


def my_func():
   print("spam")
   print("spam")
   print("spam")


def welcome():
   user = input()
   print("Welcome, user")
welcome()

#Functions can take arguments, which can be used to generate the function output.
def exclamation(word):
   print(word + "!")
exclamation("spam")



def exclamation(word):
   print(word + "!")
exclamation("spam")
exclamation("eggs")
exclamation("python")


def x(y):
   print(y+2)
x(5)

#you can define functions with more than one argument; separate them with commas.
def print_sum_twice(x, y):
   print(x + y)
   print(x + y)
print_sum_twice(5, 8)


def printBill(text):
  print("======")
  print(text)
  print("======")

printBill(input())


def sum(x, y):
   return x+y 

def sum(x, y):
  return x+y
res = sum(42, 7)
print(res)


def max(x, y):
  if x >=y:
    return x
  else:
    return y
if(max(6, 4) > 10):
  print("Yes")
else:
  print("Nope")


#Once you return a value from a function, it immediately stops being executed.
#Any code placed after the return statement won’t be executed


def double(a, b):
   return [a*2, b*2]
x = double(6, 9)
print(x)


def area(x, y):
   #your code goes here
   return(x*y)

w = int(input())
h = int(input())

#call the function
print(area(w,h))


#Docstrings (documentation strings) are similar to comments, in that they’re designed to explain code. But, they’re more specific and have a different syntax.
#They’re created by putting a multiline string containing an explanation of the function below the function's first line


text = input()
word = input()

def search(text, word):
    if word in text:
        print("Word found")
    else: 
        print("Word not found") 

search(text, word)


