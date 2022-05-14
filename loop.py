shopping = []

how_many = input("how many items of shopping do you have? ")
how_many = int(how_many)

for item_number in range(how_many):
    item = input("what is item number " + str(item_number) + "? ")
    shopping.append(item)

print(shopping)


while i in range(5):
    print(i)
    i = i + 1 


elif command == "average":
    how_many = input("How many numbers> ")
    how_many = int(how_many)
    total = 0
    for number_count in range(how_many):
        number = input("Enter number " + str(number_count) + "> ")
        total = total + int(number)
    result = total / how_many
    print("the average = " + str(result))


    #while loops

guess = input("guess my name ")
while guess != "Martin":
    guess = input("wrong - guess again ")
print("well done")


str = "testing for loops"
count = 0

for x in str:
  if(x == 't'):
    count += 1

print(count) 
#Usually we’d use the for loop when the number of iterations is fixed. For example, iterating over a fixed list of items in a shopping list.
#The while loop is useful in cases when the number of iterations isn’t known and depends on some calculations and conditions in the code block of the loop


x = [42, 8, 7, 1, 0, 124, 8897, 555, 3, 67, 99]
sum = 0
for x in x:
    sum += x
print(sum)

#If range is called with one argument, it’ll produce an object with values from 0 to that argument.

#If it’s called with two arguments, it’ll produce values from the first to the second.

#There’s also a third argument you can use with range(), and it’s really useful. It’s called Step and it determines the interval of the sequence produced.

#Want to go backward? No problem! We can also create a list of decreasing numbers, using a negative number as the third argument.
numbers = list(range(5, 20, 2))
print(numbers)



a = int(input())
b = int(input())
date = list(range(a,b))
print(date)
