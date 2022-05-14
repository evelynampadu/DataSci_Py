market_list = ["yam", "moimoi", "rice", "lettuce"]
answer = input("What else do you want to buy?")
market_list.append(answer)
print(market_list)
#append add stuff to the end of the list
#insert adds stuff anywhere but you have to add the index of where you want to insert it

people = ["family", "spouse", "friends", "money"]
people.insert(0, "God")
print(people)
#this will insert God in the 0 index position
people.remove("Money")
print(people)

#or 
people.pop[0] #remove index 0
del people[1] #delete index 1

#List slices allow you to get a part of the list using two colon-separated indices. This returns a new list containing all the values between the indices.
squares = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
print(squares[2:6])
print(squares[3:8])
print(squares[0:1])


squares = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
print(squares[7:])  

squares = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
print(squares[:5])  

#Just like with ranges, your list slices can include a third number, representing the step, to include only alternate values in the slice.
squares = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
print(squares[::2])
print(squares[2:8:3])

#Using [::-1] as a slice is a common and idiomatic way to reverse a list.
nums = [5, 42, 7, 1, 0]
res = nums[::-1]
print(res)


N = int(input())
sum = 0
for i in range(N+1):
  sum += i
print(sum)