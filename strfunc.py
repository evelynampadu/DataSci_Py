#The append() function is used to add an item to the end of the list
#insert() inserts a new item at the given position in the list
words = ["Python", "fun"]
words.insert(1, "is")
print(words) 

#index() finds the first occurrence of a list item and returns its index
letters = ['p', 'q', 'r', 's', 'p', 'u']
print(letters.index('r'))
print(letters.index('p'))
print(letters.index('q')) 

#max(list): Returns the maximum value.
#min(list): Returns the minimum value.


#list.count(item): Returns a count of how many times an item occurs in a list.
#list.remove(item): Removes an item from a list.
#list.reverse(): Reverses items in a list.
x = [2, 4, 6, 2, 7, 2, 9]
print(x.count(2))

x.remove(4)
print(x)

x.reverse()
print(x)

queue = ['John', 'Amy', 'Bob', 'Adam']
next = input()
queue.append(next)
print(queue)



#Strings have a format() function, which enables values to be embedded in it, using placeholders.
nums = [4, 5, 6]
msg = "Numbers: {0} {1} {2}". format(nums[0], nums[1], nums[2])
print(msg)


print("{0}{1}{0}".format("abra", "cad"))

a = "{x}, {y}".format(x=5, y=12)
print(a)

#join() joins a list of strings with another string as a separator.
x = ", ".join(["spam", "eggs", "ham"])
print(x)
#prints "spam, eggs, ham"

#split() is the opposite of join(). It turns a string with a certain separator into a list.
str = "some text goes here"
x = str.split(' ')
print(x)

#replace() replaces one substring in a string with another.
x = "Hello ME"
print(x.replace("ME", "world")) 

#lower() and upper() change the case of a string to lowercase and uppercase.
print("This is a sentence.".upper())
print("AN ALL CAPS SENTENCE".lower())

