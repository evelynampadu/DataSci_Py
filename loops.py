


shouts = 6
i=1
while i <= shouts:
    print(f'{i}. Hallelujah')
    i=i+1




name = ["Evelyn", "Ama", "Larry", "Kaylee"]
a = "Evelyn"
i = 0

while i <= len(a):
    print(a)
    i=i + 1



name = ["Evelyn", "Ama", "Larry", "Kaylee"]
i = 0

while i < len(name):
    a = name[i]
    print(a)
    i=i + 1



#nested loops - for loop
cars = ["Kia", "Hyundai", "Toyota", "Nissan", "Tata", "BMW"]
for i in range(len(cars)):
    print(i)
    for b in cars:
        print(b)
#the i runs through all of the b list before going on to the next i


taste = ["sweet", "nice", "delicious"]
food = ["yam", "rice", "stew"]
for x in taste:
    for y in food:
        print(x, y)
#first of x will run through all of y before, second of x will run through all of y....

i = 0
while i<5:
  i += 1
  if i==3:
    print("Skipping 3")
    continue
  print(i)
#the continue statement stops the current iteration and continues with the next one.


i=0
while True:
    i+=1
    if(i == 2):
        continue
    if(i == 5):
        break
    
    print(i)


    ticket = 100
total = 0
passengers = 1
while passengers <= 5:
   age = int(input()) # in the loop
   passengers += 1 # always increment
   if age < 3:
       continue
   else:
       total += ticket
   # passengers += 1
print(total)