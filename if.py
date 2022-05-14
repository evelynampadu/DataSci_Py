guys_age = 35
sims_age = 30

if sims_age > guys_age:
    print('He is older')

if sims_age < guys_age:
    print("He is not older")

if sims_age == guys_age:
    print('They are the same age')



you_washed = False

if you_washed:
    print(you_washed)
    print('You get rewarded for washing')
else:
    print('You didn\'t wash')



answer = input('Did you read the book (yes/no)')
answer = answer.lower()
if answer == 'yes' or answer =='y':
    print('Well done!')
elif answer == 'no' or answer == 'n':
    print('Why not?') 
elif answer == '':
    print('Kindly respond')
else:
    print('Answer yes or no')


    weight = float(input())
height = float(input())
BMI = weight/height**2
if BMI < 18.5:
    print("Underweight")
elif BMI >= 18.5 and BMI < 25:
    print("Normal")
elif BMI >= 25 and BMI < 30:
    print("Overweight")
else: print("Obesity")