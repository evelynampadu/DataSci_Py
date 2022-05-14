#generate a random number
#ask for number guessed
#compare guessed number to generated number
#if correct, you win. If not, try again




import random
n = random.randint(0,10)
def game():
    guess = int(input("Guess a number:" ))
    if guess > 10:
        print("Number must be less than 10")
    if guess == n:
        print("you win")
    else: print("try again")  
game()    
print("The random number was: " + str(n)) 
