#tuples are an ordered and the contents in the variables cannot be changed, unlike list.

attendees = ("Ahmed", "Evelyn", "Guy", "Temi")
print(type(attendees))  #to fund the data type
print(len(attendees)) #to find the length of 
#a string with a comma at the end becomes a tuple
#() - tuple
#[] - array
#{} - sets cannot have repeated values(keys) in them. Values(keys) must be unique. They are unordered and the positions of the values(keys) will keep changing if you keep running the set

#when costructors are used the normal brackets can be used
attendees = set(("Ahmed", "Evelyn", "Guy", "Temi"))
attenders = {"Kofi", "Yaw", "Kwame"}
attenders.update(attenders) #result will merge both sets into one

#dictionary - key value pairs. They are ordered and changeable but keys cannot be duplicated
dictionary = {
    "KeyOne": "This the the value of key one",
    "KeyTwo": "This is the value of key two",
    "KeyThree": "This is the value of key three"
}
print(dictionary)
print(dictionary["KeyOne"])
#or
x = dictionary.get("KeyOne")
print(x)

print(dictionary.keys())  #to list only keys
print(dictionary.values())

#changing the value of a key
dictionary["KeyOne"] = "Whatever you want to change it to"

#add a new key, just add
dictionary["KeyFour"] = "This is the value of key four"

#to remove a key
dictionary.pop("keyThree")