#1. Split this string
s = "Hi there Hari!"
x = s.split()
print(x)

#2. Use .format() to print the following string.
#Output should be: The diameter of Earth is 12742 kilometers.
planet = "Earth"
diameter = 12742
x="The diameter of {planet} is {diameter} kilometers."
print(x.format(planet="Earth",diameter=12742))

#3. In this nest dictionary grab the word "hello"
d = {'k1':[1,2,3,{'tricky':['oh','man','inception',{'target':[1,2,3,'hello']}]}]}
d['k1'][3]['tricky'][3]['target'][3]

#Numpy
import numpy as np

#4.1 Create an array of 10 zeros?
array10=np.zeros(10)
print("Array of 10 Zeros:",array10)

#4.2 Create an array of 10 fives?
array5=np.ones(10)*5
print("Array of 10 fives:",array5)

#5. Create an array of all the even integers from 20 to 35
arrayeven=np.arange(20,35,2)
print("Array of all the even integers from 20 to 35:",arrayeven)

#6. Create a 3x3 matrix with values ranging from 0 to 8
x = np.arange(0,9).reshape(3,3)
print(x)

#7. Concatinate a and b
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
ab = np.concatenate((a,b))
print(ab)

#Pandas
import pandas as pd

#8. Create a dataframe with 3 rows and 2 columns
data = {'Name':['Tom','Arun','Krish'],'Age':[20,21,19]}
df=pd.DataFrame(data)
print(df)

#9. Generate the series of dates from 1st Jan, 2023 to 10th Feb, 2023
pd.date_range(start='1/1/2023', end='2/10/2023')

#10. Create 2D list to DataFrame
lists = [[1, 'aaa', 22], [2, 'bbb', 25], [3, 'ccc', 24]]
df = pd.DataFrame(lists)
print(df)
