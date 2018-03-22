import csv #Importing the built-in csv module
import numpy as np #Importing the numPy module.This module is used for scientific computation in python.
# This also has a n-dimensional array object,tools for integrating C,C++.Useful linear algebra and number capabilities
with open('/Users/bonythomas/FisherIris_MDL.csv', 'rb') as Iris: # Opening the csv file from its source location
    reader = csv.reader(Iris, delimiter=",") #The reader object of the csv module reads the csv file line by line.
    data_as_list = list(reader) #The list funtion converts the file elements into a list and this is stored in the data_as_list variable
Iris=np.int32(data_as_list) #Converting each list items to integer
print ("IRIS Matrix",Iris) #Printing th Iris matrix
a=np.shape(Iris) #Shape function of the numpy module returns the shape of the array as in rows and columns
print("No. Of Rows:",a[0]) #Returns the number of rows of Iris
print("No. Of Columns",a[1]) #Returns the number of columns of Iris
for i in range(a[0]):
    if Iris[i][4]<0: #Checks if the 5th col elements are lesser than 0.
        print ("Row number with 5th element as a negative value:",i+1 )#Prints the row number with 5th element value lesser than 0
Iris=Iris[~(Iris[:,4]<0), :] #Negates the Iris Matrix of all rows that have 5th column values lesser tha 0.
a=np.shape(Iris)
print("No. Of Rows:",a[0])
print("No. Of Columns",a[1])
x = Iris[:, [0,1,2,3]]#Copies first 4 columns from Iris to a new matrix X.
b=np.shape(x)
y=Iris[:, 4] #Copying 5th colum values of Iris into Y variable
print("New Variable Y", y)
print("New Matrix X", x)
print ("Matrix X First Col:Max" ,x[:,0].max() )#Max funtion in numpy calculates the max val of the row/col given,in this case of col1
print ("Matrix X Second Col:Max" ,x[:,1].max())#Prints the max value of col2 of X Matrix
print ("Matrix X Third Col:Max" ,x[:,2].max())#Prints the max value of col3 of X Matrix
print ("Matrix X Fourth Col:Max" ,x[:,3].max())#Prints the max value of col4 of X Matrix
print ("Matrix X First Col:Max" ,x[:,0].min())#Prints the min value of col1 of X Matrix
print ("Matrix X Second Col:Max" ,x[:,1].min())#Prints the min value of col2 of X Matrix
print ("Matrix X Third Col:Max" ,x[:,2].min())#Prints the min value of col3 of X Matrix
print ("Matrix X Fourth Col:Max" ,x[:,3].min())#Prints the min value of col4 of X Matrix
c=0
for i in range(b[0]):
    if x[i][2]>36: #Checks id the 3rd column elements are greater than 36
     c=c+1
print("No. of elements in third column greater than 36:",c)

