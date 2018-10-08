import numpy as np

# Make the array `my_array`
my_array = np.array([[1,2,3,4], [5,6,7,8]], dtype=np.int64)

# Print `my_array`
print'\r\nInit array'
print my_array
print my_array[0:,0]


# Print shape
print(my_array.shape)

# Create an array of ones
np.ones((3,4))

# Create an array of zeros
np.zeros((2,3,4),dtype=np.int16)

# Create an array with random values
np.random.random((2,2))

# Create an empty array
np.empty((3,2))

# Create a full array
print '\r\nFull'
print np.full((2,2),7)


# Create an array of evenly-spaced values
np.arange(10,25,5)

# Create an array of evenly-spaced values
myarray = np.linspace(0,2,9)
print '\r\nLinspace'
print (myarray)

mylist = [[1,2], [3,4], [5,6]]
myarray = np.array(mylist)
print '***'
print myarray

x = myarray[0:,0]
y = myarray[0:,1]
print 'x: ', x
print 'y: ', y