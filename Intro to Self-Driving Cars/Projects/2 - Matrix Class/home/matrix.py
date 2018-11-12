import math
from math import sqrt
import numbers

def zeroes(height, width):
        """
        Creates a matrix of zeroes.
        """
        g = [[0.0 for _ in range(width)] for __ in range(height)]
        return Matrix(g)


def identity(n):
        """
        Creates a n x n identity matrix.
        """
        I = zeroes(n, n)
        for i in range(n):
            I.g[i][i] = 1.0
        return I


class Matrix(object):

    # Constructor
    def __init__(self, grid):
        self.g = grid
        self.h = len(grid)
        self.w = len(grid[0])

    #
    # Primary matrix math methods
    #############################
 
    def determinant(self):
        """
        Calculates the determinant of a 1x1 or 2x2 matrix.
        """
        # validation
        if not self.is_square():
            raise(ValueError, "Cannot calculate determinant of non-square matrix.")
        if self.h > 2:
            raise(NotImplementedError, "Calculating determinant not implemented for matrices largerer than 2x2.")

        if self.h == 1:
            # for 1x1 matrices the determinant is just the value of the matrice's only element
             result = self.g[0][0]
        else:
            # for 2x2 matrices the determinant is given by:
            #             |A| = [[a  b],  = ad - bc
            #                    [c  d]]
            result = (self.g[0][0] * self.g[1][1]) - (self.g[0][1] * self.g[1][0])

        return result


    def trace(self):
        """
        Calculates the trace of a matrix (sum of diagonal entries).
        
        The trace of an 𝑛×𝑛 square matrix 𝐀 is the sum of the elements on the main diagonal of the matrix.
        """
        # validation
        if not self.is_square():
            raise(ValueError, "Cannot calculate the trace of a non-square matrix.")

        result = 0
        for i in range(self.h):
            result = result + self.g[i][i]

        return result


    def inverse(self):
        """
        Calculates the inverse of a 1x1 or 2x2 Matrix.
        """
        # validation
        if not self.is_square():
            raise(ValueError, "Non-square Matrix does not have an inverse.")
        if self.h > 2:
            raise(NotImplementedError, "inversion not implemented for matrices larger than 2x2.")

        result = []

        if self.h == 1:
            # for a 1×1 matrix with a single element with value 'a', the inverse is simply '1/a'
            result.append([1/self.g[0][0]])
            return Matrix(result)
        else:
            # the inverse of a  2×2  matrix is given by the following equation:
            # 𝐀^(-1) = 1/det(𝐀) [(tr 𝐀)𝐈 − 𝐀]
            det = self.determinant()
            if det == 0:
                raise ValueError('The matrix is not invertible')

            result = (1/det) * ((self.trace() * identity(2)) - self)
            return result


    def T(self):
        """
        Returns a transposed copy of this Matrix.
        """
        result = []

        for i in range(self.w):
            m = [x[i] for x in self.g]
            result.append(m)

        return Matrix(result)


    def is_square(self):
        """
        Returns if the matrix is square.
        """
        return self.h == self.w


    def get_row(self, row):
        """
        Returns selected row in the form of a list.
        
        Inputs:
            row _ row number to be returned starting from 0.
        """
        # validation
        if row >= self.h:
            raise(ValueError, "Row number does not exist.")

        return self.g[row]


    def get_column(self, column_number):
        """
        Returns selected column in the form of a list.
        
        Inputs:
            column_number _ column number to be returned starting from 0. 
        """
        if column_number >= self.w:
            raise(ValueError, "Column number does not exist.")
            
        return [x[column_number] for x in self.g]
    
    
    def dot_product(self, vector_one, vector_two):
        """
        Return the dot product of two vectors.
        """
        # validation
        if len(vector_one) != len(vector_two):
            raise(ValueError, "Vectors must have same size")

        return sum([a*b for a,b in zip(vector_one, vector_two)])


    #
    # Begin Operator Overloading
    ############################
    def __getitem__(self,idx):
        """
        Defines the behavior of using square brackets [] on instances
        of this class.

        Example:

        > my_matrix = Matrix([ [1, 2], [3, 4] ])
        > my_matrix[0]
          [1, 2]

        > my_matrix[0][0]
          1
        """
        return self.g[idx]


    def __repr__(self):
        """
        Defines the behavior of calling print on an instance of this class.
        """
        s = ""
        for row in self.g:
            s += " ".join(["{} ".format(x) for x in row])
            s += "\n"
        return s


    def __add__(self, other):
        """
        Defines the behavior of the + operator.
        
        Inputs:
            other _ matrix to be added to the self object.
        """
        # validation
        if self.h != other.h or self.w != other.w:
            raise(ValueError, "Matrices can only be added if the dimensions are the same") 
        
        # initialize auxiliar matrices
        matrixSum = []
        row = []

        for i in range(self.h):
            for j in range(self.w):
                row.append(self.g[i][j] + other[i][j])
            matrixSum.append(row)
            
            # clean row for next row interaction
            row = [] 
        
        return Matrix(matrixSum)
    
    
    def __neg__(self):
        """
        Defines the behavior of - operator (NOT subtraction).

        Example:

        > my_matrix = Matrix([ [1, 2], [3, 4] ])
        > negative  = -my_matrix
        > print(negative)
          -1.0  -2.0
          -3.0  -4.0
        """
        return Matrix([[-col for col in row] for row in self.g])
    

    def __sub__(self, other):
        """
        Defines the behavior of - operator (as subtraction).
        """
        return self + (-other)


    def __mul__(self, other):
        """
        Defines the behavior of * operator (matrix multiplication).
        """
        # validation
        if self.w != other.h:
            raise(ValueError, "Not possible to multiple matrices")

        # initialize auxiliar matrices
        result = []
        row_result = []
        
        for i in range(self.h):
            a = self.get_row(i)
            for j in range(other.w):  
                b = other.get_column(j)
                dot_p = self.dot_product(a, b)
                row_result.append(dot_p)
            
            # append calculated row and clean it
            result.append(row_result)
            row_result = []            
        
        return Matrix(result)


    def __rmul__(self, other):
        """
        Called when the thing on the left of the * is not a matrix.
        Return None if multiplier is not a number.
        
        Example:

        > identity = Matrix([ [1,0], [0,1] ])
        > doubled  = 2 * identity
        > print(doubled)
          2.0  0.0
          0.0  2.0
        """                        
        if isinstance(other, numbers.Number):
            return Matrix([[other * col for col in row] for row in self.g])
        else:
            raise(ValueError, "Multipler is not a number")
