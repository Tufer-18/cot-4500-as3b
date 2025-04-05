import numpy as np

def gaussian_elimination(A, b): #Gaussian Elimination Function----------------------------------------------------------------------------------------------------------
    A = A.astype(float)
    b = b.astype(float)
    n = len(A)

    for i in range(n):
        max_row = i + np.argmax(abs(A[i:, i]))
        A[[i, max_row]] = A[[max_row, i]]
        b[[i, max_row]] = b[[max_row, i]]
        
        for j in range(i+1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    
   
    return x


def lu_factorization(A): #LU Factorization Function---------------------------------------------------------------------------------------------------------------------
    n = len(A)
    L = np.eye(n)
    U = A.astype(float)

    for i in range(n):
        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]
            U[j] -= factor * U[i]
            L[j, i] = factor

    determinant = np.prod(np.diag(U))
    return L, U, determinant

def is_diagonally_dominant(A): #Program to check for diagonal dominance------------------------------------------------------------------------------------------------
    for i in range(len(A)):
        row_sum = sum(abs(A[i, j]) for j in range(len(A)) if i != j)
        if abs(A[i, i]) < row_sum:
            
            return False
    
    return True

def is_positive_definite(A): #Function to check for positive definite.------------------------------------------------------------------------------------------------
    try:
        np.linalg.cholesky(A)
       
        return True
    except np.linalg.LinAlgError:
       
        return False


# Gaussian Elimination function test------------------------------------------------------------------------------------------------------------------------------
A = np.array([[val,val,val...], 
              [val,val,val...], 
              [val,val,val...]], dtype=float) #Enter rows and values for Matrix A here.
b = np.array([val, val, val...], dtype=float) # Enter values for Matrix B here.
print(gaussian_elimination(A, b))

# LU Factorization Function test----------------------------------------------------------------------------------------------------------------------------------
A = np.array([[val,val,val,val...], 
              [val,val,val,val...], 
              [val,val,val,val...], 
              [val,val,val,val...]], dtype=float) #Enter values and rows for matrix here.
L, U, det_A = lu_factorization(A)
print("\n", det_A)
print("\n", L)
print("\n", U)
print("\n")    

# Diagonal Dominant test---------------------------------------------------------------------------------------------------------------------------------------------
A = np.array([
    [val,val,val,val,val...],
    [val,val,val,val,val...],
    [val,val,val,val,val...],
    [val,val,val,val,val...],
    [val,val,val,val,val...]
    ], dtype=float) # Enter rows and values for matrix here.
print("Diagonally Dominant:", is_diagonally_dominant(A))
print("\n")


# Positive Definite test-----------------------------------------------------------------------------------------------------------------------------------------------
A = np.array([
     [2, 2, 1],
     [2, 3, 0],
     [1, 0, 2]
    ], dtype=float) #Enter values and rows for matrix here.
print("Positive Definite:", is_positive_definite(A))
