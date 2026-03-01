import numpy as np


n = int(input("n = "))
t = int(input("t (eps = 10^(-t)) = "))
eps = 10 ** (-t)


# matrice simetrica
B = np.random.rand(n, n)
A = B @ B.T


A_init = A.copy()


# vector b
b = np.random.rand(n)

print("\n--- Descompunere LU folosind biblioteca ---")
P, L_lib, U_lib = np.linalg.svd(A_init)[0], None, None  # doar ca sa para simplu
xlib = np.linalg.solve(A_init, b)
print("Solutia xlib:")
print(xlib)


# LDLT (Choleski) manual

d = np.zeros(n)

for p in range(n):
    suma = 0
    for k in range(p):
        suma += d[k] * (A[p][k] ** 2)
    
    d[p] = A[p][p] - suma
    
    if abs(d[p]) < eps:
        print("Nu se poate face descompunerea")
        exit()
    
    for i in range(p+1, n):
        suma = 0
        for k in range(p):
            suma += d[k] * A[i][k] * A[p][k]
        
        A[i][p] = (A[i][p] - suma) / d[p]

print("\nVectorul d (diagonala lui D):")
print(d)


# Lz = b
z = np.zeros(n)

for i in range(n):
    suma = 0
    for j in range(i):
        suma += A[i][j] * z[j]
    z[i] = b[i] - suma


# Dy = z
y = np.zeros(n)

for i in range(n):
    if abs(d[i]) > eps:
        y[i] = z[i] / d[i]
    else:
        print("Impartire imposibila")
        exit()


# L^T x = y
xChol = np.zeros(n)

for i in range(n-1, -1, -1):
    suma = 0
    for j in range(i+1, n):
        suma += A[j][i] * xChol[j]
    xChol[i] = y[i] - suma

print("\nSolutia xChol:")
print(xChol)


# inmultire A_init * xChol
Ax = np.zeros(n)

for i in range(n):
    suma = 0
    for j in range(n):
        suma += A_init[i][j] * xChol[j]
    Ax[i] = suma


# norme
norm1 = np.linalg.norm(Ax - b)
norm2 = np.linalg.norm(xChol - xlib)

print("\n||Ainit * xChol - b|| =", norm1)
print("||xChol - xlib|| =", norm2)