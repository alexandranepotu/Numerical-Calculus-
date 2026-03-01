# 15% LLM
import numpy as np
from scipy.linalg import lu

n = int(input("n = "))
t = int(input("t (eps = 10^(-t)) = "))
eps = 10**(-t)

B = np.random.rand(n, n)
A = B @ B.T 
# Pastram b-ul separat pentru a nu-l modifica
b = np.random.rand(n)

print("\n--- LU si solutia x_lib ---")
# Cerinta 1: Descompunere LU si solutie din biblioteca
P_lu, L_lu, U_lu = lu(A)
xlib = np.linalg.solve(A, b)
print("Solutia xlib calculata.")

print("\n--- Descompunere LDLT (Choleski) ---")
d = np.zeros(n)

for p in range(n):
    s = 0
    for k in range(p):
        s = s + d[k] * A[p][k] * A[p][k]

    d[p] = A[p][p] - s 

    if abs(d[p]) < eps:
        print("Eroare: d[p] prea mic")
        exit()

    for i in range(p+1, n):
        s2 = 0
        for k in range(p):
            s2 = s2 + d[k] * A[i][k] * A[p][k] 

        A[i][p] = (A[i][p] - s2) / d[p] 

# Determinantul: det A = det L * det D * det LT = det D
detA = 1
for i in range(n):
    detA = detA * d[i] 
print("detA =", detA)

print("\n--- Rezolvare sistem Ax=b ---")

# 1. Lz = b (substitutie directa, l_ii = 1)
z = np.zeros(n)
for i in range(n):
    s = 0
    for j in range(i):
        s = s + A[i][j] * z[j] 
    z[i] = b[i] - s 

# 2. Dy = z
y = np.zeros(n)
for i in range(n):
    y[i] = z[i] / d[i] 

# 3. LT x = y (substitutie inversa, l_ii = 1)
xChol = np.zeros(n)
for i in range(n-1, -1, -1):
    s = 0
    for j in range(i+1, n):
        s = s + A[j][i] * xChol[j] 
    xChol[i] = y[i] - s 

print("\n--- Verificare norme ---")

# Inmultirea A_init * xChol folosind simetria (fara a folosi alta matrice)
Ax = np.zeros(n)
for i in range(n):
    res = 0
    for j in range(n):
        if i <= j:
            val_originala = A[i][j] 
        else:
            val_originala = A[j][i]
        res += val_originala * xChol[j] 
    Ax[i] = res

norma1 = 0
for i in range(n):
    norma1 += (Ax[i] - b[i])**2 
norma1 = np.sqrt(norma1)

norma2 = 0
for i in range(n):
    norma2 += (xChol[i] - xlib[i])**2 
norma2 = np.sqrt(norma2)

print("||Ainit * xChol - b|| =", norma1) 
print("||xChol - xlib|| =", norma2) 