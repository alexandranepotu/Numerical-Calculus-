# 20% LLM
import numpy as np

n = 4
eps = 1e-12

A_init = np.random.rand(n, n)
s = np.random.rand(n)

#  b = A * s 
b_init = np.zeros(n)
for i in range(n):
    for j in range(n):
        b_init[i] += A_init[i][j] * s[j]

# descompunerea QR Householder 
R = A_init.copy()
Qt = np.eye(n) 
b_house = b_init.copy()

for r in range(n - 1):
    sigma = 0
    for i in range(r, n):
        sigma += R[i][r]**2 
        
    if sigma <= eps: 
        break

    k = np.sqrt(sigma) 
    if R[r][r] > 0: k = -k 
    
    beta = sigma - k * R[r][r] 
    u = np.zeros(n)
    u[r] = R[r][r] - k 
    for i in range(r + 1, n):
        u[i] = R[i][r] 
    
    # R 
    for j in range(r + 1, n):
        gamma = 0
        for i in range(r, n):
            gamma += u[i] * R[i][j]
        gamma /= beta
        for i in range(r, n):
            R[i][j] -= gamma * u[i]
    
    # coloana r din R 
    R[r, r] = k
    for i in range(r + 1, n):
        R[i][r] = 0
    
    # b 
    gamma_b = 0
    for i in range(r, n):
        gamma_b += u[i] * b_house[i]
    gamma_b /= beta
    for i in range(r, n):
        b_house[i] -= gamma_b * u[i]
    
    # Qt
    for j in range(n):
        gamma_q = 0
        for i in range(r, n):
            gamma_q += u[i] * Qt[i][j]
        gamma_q /= beta
        for i in range(r, n):
            Qt[i][j] -= gamma_q * u[i]

# rezolvare sistem 
x_house = np.zeros(n)
for i in range(n - 1, -1, -1):
    suma = 0
    for j in range(i + 1, n):
        suma += R[i][j] * x_house[j]
    x_house[i] = (b_house[i] - suma) / R[i][i]

# comparatie cu biblioteca 
Q_lib, R_lib = np.linalg.qr(A_init)
x_qr = np.linalg.solve(A_init, b_init)

print("Norma diferenta solutii:", np.linalg.norm(x_qr - x_house)) 

# erori 
err1 = np.linalg.norm(A_init @ x_house - b_init)
err2 = np.linalg.norm(A_init @ x_qr - b_init)
err3 = np.linalg.norm(x_house - s) / np.linalg.norm(s)
err4 = np.linalg.norm(x_qr - s) / np.linalg.norm(s)

print(f"Eroare Householder: {err1}")
print(f"Eroare QR bibliot.: {err2}")
print(f"Eroare relativa Householder: {err3}")
print(f"Eroare relativa QR bibliot.: {err4}")

# calcul inversa 
A_inv_house = np.zeros((n, n))
for j in range(n):
    # b = Qt * ej
    bj = Qt[:, j].copy()
    xj = np.zeros(n)
    for i in range(n - 1, -1, -1):
        suma = 0
        for k_idx in range(i + 1, n):
            suma += R[i][k_idx] * xj[k_idx]
        xj[i] = (bj[i] - suma) / R[i][i]
    A_inv_house[:, j] = xj 
A_inv_lib = np.linalg.inv(A_init)
print("Norma diferenta inverse:", np.linalg.norm(A_inv_house - A_inv_lib)) 