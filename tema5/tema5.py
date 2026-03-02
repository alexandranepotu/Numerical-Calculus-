import numpy as np

# --- 1. PARAMETRI INITIALI ---
eps = 1e-6
k_max = 100

# Matrice simetrica (Pag. 10)
A = np.array([[1.0, 1.0, 2.0], 
              [1.0, 1.0, 2.0], 
              [2.0, 2.0, 2.0]])
A_init = A.copy()
n = len(A)
U = np.eye(n)

# --- 2. METODA JACOBI (Pag. 3-7) ---
for k in range(k_max):
    # Gasim cel mai mare element nediagonal (Pag. 4) 
    p, q = 1, 0
    maxim = abs(A[1, 0])
    for i in range(n):
        for j in range(i):
            if abs(A[i, j]) > maxim:
                maxim = abs(A[i, j])
                p, q = i, j
    
    if maxim < eps: # Test oprire (Pag. 7) 
        break
        
    # Calcul unghi (Pag. 6) 
    alpha = (A[p, p] - A[q, q]) / (2 * A[p, q]) 
    t = -alpha + (1 if alpha >= 0 else -1) * np.sqrt(alpha**2 + 1) # t de modul minim
    c = 1 / np.sqrt(1 + t**2)
    s = t / np.sqrt(1 + t**2)
    
    # Actualizare matrice A (Pag. 7)
    for j in range(n):
        if j != p and j != q:
            A_pj = A[p, j]
            A[p, j] = A[j, p] = c * A_pj + s * A[q, j] 
            A[q, j] = A[j, q] = -s * A_pj + c * A[q, j] 
            
    A[p, p] = A[p, p] + t * A[p, q] 
    A[q, q] = A[q, q] - t * A[p, q] 
    A[p, q] = A[q, p] = 0 
    
    # Actualizare vectori proprii (Pag. 8) 
    for i in range(n):
        u_ip = U[i, p]
        U[i, p] = c * u_ip + s * U[i, q] 
        U[i, q] = -s * u_ip + c * U[i, q] 

print("Valori proprii (Jacobi):", np.diag(A))
print("Norma verificare A*U - U*L:", np.linalg.norm(np.dot(A_init, U) - np.dot(U, np.diag(np.diag(A))))) 

# --- 3. CHOLESKY ITERATIV (Pag. 1) ---
# Folosim o matrice care sa nu dea eroare (pozitiv definita)
A_chol = np.array([[4.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 2.0]])
for i in range(k_max):
    L = np.linalg.cholesky(A_chol) # Factorizarea 
    A_chol = np.dot(L.T, L) # Inmultirea (L^T * L) 

print("\nMatricea Cholesky (tinde la forma diagonala):") 
print(A_chol)

# --- 4. SVD p > n (Pag. 2, 8, 9) ---
# Matrice cu coloane independente pentru a putea calcula A_J
A_rect = np.array([[1, 2], [3, 4], [5, 2], [1, 7], [0, 1]], dtype=float)

u, s_val, vt = np.linalg.svd(A_rect) # 
print("\nValori singulare:", s_val) 
print("Rang matrice:", np.count_nonzero(s_val > 1e-10)) 
print("Nr conditiunare:", max(s_val) / min(s_val[s_val > 0])) 

# Pseudoinversa Moore-Penrose A^I (SVD) 
S_inv = np.zeros((A_rect.shape[1], A_rect.shape[0]))
for i in range(len(s_val)):
    S_inv[i, i] = 1/s_val[i]
A_I = np.dot(vt.T, np.dot(S_inv, u.T))

# Pseudoinversa prin metoda celor mai mici patrate A^J 
A_J = np.dot(np.linalg.inv(np.dot(A_rect.T, A_rect)), A_rect.T)

print("Norma ||A^I - A^J||_1:", np.linalg.norm(A_I - A_J, 1)) 