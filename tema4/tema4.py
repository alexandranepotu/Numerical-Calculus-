# 15 LLM
import math

def rezolva_tema(index_fisier, p_precizie):
    epsilon = 10**(-p_precizie)
    
    # 1. Citirea datelor (presupunem ca fisierele sunt in acelasi folder)
    def citeste_vector(nume_fisier):
        v = []
        try:
            with open(nume_fisier, 'r') as f:
                for linie in f:
                    if linie.strip():
                        v.append(float(linie.strip()))
        except FileNotFoundError:
            return None
        return v

    d0 = citeste_vector(f"d0_{index_fisier}.txt")
    d1 = citeste_vector(f"d1_{index_fisier}.txt")
    d2 = citeste_vector(f"d2_{index_fisier}.txt")
    b = citeste_vector(f"b_{index_fisier}.txt")

    if not d0 or not b:
        print(f"Eroare: Nu s-au putut incarca fisierele pentru setul {index_fisier}")
        return

    n = len(b)
    print(f"--- Sistemul {index_fisier} ---")
    print(f"1. Dimensiunea sistemului n = {n}") 

    # 2. Determinarea ordinelor p si q pentru d1 si d2
    # Din formula: len(d1) = x + 1, unde x = n - 1 - p => p = n - len(d1)
    p_ord = n - len(d1)
    q_ord = n - len(d2)
    print(f"2. Ordinul diagonalei d1 (p): {p_ord}, Ordinul diagonalei d2 (q): {q_ord}") 

    # 3. Verificare diagonala principala sa nu aiba zerouri
    toate_nenule = True
    for elem in d0:
        if abs(elem) < epsilon:
            toate_nenule = False
            break
    
    if not toate_nenule:
        print("3. Eroare: Exista elemente nule pe diagonala principala!") 
        return
    else:
        print("3. Toate elementele din d0 sunt nenule.") 

    # 4. Metoda Gauss-Seidel
    xc = [0.0] * n # x current 
    xp = [0.0] * n # x precedent
    
    k = 0
    k_max = 10000
    convergent = False

    while True:
        # Copiem xc in xp pentru a calcula diferenta ulterior
        for i in range(n):
            xp[i] = xc[i]

        # Calculam noul xc[i] folosind formula (2) adaptata
        for i in range(n):
            suma = 0.0
            
            # Verificam vecinii pe baza diagonalelor p_ord si q_ord (Simetrie!)
            # Diagonala p (superior si inferior)
            # Elementul a[i][i-p_ord] (inferior) si a[i][i+p_ord] (superior)
            
            # Verificam diagonala p_ord
            if i - p_ord >= 0:
                # elementul este d1[i - p_ord]
                suma += d1[i - p_ord] * xc[i - p_ord]
            if i + p_ord < n:
                # elementul este d1[i]
                suma += d1[i] * xp[i + p_ord]

            # Verificam diagonala q_ord
            if i - q_ord >= 0:
                # elementul este d2[i - q_ord]
                suma += d2[i - q_ord] * xc[i - q_ord]
            if i + q_ord < n:
                # elementul este d2[i]
                suma += d2[i] * xp[i + q_ord]

            xc[i] = (b[i] - suma) / d0[i] 

        # Calculam delta_x (norma euclidiana sau infinit?) Cerinta zice ||xc - xp||
        delta_x = 0.0
        for i in range(n):
            delta_x += (xc[i] - xp[i])**2
        delta_x = math.sqrt(delta_x)

        k += 1
        
        if delta_x < epsilon:
            convergent = True
            break
        if k > k_max or delta_x > 10**10: 
            break

    if convergent:
        print(f"4. Solutie gasita in {k} iteratii.")
        
        # 5. Calcul y = A * x_GS (o singura parcurgere)
        y = [0.0] * n
        for i in range(n):
            # Diagonala principala
            val_y = d0[i] * xc[i]
            # Diagonalele secundare
            if i + p_ord < n:
                val_y += d1[i] * xc[i + p_ord]
            if i - p_ord >= 0:
                val_y += d1[i - p_ord] * xc[i - p_ord]
            if i + q_ord < n:
                val_y += d2[i] * xc[i + q_ord]
            if i - q_ord >= 0:
                val_y += d2[i - q_ord] * xc[i - q_ord]
            y[i] = val_y 

        # 6. Norma infinit ||Ax - b||
        norma_inf = 0.0
        for i in range(n):
            dif = abs(y[i] - b[i])
            if dif > norma_inf:
                norma_inf = dif
        print(f"6. Norma ||Ax_GS - b||_inf: {norma_inf}") 
    else:
        print("4. Metoda Gauss-Seidel a divergent.") 

# Rulare pentru sistemul 1 cu precizie 10^-8
rezolva_tema(1, 8)