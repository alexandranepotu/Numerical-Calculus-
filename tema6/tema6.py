#40% llm
import numpy as np
import matplotlib.pyplot as plt

#Functiile de test din enunt

def f1(x):
    # f(x) = x^4 - 12x^3 + 30x^2 + 12
    return x**4 - 12*x**3 + 30*x**2 + 12

def f1_deriv(x):
    # f'(x) = 4x^3 - 36x^2 + 60x
    return 4*x**3 - 36*x**2 + 60*x

def f2(x):
    # f(x) = x^3 + 3x^2 - 5x + 12
    return x**3 + 3*x**2 - 5*x + 12

def f2_deriv(x):
    # f'(x) = 3x^2 + 6x - 5
    return 3*x**2 + 6*x - 5


#Generarea nodurilor de interpolare

def genereaza_noduri(a, b, n, f):
    # Generam n-1 puncte aleatorii in (a, b)
    puncte_interioare = np.sort(np.random.uniform(a, b, n - 1))

    # Construim vectorul complet de noduri
    x = np.concatenate(([a], puncte_interioare, [b]))

    # Calculam valorile functiei in noduri
    y = np.array([f(xi) for xi in x])

    return x, y


# Schema lui Horner
def horner(coeficienti, x_val):
    # Initializam cu coeficientul de grad maxim
    d = coeficienti[0]

    # Parcurgem restul coeficientilor
    for i in range(1, len(coeficienti)):
        d = d * x_val + coeficienti[i]

    return d


#Metoda celor mai mici patrate (MCMMP)
def metoda_cmmmp(x, y, m, x_bar):
    n = len(x) - 1  # numarul de intervale (avem n+1 puncte)

    # Construim matricea B (sistemul normal)
    # B[i][j] = sum(x_k^(i+j), k=0..n)
    B = np.zeros((m + 1, m + 1))
    for i in range(m + 1):
        for j in range(m + 1):
            # Suma x_k^(i+j) pentru toate punctele
            B[i][j] = np.sum(x ** (i + j))

    # Construim vectorul f (partea dreapta)
    # f[i] = sum(y_k * x_k^i, k=0..n)
    f_vec = np.zeros(m + 1)
    for i in range(m + 1):
        f_vec[i] = np.sum(y * (x ** i))

    # Rezolvam sistemul B * a = f_vec
    # a = [a0, a1, ..., am] - coeficientii polinomului
    a = np.linalg.solve(B, f_vec)

    # Pregatim coeficientii pentru schema lui Horner
    # Horner asteapta [a_m, a_{m-1}, ..., a_1, a_0] (de la grad mare la mic)
    coef_horner = a[::-1]  # inversam ordinea

    # Calculam valoarea polinomului in x_bar cu Horner
    pm_x_bar = horner(coef_horner, x_bar)

    # Calculam suma |Pm(xi) - yi| pentru toate punctele
    suma_erori = 0
    for i in range(n + 1):
        pm_xi = horner(coef_horner, x[i])
        suma_erori += abs(pm_xi - y[i])

    return a, pm_x_bar, suma_erori

#Functii spline cubice de clasa C2

def spline_cubice(x, y, da, db, x_bar):
    n = len(x) - 1  # numarul de intervale

    # Calculam h_i = x_{i+1} - x_i
    h = np.zeros(n)
    for i in range(n):
        h[i] = x[i + 1] - x[i]

    # Construim matricea H (tridiagonala) de dimensiune (n+1) x (n+1)
    H = np.zeros((n + 1, n + 1))

    # Prima linie: 2*h0*A0 + h0*A1 = ...
    H[0][0] = 2 * h[0]
    H[0][1] = h[0]

    # Liniile din mijloc: h_{i-1}*A_{i-1} + 2*(h_{i-1}+h_i)*A_i + h_i*A_{i+1} = ...
    for i in range(1, n):
        H[i][i - 1] = h[i - 1]
        H[i][i] = 2 * (h[i - 1] + h[i])
        H[i][i + 1] = h[i]

    # Ultima linie: h_{n-1}*A_{n-1} + 2*h_{n-1}*A_n = ...
    H[n][n - 1] = h[n - 1]
    H[n][n] = 2 * h[n - 1]

    # Construim vectorul f (partea dreapta a sistemului)
    f_vec = np.zeros(n + 1)

    # Prima ecuatie: 6 * ((y1 - y0) / h0 - da)
    f_vec[0] = 6 * ((y[1] - y[0]) / h[0] - da)

    # Ecuatiile din mijloc
    for i in range(1, n):
        f_vec[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    # Ultima ecuatie: 6 * (db - (yn - y_{n-1}) / h_{n-1})
    f_vec[n] = 6 * (db - (y[n] - y[n - 1]) / h[n - 1])

    # Rezolvam sistemul H * A = f_vec
    A = np.linalg.solve(H, f_vec)

    # Gasim intervalul in care se afla x_bar
    # Cautam i0 astfel incat x_bar in [x_{i0}, x_{i0+1}]
    i0 = 0
    for i in range(n):
        if x[i] <= x_bar <= x[i + 1]:
            i0 = i
            break

    # Calculam coeficientii b_i si c_i pentru intervalul gasit
    bi = (y[i0 + 1] - y[i0]) / h[i0] - h[i0] * (A[i0 + 1] - A[i0]) / 6
    ci = (x[i0 + 1] * y[i0] - x[i0] * y[i0 + 1]) / h[i0] - h[i0] * (x[i0 + 1] * A[i0] - x[i0] * A[i0 + 1]) / 6

    # Calculam Sf(x_bar) folosind formula din enunt
    sf_val = ((x_bar - x[i0])**3 * A[i0 + 1]) / (6 * h[i0]) \
           + ((x[i0 + 1] - x_bar)**3 * A[i0]) / (6 * h[i0]) \
           + bi * x_bar + ci

    return sf_val, A, h


def evalueaza_spline(x, y, A, h, x_val):
    n = len(x) - 1

    # Gasim intervalul
    i0 = 0
    for i in range(n):
        if x[i] <= x_val <= x[i + 1]:
            i0 = i
            break

    # Calculam coeficientii
    bi = (y[i0 + 1] - y[i0]) / h[i0] - h[i0] * (A[i0 + 1] - A[i0]) / 6
    ci = (x[i0 + 1] * y[i0] - x[i0] * y[i0 + 1]) / h[i0] - h[i0] * (x[i0 + 1] * A[i0] - x[i0] * A[i0 + 1]) / 6

    # Calculam valoarea
    val = ((x_val - x[i0])**3 * A[i0 + 1]) / (6 * h[i0]) \
        + ((x[i0 + 1] - x_val)**3 * A[i0]) / (6 * h[i0]) \
        + bi * x_val + ci

    return val


#Functia principala care ruleaza totul

def rezolva(a, b, n, m, x_bar, f, f_deriv, da, db, titlu=""):
    print("=" * 60)
    print(f"  {titlu}")
    print("=" * 60)

    # Generam nodurile
    x, y = genereaza_noduri(a, b, n, f)

    print(f"\nNoduri de interpolare (n = {n}, deci {n+1} puncte):")
    for i in range(len(x)):
        print(f"  x[{i}] = {x[i]:.6f},  y[{i}] = {y[i]:.6f}")

    print(f"\nPunctul de aproximare: x_bar = {x_bar}")
    print(f"Valoarea exacta: f(x_bar) = {f(x_bar):.10f}")

    #1.Metoda celor mai mici patrate
    print(f"\n--- Metoda celor mai mici patrate (grad m = {m}) ---")

    coef, pm_val, suma_erori = metoda_cmmmp(x, y, m, x_bar)

    print(f"Coeficientii polinomului Pm:")
    for i in range(len(coef)):
        print(f"  a[{i}] = {coef[i]:.10f}")

    print(f"\nPm(x_bar) = {pm_val:.10f}")
    print(f"|Pm(x_bar) - f(x_bar)| = {abs(pm_val - f(x_bar)):.10e}")
    print(f"Suma |Pm(xi) - yi| = {suma_erori:.10e}")

    #2.Spline cubice
    print(f"\n--- Functii spline cubice de clasa C2 ---")
    print(f"da = f'(a) = {da},  db = f'(b) = {db}")

    sf_val, A_spline, h_spline = spline_cubice(x, y, da, db, x_bar)

    print(f"\nSf(x_bar) = {sf_val:.10f}")
    print(f"|Sf(x_bar) - f(x_bar)| = {abs(sf_val - f(x_bar)):.10e}")

    #3.Graficul
    # Generam puncte fine pentru desenare
    x_plot = np.linspace(a, b, 500)

    # Valorile functiei exacte
    y_exact = np.array([f(xi) for xi in x_plot])

    # Valorile polinomului Pm
    coef_horner = coef[::-1]  # inversam pentru Horner
    y_pm = np.array([horner(coef_horner, xi) for xi in x_plot])

    # Valorile spline-ului
    y_sf = np.array([evalueaza_spline(x, y, A_spline, h_spline, xi) for xi in x_plot])

    # Desenam graficul
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_exact, 'b-', linewidth=2, label='f(x) - functia exacta')
    plt.plot(x_plot, y_pm, 'r--', linewidth=1.5, label=f'Pm(x) - MCMMP (grad {m})')
    plt.plot(x_plot, y_sf, 'g-.', linewidth=1.5, label='Sf(x) - Spline cubic C2')
    plt.plot(x, y, 'ko', markersize=5, label='Noduri de interpolare')
    plt.plot(x_bar, f(x_bar), 'r*', markersize=12, label=f'x_bar = {x_bar}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(titlu)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'grafic_{titlu.replace(" ", "_")}.png', dpi=150)
    plt.show()

    print(f"\nGraficul a fost salvat.")
    print()


def rezolva_cu_tabel(x, y, m, x_bar, da, db, f_exact_xbar, titlu=""):
    print("=" * 60)
    print(f"  {titlu}")
    print("=" * 60)

    n = len(x) - 1

    print(f"\nNoduri de interpolare (n = {n}, deci {n+1} puncte):")
    for i in range(len(x)):
        print(f"  x[{i}] = {x[i]},  y[{i}] = {y[i]}")

    print(f"\nPunctul de aproximare: x_bar = {x_bar}")
    print(f"Valoarea exacta: f(x_bar) = {f_exact_xbar}")

    #1.Metoda celor mai mici patrate
    print(f"\n--- Metoda celor mai mici patrate (grad m = {m}) ---")

    coef, pm_val, suma_erori = metoda_cmmmp(x, y, m, x_bar)

    print(f"Coeficientii polinomului Pm:")
    for i in range(len(coef)):
        print(f"  a[{i}] = {coef[i]:.10f}")

    print(f"\nPm(x_bar) = {pm_val:.10f}")
    print(f"|Pm(x_bar) - f(x_bar)| = {abs(pm_val - f_exact_xbar):.10e}")
    print(f"Suma |Pm(xi) - yi| = {suma_erori:.10e}")

    #2.Spline cubice
    print(f"\n--- Functii spline cubice de clasa C2 ---")
    print(f"da = f'(a) = {da},  db = f'(b) = {db}")

    sf_val, A_spline, h_spline = spline_cubice(x, y, da, db, x_bar)

    print(f"\nSf(x_bar) = {sf_val:.10f}")
    print(f"|Sf(x_bar) - f(x_bar)| = {abs(sf_val - f_exact_xbar):.10e}")

    #3.Graficul
    a = x[0]
    b = x[-1]
    x_plot = np.linspace(a, b, 500)

    # Valorile polinomului Pm
    coef_horner = coef[::-1]
    y_pm = np.array([horner(coef_horner, xi) for xi in x_plot])

    # Valorile spline-ului
    y_sf = np.array([evalueaza_spline(x, y, A_spline, h_spline, xi) for xi in x_plot])

    # Desenam graficul
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_pm, 'r--', linewidth=1.5, label=f'Pm(x) - MCMMP (grad {m})')
    plt.plot(x_plot, y_sf, 'g-.', linewidth=1.5, label='Sf(x) - Spline cubic C2')
    plt.plot(x, y, 'ko', markersize=5, label='Noduri de interpolare')
    plt.plot(x_bar, f_exact_xbar, 'r*', markersize=12, label=f'x_bar = {x_bar}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(titlu)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'grafic_{titlu.replace(" ", "_")}.png', dpi=150)
    plt.show()

    print(f"\nGraficul a fost salvat.")
    print()

if __name__ == "__main__":

    # Setam seed-ul pentru reproducibilitate
    np.random.seed(42)

    # Numarul de noduri (n+1 puncte) si gradul polinomului m
    n = 10    # 11 puncte de interpolare
    m = 4     # gradul polinomului (m < 6, conform enuntului)

    #Exemplul 1
    # f(x) = x^4 - 12x^3 + 30x^2 + 12, a=0, b=2
    # da = f'(0) = 0, db = f'(2) = 8
    x_bar1 = 1.0  # punct de aproximare (ales de noi)
    rezolva(
        a=0, b=2, n=n, m=m,
        x_bar=x_bar1,
        f=f1, f_deriv=f1_deriv,
        da=0, db=8,
        titlu="Exemplul 1: f(x) = x^4 - 12x^3 + 30x^2 + 12"
    )

    #Exemplul 2
    # f(x) = x^3 + 3x^2 - 5x + 12, a=1, b=5
    # da = f'(1) = 4, db = f'(5) = 100
    x_bar2 = 3.0  # punct de aproximare
    rezolva(
        a=1, b=5, n=n, m=m,
        x_bar=x_bar2,
        f=f2, f_deriv=f2_deriv,
        da=4, db=100,
        titlu="Exemplul 2: f(x) = x^3 + 3x^2 - 5x + 12"
    )

    #Exemplul 3(din tabel)
    # Noduri date direct, x_bar = 1.5
    # Tabelul din enunt: x = 0,1,2,3,4,5, f = 50,47,-2,-121,-310,-545
    # Functia este f(x) = x^4 - 10x^3 + 6x + 50
    x3 = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    y3 = np.array([50.0, 47.0, -2.0, -121.0, -310.0, -545.0], dtype=float)
    da3 = 6.0
    db3 = -244.0
    x_bar3 = 1.5
    f_exact_xbar3 = 30.3125  # dat in enunt

    rezolva_cu_tabel(
        x=x3, y=y3, m=m,
        x_bar=x_bar3,
        da=da3, db=db3,
        f_exact_xbar=f_exact_xbar3,
        titlu="Exemplul 3: Date din tabel"
    )

    print("Program terminat cu succes!")
