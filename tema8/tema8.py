#50% llm
import numpy as np

# Definirea functiilor si a gradientilor analitici
def sigma(z):
    """Functia sigmoid"""
    return 1.0 / (1.0 + np.exp(-z))


#Functia 1: l(w0,w1) = -ln(1-sigma(w0-w1))-ln(sigma(w0+w1)) 
def f1(x):
    w0, w1 = x[0], x[1]
    # Protectie numerica pentru logaritm
    s1 = sigma(w0 - w1)
    s2 = sigma(w0 + w1)
    val1 = np.clip(1 - s1, 1e-15, 1 - 1e-15)
    val2 = np.clip(s2, 1e-15, 1 - 1e-15)
    return -np.log(val1) - np.log(val2)

def grad_f1(x):
    w0, w1 = x[0], x[1]
    s1 = sigma(w0 - w1)
    s2 = sigma(w0 + w1)
    df_dw0 = s1 + s2 - 1
    df_dw1 = s2 - s1 - 1
    return np.array([df_dw0, df_dw1])


#Functia 2: F(x1,x2) = x1^2+x2^2-2*x1-4*x2-1 
# x* = (1,2)
def f2(x):
    x1, x2 = x[0], x[1]
    return x1**2 + x2**2 - 2*x1 - 4*x2 - 1

def grad_f2(x):
    x1, x2 = x[0], x[1]
    return np.array([2*x1 - 2, 2*x2 - 4])


#Functia 3: F(x1,x2) = 3*x1^2-12*x1+2*x2^2+16*x2-10
# x* = (2,-4)
def f3(x):
    x1, x2 = x[0], x[1]
    return 3*x1**2 - 12*x1 + 2*x2**2 + 16*x2 - 10

def grad_f3(x):
    x1, x2 = x[0], x[1]
    return np.array([6*x1 - 12, 4*x2 + 16])


#Functia 4: F(x1,x2) = x1^2-4*x1*x2+4.5*x2^2-4*x2+3 
# x* = (8,4)
def f4(x):
    x1, x2 = x[0], x[1]
    return x1**2 - 4*x1*x2 + 4.5*x2**2 - 4*x2 + 3

def grad_f4(x):
    x1, x2 = x[0], x[1]
    return np.array([2*x1 - 4*x2, -4*x1 + 9*x2 - 4])


#Functia 5: F(x1,x2) = x1^2*x2-2*x1*x2^2+3*x1*x2+4
# x* = (-1, 0.5)
def f5(x):
    x1, x2 = x[0], x[1]
    return x1**2 * x2 - 2*x1 * x2**2 + 3*x1*x2 + 4

def grad_f5(x):
    x1, x2 = x[0], x[1]
    return np.array([2*x1*x2 - 2*x2**2 + 3*x2,
                     x1**2 - 4*x1*x2 + 3*x1])

# Gradientul aproximativ (formula cu diferente finite de ordin 4)
def gradient_aproximativ(F, x, h=1e-6):
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        # Vectorii cu perturbatii pe componenta i
        x_plus_2h = x.copy()
        x_plus_h = x.copy()
        x_minus_h = x.copy()
        x_minus_2h = x.copy()

        x_plus_2h[i] += 2 * h
        x_plus_h[i] += h
        x_minus_h[i] -= h
        x_minus_2h[i] -= 2 * h

        F1 = F(x_plus_2h)
        F2 = F(x_plus_h)
        F3 = F(x_minus_h)
        F4 = F(x_minus_2h)

        grad[i] = (-F1 + 8*F2 - 8*F3 + F4) / (12 * h)
    return grad

# Strategii de calcul a ratei de invatare

def rata_constanta(F, x, grad_val, eta_val=1e-3):
    return eta_val

def rata_backtracking(F, x, grad_val, beta=0.8):
    eta = 1.0
    p = 1
    norm_grad_sq = np.dot(grad_val, grad_val)
    while p < 8:
        x_nou = x - eta * grad_val
        if F(x_nou) <= F(x) - (eta / 2.0) * norm_grad_sq:
            break
        eta = eta * beta
        p += 1
    return eta


# Metoda gradientului descendent
def gradient_descendent(F, grad_func, x0, metoda_eta='constanta', epsilon=1e-5, kmax=30000, eta_val=1e-3, beta=0.8):
    x = x0.copy()
    k = 0

    while True:
        # Calculeaza gradientul
        g = grad_func(x)

        # Calculeaza rata de invatare
        if metoda_eta == 'constanta':
            eta = rata_constanta(F, x, g, eta_val)
        else:
            eta = rata_backtracking(F, x, g, beta)

        norm_test = eta * np.linalg.norm(g)

        # Verificam conditia de oprire
        if norm_test < epsilon:
            return x, k, True

        if k >= kmax:
            # Am depasit numarul maxim de iteratii
            return x, k, False

        if norm_test > 1e10:
            # Divergenta
            return x, k, False

        # Actualizam x
        x = x - eta * g
        k += 1

# Functie wrapper pentru gradient aproximativ
def make_grad_aprox(F, h=1e-6):
    def grad_aprox(x):
        return gradient_aproximativ(F, x, h)
    return grad_aprox

# Testare si comparare
def testeaza_functie(nume, F, grad_analitic, x0, x_star=None):
    print("=" * 70)
    print(f"  Functia: {nume}")
    if x_star is not None:
        print(f"  Minim exact: x* = {x_star}")
    print(f"  Punct initial: x0 = {x0}")
    print("=" * 70)

    grad_aprox = make_grad_aprox(F, h=1e-6)
    epsilon = 1e-5

    # Lista configuratiilor de testat
    configuratii = [
        ("Gradient ANALITIC + Rata CONSTANTA",    grad_analitic, 'constanta'),
        ("Gradient ANALITIC + Rata BACKTRACKING", grad_analitic, 'backtracking'),
        ("Gradient APROXIMATIV + Rata CONSTANTA",    grad_aprox, 'constanta'),
        ("Gradient APROXIMATIV + Rata BACKTRACKING", grad_aprox, 'backtracking'),
    ]

    for desc, gf, metoda in configuratii:
        x_sol, iteratii, converge = gradient_descendent(
            F, gf, x0.copy(),
            metoda_eta=metoda,
            epsilon=epsilon,
            kmax=30000,
            eta_val=1e-3,
            beta=0.8
        )
        status = "CONVERGENTA" if converge else "DIVERGENTA"
        print(f"\n  {desc}")
        print(f"    Status: {status}")
        print(f"    Iteratii: {iteratii}")
        print(f"    Solutie gasita: x = {x_sol}")
        print(f"    F(x) = {F(x_sol):.10f}")
        if x_star is not None:
            eroare = np.linalg.norm(x_sol - np.array(x_star))
            print(f"    Eroare fata de x*: ||x - x*|| = {eroare:.2e}")
    print()

if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("#  TEMA 8 - Metoda Gradientului Descendent")
    print("#  Minimizarea functiilor F: R^n -> R")
    print("#" * 70 + "\n")

    #Testare Functia 1
    x0_f1 = np.array([0.5, 0.5])
    testeaza_functie("l(w0,w1) = -ln(1-sigma(w0-w1)) - ln(sigma(w0+w1))",
                     f1, grad_f1, x0_f1, x_star=None)

    #Testare Functia 2
    x0_f2 = np.array([0.0, 0.0])
    testeaza_functie("F(x1,x2) = x1^2 + x2^2 - 2*x1 - 4*x2 - 1",
                     f2, grad_f2, x0_f2, x_star=[1.0, 2.0])

    #Testare Functia 3
    x0_f3 = np.array([0.0, 0.0])
    testeaza_functie("F(x1,x2) = 3*x1^2 - 12*x1 + 2*x2^2 + 16*x2 - 10",
                     f3, grad_f3, x0_f3, x_star=[2.0, -4.0])

    #Testare Functia 4
    x0_f4 = np.array([5.0, 3.0])
    testeaza_functie("F(x1,x2) = x1^2 - 4*x1*x2 + 4.5*x2^2 - 4*x2 + 3",
                     f4, grad_f4, x0_f4, x_star=[8.0, 4.0])

    #Testare Functia 5
    x0_f5 = np.array([-0.5, 0.3])
    testeaza_functie("F(x1,x2) = x1^2*x2 - 2*x1*x2^2 + 3*x1*x2 + 4",
                     f5, grad_f5, x0_f5, x_star=[-1.0, 0.5])

    print("\n" + "=" * 70)
    print("  COMPARATIE: Gradient Analitic vs. Gradient Aproximativ")
    print("  (numar de iteratii pentru aceeasi precizie epsilon = 1e-5)")
    print("=" * 70)

    functii = [
        ("F2", f2, grad_f2, np.array([0.0, 0.0])),
        ("F3", f3, grad_f3, np.array([0.0, 0.0])),
        ("F4", f4, grad_f4, np.array([5.0, 3.0])),
        ("F5", f5, grad_f5, np.array([-0.5, 0.3])),
    ]

    print(f"\n  {'Functie':<10} {'Metoda eta':<15} {'Iter. analitic':<18} {'Iter. aproximativ':<18}")
    print("  " + "-" * 65)

    for nume, F, grad_an, x0 in functii:
        grad_ap = make_grad_aprox(F, h=1e-6)
        for metoda in ['constanta', 'backtracking']:
            _, it_an, conv_an = gradient_descendent(F, grad_an, x0.copy(),
                                                     metoda_eta=metoda, epsilon=1e-5)
            _, it_ap, conv_ap = gradient_descendent(F, grad_ap, x0.copy(),
                                                     metoda_eta=metoda, epsilon=1e-5)
            s_an = str(it_an) if conv_an else f"{it_an} (div)"
            s_ap = str(it_ap) if conv_ap else f"{it_ap} (div)"
            print(f"  {nume:<10} {metoda:<15} {s_an:<18} {s_ap:<18}")

    print("\n  Concluzie: Gradientul analitic si cel aproximativ produc rezultate")
    print("  similare. Numarul de iteratii este aproximativ egal pentru ambele")
    print("  metode de calcul al gradientului, ceea ce confirma acuratetea")
    print("  formulei de diferente finite centrate de ordin 4.\n")
