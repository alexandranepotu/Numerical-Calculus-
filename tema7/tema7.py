#40% llm

def horner(coef, v):
    b = coef[0]
    for i in range(1, len(coef)):
        b = coef[i] + b * v
    return b

# Calculul coeficientilor derivatei unui polinom
def coef_derivata(coef):
    n = len(coef) - 1  # gradul polinomului
    if n == 0:
        return [0.0]
    return [coef[i] * (n - i) for i in range(n)]


# Calculul intervalului [-R, R] ce contine toate radacinile reale
def calcul_R(coef):
    a0 = abs(coef[0])
    A = max(abs(c) for c in coef[1:])
    R = (a0 + A) / a0
    return R


# Metoda Newton 
def metoda_newton(coef, x0, epsilon, kmax=10000):
    # Calculam coeficientii derivatei P'
    d_coef = coef_derivata(coef)

    x = x0

    for k in range(kmax):
        # Evaluam P(xk) si P'(xk) folosind schema lui Horner
        Px = horner(coef, x)
        Ppx = horner(d_coef, x)

        # Daca derivata e prea mica, nu putem continua
        if abs(Ppx) <= epsilon:
            return x, k, False

        # Calculam pasul delta_x conform formulei Newton(3)
        delta_x = Px / Ppx

        # Actualizam aproximarea
        x = x - delta_x

        # Verificam convergenta: |delta_x| < epsilon
        if abs(delta_x) < epsilon:
            return x, k + 1, True

        # Verificam divergenta: |delta_x| > 10^8
        if abs(delta_x) > 1e8:
            return x, k + 1, False

    # Am depasit numarul maxim de iteratii
    return x, kmax, False


# Metoda Olver
def metoda_olver(coef, x0, epsilon, kmax=10000):
    # Calculam coeficientii derivatelor P' si P''
    d_coef = coef_derivata(coef)
    dd_coef = coef_derivata(d_coef)

    x = x0

    for k in range(kmax):
        # Evaluam P(xk), P'(xk), P''(xk) folosind schema lui Horner
        Px = horner(coef, x)
        Ppx = horner(d_coef, x)
        Pppx = horner(dd_coef, x)

        # Daca derivata e prea mica, nu putem continua
        if abs(Ppx) <= epsilon:
            return x, k, False

        # Calculam ck conform formulei (4)
        ck = (Px ** 2) * Pppx / (Ppx ** 3)

        # Calculam pasul delta_x conform formulei Olver (4)
        delta_x = Px / Ppx + 0.5 * ck

        # Actualizam aproximarea
        x = x - delta_x

        # Verificam convergenta: |delta_x| < epsilon
        if abs(delta_x) < epsilon:
            return x, k + 1, True

        # Verificam divergenta: |delta_x| > 10^8
        if abs(delta_x) > 1e8:
            return x, k + 1, False

    # Am depasit numarul maxim de iteratii
    return x, kmax, False

def e_radacina_noua(lista_radacini, valoare, epsilon):
    for r in lista_radacini:
        if abs(r - valoare) <= epsilon:
            return False
    return True

# Cautarea radacinilor din puncte de start diferite si compararea celor doua metode
def gaseste_radacini(coef, epsilon, kmax=10000, num_puncte=200):
    R = calcul_R(coef)

    # Generam puncte de start uniform distribuite in [-R, R]
    pas = 2 * R / (num_puncte - 1)
    puncte_start = [-R + i * pas for i in range(num_puncte)]

    # Liste pentru radacinile distincte si numarul de pasi
    rad_newton = []
    pasi_newton = []
    rad_olver = []
    pasi_olver = []

    # Comparatii directe (din acelasi x0, ambele metode converg la aceeasi radacina)
    comparatii = []

    for x0 in puncte_start:
        # Rulam ambele metode din acelasi punct de start
        rn, pn, cn = metoda_newton(coef, x0, epsilon, kmax)
        ro, po, co = metoda_olver(coef, x0, epsilon, kmax)

        # Colectam radacini distincte din Newton
        if cn and abs(rn) <= R + 1:
            if e_radacina_noua(rad_newton, rn, epsilon):
                rad_newton.append(rn)
                pasi_newton.append(pn)

        # Colectam radacini distincte din Olver
        if co and abs(ro) <= R + 1:
            if e_radacina_noua(rad_olver, ro, epsilon):
                rad_olver.append(ro)
                pasi_olver.append(po)

        # Daca ambele metode au convergit la aceeasi radacina,
        # salvam comparatia (doar prima aparitie pentru fiecare radacina)
        if cn and co and abs(rn - ro) <= epsilon:
            gasit = False
            for c in comparatii:
                if abs(c[0] - rn) <= epsilon:
                    gasit = True
                    break
            if not gasit:
                comparatii.append((rn, pn, po, x0))

    return R, rad_newton, pasi_newton, rad_olver, pasi_olver, comparatii

def main():
    epsilon = 1e-6

    # Cele 4 exemple
    exemple = [
        {
            'nume': 'P1(x) = (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6',
            'coef': [1.0, -6.0, 11.0, -6.0],
            'radacini_exacte': [1.0, 2.0, 3.0]
        },
        {
            'nume': 'P2(x) = (x-2/3)(x-1/7)(x+1)(x-3/2)',
            'coef': [42.0, -55.0, -42.0, 49.0, -6.0],
            'radacini_exacte': [-1.0, 1/7, 2/3, 3/2]
        },
        {
            'nume': 'P3(x) = (x-1)(x-1/2)(x-3)(x-1/4)',
            'coef': [8.0, -38.0, 49.0, -22.0, 3.0],
            'radacini_exacte': [0.25, 0.5, 1.0, 3.0]
        },
        {
            'nume': 'P4(x) = (x-1)^2*(x-2)^2 = x^4 - 6x^3 + 13x^2 - 12x + 4',
            'coef': [1.0, -6.0, 13.0, -12.0, 4.0],
            'radacini_exacte': [1.0, 2.0]
        }
    ]

    fisier = open('rezultate.txt', 'w')

    def scrie(text):
        """Scrie simultan pe ecran si in fisier."""
        print(text, end='')
        fisier.write(text)

    scrie("=" * 70 + "\n")
    scrie("  Tema 7 - Metoda Newton si Metoda Olver\n")
    scrie("  Aproximarea radacinilor reale ale unui polinom\n")
    scrie("  Schema lui Horner pentru evaluarea polinoamelor\n")
    scrie("  Precizia epsilon = {}\n".format(epsilon))
    scrie("=" * 70 + "\n\n")

    for ex in exemple:
        coef = ex['coef']

        scrie("Polinom: {}\n".format(ex['nume']))
        scrie("Coeficienti: {}\n".format(coef))
        scrie("Radacini exacte: {}\n".format(
            [round(r, 10) for r in ex['radacini_exacte']]))

        # Calculam intervalul [-R, R]
        R = calcul_R(coef)
        scrie("Intervalul [-R, R] = [{:.6f}, {:.6f}]\n\n".format(-R, R))

        # Gasim radacinile si comparam metodele
        R, rad_n, pasi_n, rad_o, pasi_o, comp = gaseste_radacini(
            coef, epsilon
        )

        # Sortam radacinile pentru afisare
        newton_sorted = sorted(zip(rad_n, pasi_n), key=lambda t: t[0])
        olver_sorted = sorted(zip(rad_o, pasi_o), key=lambda t: t[0])

        # Afisam rezultatele metodei Newton
        scrie("--- Metoda Newton ---\n")
        if newton_sorted:
            for rad, pasi in newton_sorted:
                scrie("  Radacina: {:>14.10f}  |  Pasi: {}\n".format(rad, pasi))
        else:
            scrie("  Nu s-au gasit radacini.\n")
        scrie("\n")

        # Afisam rezultatele metodei Olver
        scrie("--- Metoda Olver ---\n")
        if olver_sorted:
            for rad, pasi in olver_sorted:
                scrie("  Radacina: {:>14.10f}  |  Pasi: {}\n".format(rad, pasi))
        else:
            scrie("  Nu s-au gasit radacini.\n")
        scrie("\n")

        # Afisam comparatia directa Newton vs Olver
        scrie("--- Comparatie Newton vs Olver (numar de pasi) ---\n")
        comp_sorted = sorted(comp, key=lambda c: c[0])
        if comp_sorted:
            for rad, pn, po, x0 in comp_sorted:
                scrie("  Radacina ~{:>10.6f}: Newton = {:>3} pasi, "
                      "Olver = {:>3} pasi  (x0 = {:.4f})\n".format(
                          rad, pn, po, x0))
        else:
            scrie("  Nu exista comparatii directe.\n")
        scrie("\n")

        # Colectam si afisam radacinile distincte din ambele metode
        toate_rad = []
        for rad, _ in newton_sorted:
            if e_radacina_noua(toate_rad, rad, epsilon):
                toate_rad.append(rad)
        for rad, _ in olver_sorted:
            if e_radacina_noua(toate_rad, rad, epsilon):
                toate_rad.append(rad)
        toate_rad.sort()

        scrie("Radacini distincte gasite (ambele metode combinate):\n")
        for rad in toate_rad:
            scrie("  {:.10f}\n".format(rad))

        scrie("\n" + "-" * 70 + "\n\n")

    fisier.close()
    print("Rezultatele au fost salvate in fisierul 'rezultate.txt'.")


if __name__ == '__main__':
    main()
