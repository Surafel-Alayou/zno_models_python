import math

# Constants
E_bulk = 3.37  # Energy band gap of bulk (3.37eV for ZnO)
h = 6.626 * (10**-34)  # Planck's constant (in Js)
e = 1.602 * (10**-19)  # Charge of an electron (in C)
m_e = 2.18 * (10**-31)  # Effective mass of an electron (in kg)
m_h = 4.1 * (10**-31)  # Effective mass of a hole (in kg)
eps = 8.66  # Relative permittivity
pi = math.pi
eps_o = 8.854 * (10**-12)  # Permittivity of free space (F/m)

# Get user input
E_nano = float(input("Please enter the energy band gap of the nanoparticle (in eV): "))

# Coefficients for the quadratic equation
a_coef = ((h**2) / (8 * e)) * ((1 / m_e) + (1 / m_h))
b_coef = -(1.8 * (e**2)) / (4 * pi * eps * eps_o)
c_coef = E_bulk - E_nano

# Function to solve quadratic equations
def solve_quadratic(a, b, c):
    discriminant = b**2 - 4 * a * c
    discriminant = abs(discriminant)  # Ensure discriminant is non-negative
    root1 = abs((-b + math.sqrt(discriminant)) / (2 * a))
    root2 = abs((-b - math.sqrt(discriminant)) / (2 * a))
    print("Discriminant ", math.sqrt(discriminant))
    print("a_coef ", a_coef)
    return root1, root2

# Solve for roots
roots = solve_quadratic(a_coef, b_coef, c_coef)

# Handle the roots and compute the size of the nanoparticle
r_bruss = [1 / root for root in roots]  # Consider only positive real roots
print("The size of the nanoparticle is:", r_bruss)
