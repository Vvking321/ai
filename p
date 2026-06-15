wzorując się na tym programie:
import numpy as np import skfuzzy as fuzz
from skfuzzy import control as ctrl import matplotlib.pyplot as plt


# Definicja zmiennych stanu
x1 = ctrl.Antecedent(np.arange(-2,2.01,0.01), 'x1') x2 = ctrl.Antecedent(np.arange(-2,2.01,0.01), 'x2') u = ctrl.Consequent(np.arange(-1,1.01,0.01), 'u')

# Zbiory rozmyte dla poprzednika x1 x1['A1'] = fuzz.trimf(x1.universe, [-2, -2, 0])
x1['A2'] = fuzz.trimf(x1.universe, [-2, 0, 2])
x1['A3'] = fuzz.trimf(x1.universe, [0, 2, 2])

# Zbiory rozmyte dla poprzednika x2 x2['B1'] = fuzz.trimf(x2.universe, [-2, -2, 0])
x2['B2'] = fuzz.trimf(x2.universe, [-2, 0, 2])
x2['B3'] = fuzz.trimf(x2.universe, [ 0, 2, 2])

# Zbiory rozmyte dla następnika u u['C1'] = fuzz.trimf(u.universe, [-1, -1, 0])
u['C2'] = fuzz.trimf(u.universe, [-1, 0, 1])
u['C3'] = fuzz.trimf(u.universe, [0, 1, 1])

# Definicje reguł
regula1 = ctrl.Rule(x1['A1'] & x2['B2'], u['C1'])
regula2 = ctrl.Rule(x1['A1'] & x2['B3'], u['C2'])
regula3 = ctrl.Rule(x1['A2'] & x2['B2'], u['C2'])
regula4 = ctrl.Rule(x1['A2'] & x2['B3'], u['C3'])

# Dodanie zdefiniowanych reguł do zbioru rozmytego
napiwek_ctr = ctrl.ControlSystem([regula1,regula2,regula3,regula4]) napiwek_sym = ctrl.ControlSystemSimulation(napiwek_ctr)

# Obliczenie wyniku dla wartości x1 = -1.7, x2 = 0.9 napiwek_sym.input['x1'] = -1.7
napiwek_sym.input['x2'] = 0.9 napiwek_sym.compute() print('Wynik',napiwek_sym.output['u']) u.view(sim=napiwek_sym)

plt.show()



dzisiejszym zadaniem będzie stworzenie programu python do wyznaczenia wyjścia systemu wnioskującego mamdami wykorzystując bibliotekę skfuzzy, numpy, matplotlib z określonymi danymi:
3 wejścia: left = 70 stopni, front = 90 stopni, right = 0 stopni
2 wyjścia: vl i vr
mamy również podane wykresy:
pierwszy wykres ma w osi x przedział 0 100, y ma 0 1 . ma on również na sobie trójkąty, w których wykorzystujemy trimf czyli S trimf[0, 0, 100], B trimf[0, 100, 100]. i jest to wykres wejść
drugi wykres oś x to -50 50, trójkąty: back trimf[-50, -50, 0], front trimf[0, 50, 50] i jest to wykres vl vr
kolejnie, mamy podane reguły, które są zapisane w tabeli, gdzie kolumny po kolei, to: left; front; right; vl; vr. dane które są w tej tabeli reguł:
S S S front front
S S B back front
S B S back front
S B B back front 
B S S front back
B S B front back
B B S front front
B B B back front
