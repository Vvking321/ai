import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# 1. Definicja dziedzin (uniwersum)
x_z = np.arange(0, 101, 1)
x_psi = np.arange(-np.pi, np.pi + 0.01, 0.01)
x_v = np.arange(-10, 10.1, 0.1)

# 2. Definicja funkcji przynależności
z_NR = fuzz.trimf(x_z, [0, 0, 100])
z_FR = fuzz.trimf(x_z, [0, 100, 100])

psi_N = fuzz.trimf(x_psi, [-np.pi, -np.pi, 0])
psi_Z = fuzz.trimf(x_psi, [-np.pi, 0, np.pi])
psi_P = fuzz.trimf(x_psi, [0, np.pi, np.pi])

v_B = fuzz.trimf(x_v, [-10, -10, 0])
v_S = fuzz.trimf(x_v, [-10, 0, 10])
v_F = fuzz.trimf(x_v, [0, 10, 10])

# --- WYKRES 1: Funkcje przynależności dla wejść i wyjść ---
fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(10, 12))

# Wykres Z
ax0.plot(x_z, z_NR, 'b', linewidth=2, label='NR (Near)')
ax0.plot(x_z, z_FR, 'g', linewidth=2, label='FR (Far)')
ax0.set_title('Zmienna wejściowa: z')
ax0.legend()

# Wykres Psi
ax1.plot(x_psi, psi_N, 'r', linewidth=2, label='N (Negative)')
ax1.plot(x_psi, psi_Z, 'g', linewidth=2, label='Z (Zero)')
ax1.plot(x_psi, psi_P, 'b', linewidth=2, label='P (Positive)')
ax1.set_title('Zmienna wejściowa: psi (radiany)')
ax1.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax1.set_xticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
ax1.legend()

# Wykres wyjść (vl i vr mają takie same bazy)
ax2.plot(x_v, v_B, 'r', linewidth=2, label='B (Backward)')
ax2.plot(x_v, v_S, 'g', linewidth=2, label='S (Stop)')
ax2.plot(x_v, v_F, 'b', linewidth=2, label='F (Forward)')
ax2.set_title('Zmienne wyjściowe: vl oraz vr')
ax2.legend()

plt.tight_layout()
plt.show()

# --- WYKRES 2: Wnioskowanie i wyostrzanie (Mamdani) ---
# Budujemy system kontroli, aby pobrać dokładne wykresy po agregacji
z_ctrl = ctrl.Antecedent(x_z, 'z')
psi_ctrl = ctrl.Antecedent(x_psi, 'psi')
vl_ctrl = ctrl.Consequent(x_v, 'vl')
vr_ctrl = ctrl.Consequent(x_v, 'vr')

z_ctrl['NR'], z_ctrl['FR'] = z_NR, z_FR
psi_ctrl['N'], psi_ctrl['Z'], psi_ctrl['P'] = psi_N, psi_Z, psi_P
for v in [vl_ctrl, vr_ctrl]:
    v['B'], v['S'], v['F'] = v_B, v_S, v_F

rules = [
    ctrl.Rule(z_ctrl['NR'] & psi_ctrl['N'], (vl_ctrl['B'], vr_ctrl['F'])),
    ctrl.Rule(z_ctrl['NR'] & psi_ctrl['Z'], (vl_ctrl['S'], vr_ctrl['S'])),
    ctrl.Rule(z_ctrl['NR'] & psi_ctrl['P'], (vl_ctrl['F'], vr_ctrl['B'])),
    ctrl.Rule(z_ctrl['FR'] & psi_ctrl['N'], (vl_ctrl['B'], vr_ctrl['F'])),
    ctrl.Rule(z_ctrl['FR'] & psi_ctrl['Z'], (vl_ctrl['F'], vr_ctrl['F'])),
    ctrl.Rule(z_ctrl['FR'] & psi_ctrl['P'], (vl_ctrl['F'], vr_ctrl['B']))
]

system = ctrl.ControlSystem(rules)
sim = ctrl.ControlSystemSimulation(system)

# Podanie danych wejściowych
sim.input['z'] = 70
sim.input['psi'] = -np.pi / 4
sim.compute()

# --- WYŚWIETLENIE WYNIKÓW W KONSOLI ---
print("="*40)
print("WYNIKI STEROWANIA ROZMYTEGO:")
print(f"Wyjście vl (lewy silnik):  {sim.output['vl']}")
print(f"Wyjście vr (prawy silnik): {sim.output['vr']}")
print("="*40)

# Wyświetlenie wykresów wyostrzenia
vl_ctrl.view(sim=sim)
plt.title('Wynik wnioskowania i środek ciężkości dla vl')
plt.show()

vr_ctrl.view(sim=sim)
plt.title('Wynik wnioskowania i środek ciężkości dla vr')
plt.show()
