# -*- coding: utf-8 -*-
import hickle as hkl
import numpy as np
import nnet as net
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Potrzebne do wykresów 3D

# 1. Generowanie danych (minimum 10 punktów, zrobimy siatkę 10x10 = 100 punktów)
x1_range = np.linspace(-2, 2, 10)
x2_range = np.linspace(-2, 2, 10)
X1, X2 = np.meshgrid(x1_range, x2_range)

# Spłaszczamy siatkę do formatu wejściowego sieci: wiersz 1 = x1, wiersz 2 = x2
x = np.vstack([X1.ravel(), X2.ravel()])
y_t = np.array([X1.ravel() ** 3 + X2.ravel() ** 3])

# 2. Parametry sieci i uczenia
max_epoch = 20000
err_goal = 0.1
disp_freq = 1000

# Parametry adaptacyjnego lr i momentum
lr = 0.01
lr_inc = 1.05  # Współczynnik zwiększania lr
lr_dec = 0.7  # Współczynnik zmniejszania lr
max_perf_inc = 1.04  # Maksymalny dopuszczalny wzrost błędu (4%)
mc = 0.9  # Współczynnik momentum (stała)

L = x.shape[0]  # L = 2 (dwa wejścia)
K1 = 15  # Liczba neuronów w 1. warstwie (można dobrać eksperymentalnie)
K2 = 15  # Liczba neuronów w 2. warstwie
K3 = y_t.shape[0]  # K3 = 1 (jedno wyjście)

SSE_vec = []

# Inicjalizacja wag
w1, b1 = net.nwtan(K1, L)
w2, b2 = net.rands(K2, K1)
w3, b3 = net.rands(K3, K2)

# Inicjalizacja zmiennych dla momentum (poprzednie poprawki)
dw1_old, db1_old = np.zeros_like(w1), np.zeros_like(b1)
dw2_old, db2_old = np.zeros_like(w2), np.zeros_like(b2)
dw3_old, db3_old = np.zeros_like(w3), np.zeros_like(b3)

# Pierwsze przejście w przód, żeby policzyć początkowy błąd
y1 = net.tansig(np.dot(w1, x), b1)
y2 = net.tansig(np.dot(w2, y1), b2)
y3 = net.purelin(np.dot(w3, y2), b3)
e = y_t - y3
SSE = net.sumsqr(e)

plt.close('all')
plt.ion()

fig1 = plt.figure(1, figsize=(15, 5))

# Główna pętla uczenia
for epoch in range(1, max_epoch + 1):
    # Zachowujemy stare wagi na wypadek, gdyby błąd za bardzo wzrósł
    w1_old, b1_old = w1.copy(), b1.copy()
    w2_old, b2_old = w2.copy(), b2.copy()
    w3_old, b3_old = w3.copy(), b3.copy()
    SSE_old = SSE

    # Wyznaczenie gradientów (wsteczna propagacja)
    d3 = net.deltalin(y3, e)
    d2 = net.deltatan(y2, d3, w3)
    d1 = net.deltatan(y1, d2, w2)

    # Obliczenie poprawek z uwzględnieniem momentum
    dw1, db1 = net.learnbp(x, d1, lr)
    dw1 = mc * dw1_old + (1 - mc) * dw1
    db1 = mc * db1_old + (1 - mc) * db1

    dw2, db2 = net.learnbp(y1, d2, lr)
    dw2 = mc * dw2_old + (1 - mc) * dw2
    db2 = mc * db2_old + (1 - mc) * db2

    dw3, db3 = net.learnbp(y2, d3, lr)
    dw3 = mc * dw3_old + (1 - mc) * dw3
    db3 = mc * db3_old + (1 - mc) * db3

    # Aktualizacja wag
    w1 += dw1;
    b1 += db1
    w2 += dw2;
    b2 += db2
    w3 += dw3;
    b3 += db3

    # Nowe przejście w przód i obliczenie nowego błędu
    y1 = net.tansig(np.dot(w1, x), b1)
    y2 = net.tansig(np.dot(w2, y1), b2)
    y3 = net.purelin(np.dot(w3, y2), b3)
    e = y_t - y3
    SSE = net.sumsqr(e)

    if np.isnan(SSE):
        print("Błąd osiągnął wartość NaN. Przerywam.")
        break

    # --- LOGIKA ADAPTACYJNEGO WSPÓŁCZYNNIKA UCZENIA ---
    if SSE > SSE_old * max_perf_inc:
        # Błąd wzrósł za bardzo -> cofamy zmiany wag, zmniejszamy lr, zerujemy momentum
        w1, b1 = w1_old, b1_old
        w2, b2 = w2_old, b2_old
        w3, b3 = w3_old, b3_old
        lr = lr * lr_dec
        dw1_old, db1_old = np.zeros_like(w1), np.zeros_like(b1)
        dw2_old, db2_old = np.zeros_like(w2), np.zeros_like(b2)
        dw3_old, db3_old = np.zeros_like(w3), np.zeros_like(b3)
        SSE = SSE_old
    else:
        # Błąd maleje lub rośnie nieznacznie -> akceptujemy wagi, zwiększamy lr
        if SSE < SSE_old:
            lr = lr * lr_inc
        # Zapisujemy obecne poprawki jako stare dla kolejnej epoki
        dw1_old, db1_old = dw1, db1
        dw2_old, db2_old = dw2, db2
        dw3_old, db3_old = dw3, db3

    SSE_vec.append(SSE)

    if SSE < err_goal:
        break

    if epoch % 1000 == 0 or epoch == 1:
       print(f"Epoch: {epoch:5d} | SSE: {SSE:5.5e} | lr: {lr:5.5f}")

        # Rysowanie w trakcie uczenia bez blokowania pętli
    if epoch % 200 == 0 or epoch == 1:
        Y_t_grid = y_t.reshape(X1.shape)
        Y3_grid = y3.reshape(X1.shape)
        E_grid = e.reshape(X1.shape)

        fig1.clf()

        # Wykres 1: Funkcja celu
        ax1 = fig1.add_subplot(131, projection='3d')
        ax1.plot_surface(X1, X2, Y_t_grid, cmap='viridis')
        ax1.set_title(f'Funkcja celu ($y_t$)')

        # Wykres 2: Wyjście sieci - teraz zobaczysz jak faluje od 1. epoki!
        ax2 = fig1.add_subplot(132, projection='3d')
        ax2.plot_surface(X1, X2, Y3_grid, cmap='plasma')
        ax2.set_title(f'Wyjście sieci ($y^{{(3)}}$) | Epoka: {epoch}')

        # Wykres 3: Błąd aproksymacji
        ax3 = fig1.add_subplot(133, projection='3d')
        ax3.plot_surface(X1, X2, E_grid, cmap='coolwarm')
        ax3.set_title('Błąd aproksymacji ($e$)')

        plt.tight_layout()

        # Szybkie przerysowanie okna
        fig1.canvas.draw()
        fig1.canvas.flush_events()
        plt.pause(1e-5)

print(f"\nKoniec uczenia. Epoch: {epoch:5d} | Końcowe SSE: {SSE:5.5e}")

# Wyłączenie trybu interaktywnego, aby końcowy wykres zatrzymał się na ekranie
plt.ioff()

# Ostatni wykres (błąd uczenia) otwiera się w nowym oknie fig2
fig2 = plt.figure(2)
plt.plot(SSE_vec, label='Sieć 3-warstwowa (Momentum + Adap LR)')
plt.xlabel('Epoka (Krok uczenia)')
plt.ylabel('Suma kwadratów błędów (SSE)')
plt.yscale('log')
plt.title('Przebieg procesu uczenia')
plt.grid(True)
plt.legend()
plt.show()
