"""
Created on Wed Mar 26 14:15:39 2025

@author: fawdywahyu
"""

import numpy as np
import matplotlib.pyplot as plt

# ========================
# 1. SIMULASI DATA NVAR(1,1)
# ========================

def simulate_nvar(A, alpha, T=100, sigma_u=0.1, seed=None):
    """
    Simulasi data dari model NVAR(1,1)
    
    Parameters:
        A (np.ndarray): Matriks adjacency jaringan (n x n)
        alpha (float): Parameter autoregressive
        T (int): Jangka waktu
        sigma_u (float): Standar deviasi inovasi
        seed (int): Seed untuk random generator
        
    Returns:
        np.ndarray: Data simulasi (T x n)
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = A.shape[0]
    y = np.zeros((T, n))
    u = np.random.normal(0, sigma_u, (T, n))
    
    for t in range(1, T):
        y[t] = alpha * A @ y[t-1] + u[t]
    
    return y

# ========================
# 2. ESTIMASI PARAMETER α
# ========================

def estimate_alpha(A, Y):
    """
    Estimasi parameter α menggunakan OLS
    
    Parameters:
        A (np.ndarray): Matriks adjacency jaringan
        Y (np.ndarray): Data time series (T x n)
        
    Returns:
        float: Estimasi α
    """
    Y_lag = Y[:-1]  # y_{t-1}
    Y_current = Y[1:]  # y_t
    
    # Reshape data ke bentuk vektor
    X = (A @ Y_lag.T).T.ravel()  # A @ y_{t-1} untuk semua t
    y = Y_current.ravel()
    
    # OLS: y = α * X + ε
    alpha = np.linalg.lstsq(X[:, np.newaxis], y, rcond=None)[0][0]
    
    return alpha

# ==================================
# 3. FUNGSI IMPULSE RESPONSE (GIRF)
# ==================================

def compute_girf(A, alpha, h_max):
    """
    Menghitung Generalized Impulse Response Function
    
    Parameters:
        A (np.ndarray): Matriks adjacency
        alpha (float): Parameter autoregressive
        h_max (int): Horizon maksimum
        
    Returns:
        list: List matriks respons untuk h=0 sampai h_max
    """
    n = A.shape[0]
    girf = []
    
    # Inisialisasi dengan matriks identitas (h=0)
    current_matrix = np.eye(n)
    girf.append(current_matrix.copy())
    
    for h in range(1, h_max+1):
        current_matrix = (alpha ** h) * np.linalg.matrix_power(A, h)
        girf.append(current_matrix)
    
    return girf

# ====================
# 4. CONTOH JARINGAN
# ====================

def example_network():
    """Jaringan contoh dari paper (Section 2.1)"""
    return np.array([
        [0.0, 0.0, 0.8],
        [0.7, 0.0, 0.2],
        [0.0, 0.9, 0.0]
    ])

# ====================
# 5. VISUALISASI GIRF
# ====================

def plot_girf(girf, shock_unit=0, h_max=6):
    """
    Plot impulse response untuk unit tertentu
    
    Parameters:
        girf (list): List matriks GIRF
        shock_unit (int): Indeks unit yang menerima shock
        h_max (int): Horizon maksimum
    """
    n = girf[0].shape[0]
    horizons = list(range(h_max + 1))
    
    plt.figure(figsize=(12, 6))
    for i in range(n):
        responses = [girf[h][i, shock_unit] for h in range(h_max + 1)]
        plt.plot(horizons, responses, marker='o', label=f'Unit {i+1}')
    
    plt.title(f'Respons Terhadap Shock di Unit {shock_unit+1}')
    plt.xlabel('Horizon')
    plt.ylabel('Respons')
    plt.xticks(horizons)
    plt.legend()
    plt.grid(True)
    plt.show()

# ==========================
# 6. DEMO DENGAN CONTOH PAPER
# ==========================

# Parameter contoh
A = example_network()
alpha_true = 0.9
h_max = 6

# Simulasi data
Y = simulate_nvar(A, alpha_true, T=1000, sigma_u=0.1, seed=42)

# Estimasi α
alpha_est = estimate_alpha(A, Y)
print(f"Estimasi α: {alpha_est:.4f} (True: {alpha_true})")

# Hitung GIRF
girf = compute_girf(A, alpha_est, h_max)

# Cetak matriks GIRF untuk beberapa horizon
print("\nContoh Matriks GIRF (h=3):")
print(girf[3])

# Plot respons untuk shock di Unit 1
plot_girf(girf, shock_unit=0, h_max=h_max)
