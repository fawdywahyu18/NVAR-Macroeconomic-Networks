"""
Created on Wed Mar 26 14:15:39 2025
@author: fawdywahyu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz

# ========================
# 1. SIMULASI DATA NVAR(1,1) - DIPERBARUI
# ========================

def simulate_nvar(A, alpha, T=100, sigma_u=0.1, burn_in=100, seed=None):
    """
    Simulasi data dari model NVAR(1,1) dengan periode burn-in
    
    Parameters:
        A (np.ndarray): Matriks adjacency jaringan (n x n)
        alpha (float): Parameter autoregressive
        T (int): Jangka waktu
        sigma_u (float): Standar deviasi inovasi
        burn_in (int): Periode stabilisasi awal
        seed (int): Seed untuk random generator
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = A.shape[0]
    total_T = T + burn_in
    y = np.zeros((total_T, n))
    u = np.random.normal(0, sigma_u, (total_T, n))
    
    # Inisialisasi dengan nilai kecil untuk stabilitas
    y[0] = np.random.normal(0, 0.1, n)
    
    for t in range(1, total_T):
        y[t] = alpha * (A @ y[t-1]) + u[t]
    
    return y[burn_in:]

# ========================
# 2. ESTIMASI PARAMETER α - DIPERBARUI
# ========================

def estimate_alpha(A, Y):
    """
    Estimasi parameter α menggunakan OLS dengan regularisasi ridge
    
    Parameters:
        A (np.ndarray): Matriks adjacency jaringan
        Y (np.ndarray): Data time series (T x n)
    """
    Y_lag = Y[:-1]  # y_{t-1}
    Y_current = Y[1:]  # y_t
    
    # Bentuk matriks desain dengan vektorisasi
    X = (A @ Y_lag.T).T.reshape(-1, 1)
    y = Y_current.reshape(-1)
    
    # Regresi ridge untuk stabilitas numerik
    alpha = np.linalg.lstsq(X.T @ X + 1e-6*np.eye(1), X.T @ y, rcond=None)[0][0]
    
    return alpha

# ==================================
# 3. FUNGSI IMPULSE RESPONSE (GIRF) - DIPERBARUI
# ==================================

def compute_girf(A, alpha, h_max):
    """
    Menghitung Generalized Impulse Response Function dengan tracking walk
    
    Parameters:
        A (np.ndarray): Matriks adjacency
        alpha (float): Parameter autoregressive
        h_max (int): Horizon maksimum
    """
    n = A.shape[0]
    girf = [np.eye(n)]  # h=0
    
    # Precompute semua pangkat matriks
    A_powers = [np.eye(n)]
    for h in range(1, h_max+1):
        A_powers.append(A @ A_powers[-1])
    
    for h in range(1, h_max+1):
        current = (alpha**h) * A_powers[h]
        girf.append(current)
    
    return girf

# ====================
# 4. CONTOH JARINGAN - DIPERBARUI
# ====================

def example_network(signed=False):
    """Jaringan contoh dari paper dengan opsi signed network"""
    if signed:
        return np.array([
            [0.0, 0.0, 0.8],
            [-0.7, 0.0, 0.2],
            [0.0, 0.9, 0.0]
        ])
    return np.array([
        [0.0, 0.0, 0.8],
        [0.7, 0.0, 0.2],
        [0.0, 0.9, 0.0]
    ])

# ====================
# 5. VISUALISASI GIRF - DIPERBARUI
# ====================

def plot_girf(girf, A, shock_unit=0, h_max=6):
    """
    Plot impulse response dengan informasi jaringan
    """
    n = girf[0].shape[0]
    horizons = list(range(h_max + 1))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot struktur jaringan
    im = ax1.imshow(A, cmap='coolwarm', vmin=-1, vmax=1)
    ax1.set_title('Network Structure')
    ax1.set_xlabel('Target Unit')
    ax1.set_ylabel('Source Unit')
    plt.colorbar(im, ax=ax1)
    
    # Plot impulse responses
    for i in range(n):
        responses = [girf[h][i, shock_unit] for h in range(h_max + 1)]
        ax2.plot(horizons, responses, marker='o', label=f'Unit {i+1}')
    
    ax2.set_title(f'Response to Shock at Unit {shock_unit+1}')
    ax2.set_xlabel('Horizon')
    ax2.set_ylabel('Response')
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# ==========================
# 6. DEMO DENGAN CONTOH PAPER - DIPERBARUI
# ==========================

if __name__ == "__main__":
    # Parameter contoh sesuai paper
    A = example_network()
    alpha_true = 1.0  # Sesuai contoh di paper
    h_max = 6
    burn_in = 100
    T = 500
    
    # Simulasi data
    Y = simulate_nvar(A, alpha_true, T=T, burn_in=burn_in, seed=42)
    
    # Estimasi α
    alpha_est = estimate_alpha(A, Y)
    print(f"Estimation of α: {alpha_est:.4f} (True: {alpha_true})")
    
    # Hitung GIRF
    girf = compute_girf(A, alpha_est, h_max)
    
    # Verifikasi matriks A^3 sesuai paper
    print("\nVerification for Matrix A^3:")
    print("Calculation results:")
    print(girf[3].round(2))  # Alpha=1, jadi sama dengan A^3
    print("\nExample in the Paper:")
    print(np.array([
        [0.50, 0, 0.14],
        [0.13, 0.50, 0.04],
        [0, 0.16, 0.50]
    ]))
    
    # Plot hasil
    plot_girf(girf, A, shock_unit=0, h_max=h_max)
