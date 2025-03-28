import numpy as np
import matplotlib.pyplot as plt
from math import ceil

# ============================================
# 1. SIMULASI NVAR(p,1) - DIPERBARUI SESUAI PERSAMAAN (2)
# ============================================

def simulate_nvar_p(A, alphas, T=1000, sigma_u=0.1, burn_in=500, seed=None):
    """
    Simulasi yang diperbaiki dengan burn-in lebih panjang dan inisialisasi stabil
    """
    np.random.seed(seed)
    p = len(alphas)
    n = A.shape[0]
    total_T = T + burn_in
    y = np.zeros((total_T, n))
    u = np.random.normal(0, sigma_u, (total_T, n))
    
    # Inisialisasi dengan proses stabil
    for t in range(1, p):
        y[t] = 0.5*y[t-1] + np.random.normal(0, 0.01, n)
    
    # Simulasi utama dengan kontrol stabilitas
    for t in range(p, total_T):
        y[t] = sum(alphas[l] * (A @ y[t-l-1]) for l in range(p)) + u[t]
        
        # Penyesuaian stabilitas
        if np.max(np.abs(y[t])) > 1e3:
            raise ValueError("Proses tidak stabil. Periksa parameter.")
    
    return y[burn_in:]

# ============================================
# 2. ESTIMASI PARAMETER DENGAN OLS - DIPERBARUI
# ============================================

def estimate_alphas(A, Y, p):
    """
    Estimasi parameter α untuk NVAR(p,1) dengan matriks desain yang benar
    
    Parameters:
        A (np.ndarray): Matriks adjacency (n x n)
        Y (np.ndarray): Data time series (T x n)
        p (int): Orde lag
        
    Returns:
        np.ndarray: Estimasi parameter [α1, ..., αp]
    """
    T, n = Y.shape
    num_obs = (T - p) * n  # Jumlah total observasi
    
    # Inisialisasi matriks desain dan target
    X = np.zeros((num_obs, p))
    y = Y[p:].ravel()  # Target dalam bentuk vektor
    
    # Bangun matriks desain
    for l in range(p):
        # Hitung AY untuk lag ke-l
        AY = (A @ Y[p-l-1:T-l-1].T).T  # Dimensi (T-p) x n
        
        # Assign ke kolom X dengan reshape yang benar
        X[:, l] = AY.ravel()  # Ravel untuk flatten menjadi vektor
    
    # Ridge regression untuk stabilitas numerik
    I = np.eye(p)
    alphas = np.linalg.lstsq(X.T @ X + 1e-6*I, X.T @ y, rcond=None)[0]
    
    return alphas

# ============================================
# 3. IMPULSE RESPONSE FUNCTION - SESUAI PERSAMAAN (3)
# ============================================

def compute_girf_p(A, alphas, h_max=20):
    """
    Menghitung Generalized Impulse Response Function (GIRF) untuk NVAR(p,1)
    yang sesuai dengan persamaan (3) dalam paper
    
    Parameters:
        A (np.ndarray): Matriks adjacency jaringan (n x n)
        alphas (list): Koefisien [α1, α2, ..., αp]
        h_max (int): Horizon maksimum respons
        
    Returns:
        list: List matriks GIRF untuk h=0 sampai h_max
    """
    p = len(alphas)
    n = A.shape[0]
    girf = [np.eye(n)]  # h=0 (matriks identitas)
    
    # Precompute semua pangkat matriks A yang diperlukan
    A_powers = [np.eye(n)]  # A^0
    for k in range(1, h_max+1):
        A_powers.append(A @ A_powers[-1])
    
    # Fungsi bantu untuk menghitung koefisien c_k^h(α)
    def compute_coeff(k, h):
        """
        Menghitung koefisien c_k^h(α) sesuai persamaan (3) dalam paper
        """
        if k > h:
            return 0.0
        
        # Kasus khusus untuk koneksi langsung (k=1)
        if k == 1 and h <= p:
            return alphas[h-1]
        
        # Hitung jumlah semua path yang valid dari 1 ke h dengan panjang k
        coeff = 0.0
        # Implementasi sederhana - bisa dioptimasi lebih lanjut
        # Ini adalah pendekatan rekursif untuk menghitung koefisien
        if k == 1:
            if h <= p:
                return alphas[h-1]
            else:
                return 0.0
        else:
            for m in range(1, min(p, h-k+1)+1):
                coeff += alphas[m-1] * compute_coeff(k-1, h-m)
        
        return coeff
    
    # Hitung GIRF untuk setiap horizon
    for h in range(1, h_max+1):
        current = np.zeros((n,n))
        k_min = ceil(h/p)  # Sesuai Proposition dalam paper
        
        for k in range(k_min, h+1):
            c = compute_coeff(k, h)
            current += c * A_powers[k]
        
        girf.append(current)
    
    return girf

# ============================================
# 4. FREKUENSI OBSERVASI vs INTERAKSI JARINGAN - DIPERBARUI
# ============================================

def simulate_frequency_mismatch(A, alphas, freq_ratio=3, T=100, seed=None):
    """
    Simulasi ketidaksesuaian frekuensi antara frekuensi jaringan dan frekuensi observasi
    
    Parameters:
        A (np.ndarray): Matriks adjacency jaringan (n x n)
        alphas (list): Koefisien [α1, α2, ..., αp]
        freq_ratio (int): Rasio frekuensi jaringan terhadap frekuensi observasi
        T (int): Jumlah observasi yang diinginkan
        seed (int): Seed untuk random generator
        
    Returns:
        tuple: (y_obs, y_high) dimana:
               y_obs: data agregasi frekuensi rendah (T x n)
               y_high: data frekuensi tinggi (T*freq_ratio x n)
    """
    np.random.seed(seed)
    n = A.shape[0]
    
    # Simulasi proses frekuensi tinggi (network interaction frequency)
    y_high = simulate_nvar_p(A, alphas, T*freq_ratio, burn_in=100)
    
    # Agregasi temporal (observational frequency)
    y_obs = np.zeros((T, n))
    for t in range(T):
        # Moving average untuk agregasi
        y_obs[t] = np.mean(y_high[t*freq_ratio:(t+1)*freq_ratio], axis=0)
    
    return y_obs, y_high

# ============================================
# 5. ANALISIS GRANGER-CAUSALITY - SESUAI PROPOSITION
# ============================================

def granger_causality_matrix(A, alphas, h):
    """
    Membuat matriks Granger-causality untuk horizon h sesuai Proposition
    """
    p = len(alphas)
    n = A.shape[0]
    k_min = ceil(h/p)
    
    # Hitung semua A^k untuk k_min ≤ k ≤ h
    gc_mat = np.zeros((n,n))
    for k in range(k_min, h+1):
        gc_mat += np.linalg.matrix_power(A, k)
    
    return (gc_mat != 0).astype(float)

# ============================================
# 6. VISUALISASI - DIPERBARUI DENGAN MULTI-HORIZON
# ============================================

def plot_granger_effects(A, girf, shock_unit=0, max_h=10):
    """
    Plot efek Granger-causality untuk berbagai horizon
    """
    n = A.shape[0]
    fig, ax = plt.subplots(1, 2, figsize=(18,6))
    
    # Panel kiri: Struktur jaringan
    im = ax[0].imshow(A, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax[0])
    ax[0].set_title('Network Structure')
    
    # Panel kanan: Respons multi-horizon
    horizons = range(1, max_h+1)
    for i in range(n):
        responses = [girf[h][i,shock_unit] for h in horizons]
        ax[1].plot(horizons, responses, marker='o', label=f'Unit {i+1}')
    
    ax[1].axhline(0, color='k', linestyle='--')
    ax[1].set_title(f'Response to Shock at Unit {shock_unit+1}')
    ax[1].set_xlabel('Horizon')
    ax[1].set_ylabel('Response')
    ax[1].legend()
    ax[1].grid(True)
    
    plt.tight_layout()
    plt.show()

# ============================================
# 7. DEMONSTRASI - DIPERBARUI DENGAN CONTOH PAPER
# ============================================

if __name__ == "__main__":
    # Contoh jaringan dan parameter
    A = np.array([
        [0.0, 0.0, 0.8],
        [0.7, 0.0, 0.2],
        [0.0, 0.9, 0.0]
    ])
    alphas = [0.6, 0.3]  # p=2
    
    # Simulasi data
    Y = simulate_nvar_p(A, alphas, T=1000, seed=42)
    
    # Estimasi parameter
    alphas_est = estimate_alphas(A, Y, p=2)
    print(f"Parameter estimated: {alphas_est}")
    
    # Hitung GIRF
    girf = compute_girf_p(A, alphas_est, h_max=10)
    
    # Visualisasi
    plot_granger_effects(A, girf, shock_unit=0)
    
    # Analisis frekuensi
    y_obs, y_high = simulate_frequency_mismatch(A, alphas, freq_ratio=3)
    
    # Plot hasil frekuensi
    plt.figure(figsize=(12,6))
    plt.plot(y_high[::3,0], 'ko-', label='Quarterly Observations')
    plt.plot(y_high[:,0], alpha=0.3, label='Monthly Process')
    plt.title('Example of Frequency Mismatch')
    plt.legend()
    plt.show()
