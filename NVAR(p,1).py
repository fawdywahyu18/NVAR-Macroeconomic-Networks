import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz

# ============================================
# 1. FUNGSI SIMULASI UNTUK NVAR(p,1) - DIPERBARUI
# ============================================

def simulate_nvar_p(A, alphas, T=100, sigma_u=0.1, burn_in=100, seed=None):
    """
    Simulasi model NVAR(p,1) dengan periode burn-in
    
    Parameters:
        A (np.ndarray): Matriks adjacency (n x n)
        alphas (list): Koefisien [α1, α2, ..., αp]
        T (int): Jumlah observasi
        sigma_u (float): Standar deviasi inovasi
        burn_in (int): Periode burn-in untuk stabilisasi
        seed (int): Seed untuk generator acak
    """
    np.random.seed(seed)
    p = len(alphas)
    n = A.shape[0]
    total_T = T + burn_in
    y = np.zeros((total_T, n))
    u = np.random.normal(0, sigma_u, (total_T, n))
    
    # Inisialisasi dengan nilai acak
    y[:p] = np.random.normal(0, 0.1, (p, n))
    
    for t in range(p, total_T):
        y[t] = sum(alphas[l] * (A @ y[t-l-1]) for l in range(p)) + u[t]
    
    return y[burn_in:]

# ============================================
# 2. ESTIMASI PARAMETER DENGAN OLS - DIPERBARUI
# ============================================

def estimate_alphas(A, Y, p):
    """
    Corrected estimation function with proper dimension handling
    
    Parameters:
        A (np.ndarray): Matriks adjacency (n x n)
        Y (np.ndarray): Data time series (T x n)
        p (int): Orde lag
        
    Returns:
        np.ndarray: Estimasi parameter [α1, ..., αp]
    """
    T, n = Y.shape
    X = np.zeros(((T-p)*n, p))  # Mengubah dimensi X
    y = Y[p:].ravel()           # Target dalam bentuk vektor
    
    # Membangun matriks desain dengan benar
    for l in range(p):
        # Hitung A @ Y untuk setiap lag
        AY = (A @ Y[p-l-1:T-l-1].T).T
        X[:, l] = AY.ravel()    # Assign ke kolom yang sesuai
        
    # Ridge regression untuk stabilitas
    alphas = np.linalg.lstsq(X.T @ X + 1e-6*np.eye(p), X.T @ y, rcond=None)[0]
    
    return alphas
# ============================================
# 3. IMPULSE RESPONSE FUNCTION - DIPERBARUI
# ============================================

def compute_girf_p(A, alphas, h_max=20):
    """
    Corrected implementation of GIRF computation for NVAR(p,1)
    
    Parameters:
        A (np.ndarray): Adjacency matrix (n x n)
        alphas (list): Coefficients [α1, α2, ..., αp]
        h_max (int): Maximum horizon
        
    Returns:
        list: List of GIRF matrices for h=0 to h_max
    """
    p = len(alphas)
    n = A.shape[0]
    girf = [np.eye(n)]  # h=0 (identity matrix)
    
    # Precompute all required powers of A
    A_powers = [np.eye(n)]  # A^0
    for k in range(1, h_max+1):
        A_powers.append(A @ A_powers[-1])
    
    # Compute GIRF recursively
    for h in range(1, h_max+1):
        current = np.zeros((n, n))
        
        # Case 1: Direct effect from current period
        if h <= p:
            current += alphas[h-1] * A
        
        # Case 2: Effects propagating through previous periods
        for m in range(1, min(p, h)+1):
            if h-m >= 0:
                current += alphas[m-1] * (A @ girf[h-m])
        
        girf.append(current)
    
    return girf
# ============================================
# 4. FREKUENSI OBSERVASI vs INTERAKSI JARINGAN
# ============================================

def simulate_frequency_mismatch(A, alphas, obs_ratio=2, T=100, seed=None):
    """
    Simulasi ketidaksesuaian frekuensi antara interaksi jaringan dan observasi
    
    obs_ratio: Rasio frekuensi jaringan terhadap observasi (misal 2 = 2 interaksi jaringan per 1 observasi)
    """
    np.random.seed(seed)
    p = len(alphas)
    n = A.shape[0]
    
    # Simulasi proses frekuensi tinggi
    total_periods = T * obs_ratio
    y_high = simulate_nvar_p(A, alphas, total_periods, seed=seed)
    
    # Agregasi temporal
    y_obs = np.zeros((T, n))
    for t in range(T):
        start = t * obs_ratio
        end = (t+1) * obs_ratio
        y_obs[t] = np.mean(y_high[start:end], axis=0)
    
    return y_obs, y_high

# ============================================
# 5. ANALISIS GRANGER-CAUSALITY
# ============================================

def granger_causality_matrix(girf, h=1, threshold=0.01):
    """
    Membuat matriks Granger-causality untuk horizon h
    """
    return (np.abs(girf[h]) > threshold).astype(int)

# ============================================
# 6. VISUALISASI - DIPERBARUI
# ============================================

def plot_network_effects(A, girf, shock_unit=0, max_horizon=10):
    """
    Plot efek jaringan untuk shock pada unit tertentu
    """
    plt.figure(figsize=(15, 6))
    
    # Plot struktur jaringan
    plt.subplot(1, 2, 1)
    plt.imshow(A, cmap='viridis')
    plt.title('Struktur Jaringan')
    plt.colorbar()
    
    # Plot impulse responses
    plt.subplot(1, 2, 2)
    n = A.shape[0]
    for i in range(n):
        responses = [girf[h][i, shock_unit] for h in range(max_horizon+1)]
        plt.plot(responses, marker='o', label=f'Unit {i+1}')
    
    plt.title(f'Respons Terhadap Shock di Unit {shock_unit+1}')
    plt.xlabel('Horizon')
    plt.ylabel('Respons')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# ============================================
# 7. DEMONSTRASI
# ============================================

# Contoh jaringan dari paper
A = np.array([
    [0.0, 0.0, 0.8],
    [0.7, 0.0, 0.2],
    [0.0, 0.9, 0.0]
])

# Parameter simulasi
alphas_true = [0.6, 0.3]  # NVAR(2,1)
p = len(alphas_true)

# Simulasi data
np.random.seed(42)
Y = simulate_nvar_p(A, alphas_true, T=1000)

# Estimasi parameter
alphas_est = estimate_alphas(A, Y, p)
print(f"Parameter True: {alphas_true}")
print(f"Parameter Estimated: {alphas_est.round(3)}")

# Hitung GIRF
girf = compute_girf_p(A, alphas_est, h_max=10)

# Visualisasi
plot_network_effects(A, girf, shock_unit=0)

# Analisis Granger-causality
gc_matrix = granger_causality_matrix(girf, h=3)
print("\nMatriks Granger-causality untuk horizon 3:")
print(gc_matrix)

# Simulasi ketidaksesuaian frekuensi
y_obs, y_high = simulate_frequency_mismatch(A, alphas_true, obs_ratio=3, T=500)

# Plot hasil
plt.figure(figsize=(12, 6))
plt.plot(y_high[::3, 0], 'ko-', label='Observasi')
plt.plot(y_high[:, 0], alpha=0.5, label='Proses Jaringan')
plt.title('Ketidaksesuaian Frekuensi Observasi vs Interaksi Jaringan')
plt.xlabel('Waktu Observasi')
plt.ylabel('Nilai')
plt.legend()
plt.show()
