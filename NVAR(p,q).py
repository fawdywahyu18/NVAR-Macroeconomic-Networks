import numpy as np
import matplotlib.pyplot as plt
from math import ceil

# ============================================
# 1. SIMULASI NVAR(p,q) - DIPERBARUI UNTUK STOCK/FLOW DAN FREKUENSI q
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


def simulate_nvar_pq(A, alphas, q=1, T=1000, variable_type='stock', sigma_u=0.1, burn_in=500, seed=None):
    """
    Simulasi NVAR(p,q) dengan frekuensi jaringan q kali frekuensi observasi
    
    Parameters:
        A (np.ndarray): Matriks adjacency (n x n)
        alphas (list): Koefisien [α1, ..., αp]
        q (int): Rasio frekuensi jaringan terhadap observasi (harus integer)
        T (int): Jumlah observasi yang diinginkan
        variable_type (str): 'stock' (stok) atau 'flow' (aliran)
        sigma_u (float): Standar deviasi inovasi
        burn_in (int): Burn-in periode untuk proses frekuensi tinggi
        seed (int): Seed untuk random generator
        
    Returns:
        np.ndarray: Data observasi (T x n)
    """
    np.random.seed(seed)
    n = A.shape[0]
    T_high_total = q * T + burn_in
    
    # Simulasi proses frekuensi tinggi (NVAR(p,1))
    tilde_y = simulate_nvar_p(A, alphas, T=T_high_total, sigma_u=sigma_u, burn_in=0, seed=seed)
    
    # Pembakaran data frekuensi tinggi
    tilde_y_burned = tilde_y[burn_in:]
    
    # Potong data sesuai kebutuhan
    if len(tilde_y_burned) < q*T:
        raise ValueError("Data frekuensi tinggi tidak cukup setelah burn-in")
    tilde_y_trimmed = tilde_y_burned[:q*T]
    
    # Agregasi berdasarkan tipe variabel
    if variable_type == 'stock':
        y_obs = tilde_y_trimmed[::q]  # Ambil setiap q periode
    elif variable_type == 'flow':
        y_obs = np.zeros((T, n))
        for t in range(T):
            blok = tilde_y_trimmed[t*q:(t+1)*q]
            y_obs[t] = np.mean(blok, axis=0)  # Rata-rata blok
    else:
        raise ValueError("Tipe variabel harus 'stock' atau 'flow'")
    
    return y_obs

# ============================================
# 2. MODIFIKASI GRANGER-CAUSALITY UNTUK q
# ============================================

def granger_causality_matrix_q(A, alphas, h, q=1):
    """
    Matriks Granger-causality untuk NVAR(p,q) sesuai Proposition
    
    Parameters:
        A (np.ndarray): Matriks adjacency
        alphas (list): Koefisien [α1, ..., αp]
        h (int): Horizon observasi
        q (int): Rasio frekuensi jaringan
        
    Returns:
        np.ndarray: Matriks biner Granger-causality
    """
    p = len(alphas)
    n = A.shape[0]
    h_total = h * q  # Konversi ke horizon frekuensi tinggi
    k_min = ceil(h_total / p)
    k_max = h_total
    
    # Hitung semua A^k yang diperlukan
    A_powers = [np.eye(n)]
    for k in range(1, k_max+1):
        A_powers.append(A @ A_powers[-1])
    
    # Jumlahkan dari k_min ke k_max
    gc_mat = np.zeros((n,n))
    for k in range(k_min, k_max+1):
        gc_mat += A_powers[k]
    
    return (gc_mat != 0).astype(float)

# ============================================
# 3. GIRF UNTUK FREKUENSI OBSERVASI DENGAN q
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


def compute_girf_pq(A, alphas, q=1, h_max=20):
    """
    Menghitung GIRF untuk proses observasi NVAR(p,q)
    
    Parameters:
        A (np.ndarray): Matriks adjacency
        alphas (list): Koefisien [α1, ..., αp]
        q (int): Rasio frekuensi jaringan
        h_max (int): Horizon maksimum observasi
        
    Returns:
        list: List matriks GIRF untuk h=0 sampai h_max
    """
    # Hitung GIRF frekuensi tinggi hingga h_max*q
    girf_high = compute_girf_p(A, alphas, h_max=q*h_max)
    
    # Ambil GIRF pada horizon observasi
    girf_obs = [girf_high[h*q] for h in range(h_max+1)]
    
    return girf_obs

# ============================================
# 4. CONTOH PENGGUNAAN
# ============================================

if __name__ == "__main__":
    # Contoh jaringan dan parameter
    A = np.array([
        [0.0, 0.0, 0.8],
        [0.7, 0.0, 0.2],
        [0.0, 0.9, 0.0]
    ])
    alphas = [0.6, 0.3]  # p=2
    q = 3  # Frekuensi jaringan 3x lebih tinggi
    
    # Simulasi data stok
    Y_stock = simulate_nvar_pq(A, alphas, q=q, T=1000, 
                              variable_type='stock', seed=42)
    
    # Simulasi data aliran
    Y_flow = simulate_nvar_pq(A, alphas, q=q, T=1000,
                             variable_type='flow', seed=42)
    
    # Hitung GIRF untuk observasi
    girf_stock = compute_girf_pq(A, alphas, q=q, h_max=10)
    
    # Visualisasi
    def plot_granger_effects_q(A, girf, q, shock_unit=0, max_h=10):
        n = A.shape[0]
        fig, ax = plt.subplots(1, 2, figsize=(18,6))
        
        # Panel kiri: Struktur jaringan
        ax[0].imshow(A, cmap='coolwarm', vmin=-1, vmax=1)
        ax[0].set_title('Struktur Jaringan')
        
        # Panel kanan: Respons multi-horizon
        horizons = range(1, max_h+1)
        for i in range(n):
            responses = [girf[h][i,shock_unit] for h in horizons]
            ax[1].plot(horizons, responses, 'o-', label=f'Unit {i+1}')
        
        ax[1].set_title(f'Respons Terhadap Shock di Unit {shock_unit+1} (q={q})')
        ax[1].set_xlabel('Horizon Observasi')
        ax[1].legend()
        plt.show()
    
    plot_granger_effects_q(A, girf_stock, q=q, shock_unit=0)
    
    # Contoh Granger-causality untuk q=3
    h_obs = 2
    gc_mat = granger_causality_matrix_q(A, alphas, h=h_obs, q=q)
    print(f"Matriks Granger-causality untuk h={h_obs} observasi (q={q}):")
    print(gc_mat)
