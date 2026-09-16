# ==============================================================================
# SIMULASI BMS INTELLIGENT: XGBOOST & FUZZY LOGIC (LiPo 3S)
# ==============================================================================

# 1. SUBSISTEM XGBOOST (Estimasi SoC)
class XGBoostSoCEstimator:
    def __init__(self, learning_rate=0.1, reg_lambda=1.0):
        self.eta = learning_rate
        self.reg_lambda = reg_lambda

    def update_prediction(self, y_true, y_pred_current):
        """Menghitung gradien, hessian, bobot daun, dan pembaruan SoC"""
        # Gradien (g_i) dan Hessian (h_i) untuk Loss Function MSE
        g_i = y_pred_current - y_true
        h_i = 1.0

        # Kalkulasi bobot daun optimal (w*)
        w_star = -g_i / (h_i + self.reg_lambda)

        # Pembaruan estimasi SoC
        y_pred_new = y_pred_current + (self.eta * w_star)

        return {
            "g_i": g_i,
            "h_i": h_i,
            "w_star": w_star,
            "y_pred_new": y_pred_new
        }


# 2. SUBSISTEM FUZZY LOGIC (Penilaian Risiko Operasional)
def trimf(x, abc):
    """Fungsi Keanggotaan Segitiga (Triangular Membership Function)"""
    a, b, c = abc
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x < c:
        return (c - x) / (c - b)
    return 0.0


class FuzzyBMSController:
    def __init__(self):
        # Kurva Keanggotaan Tegangan (V) & Suhu (T)
        self.v_sedang = [10.5, 11.1, 11.7]
        self.t_hangat = [30.0, 37.5, 45.0]
        self.t_panas  = [35.0, 45.0, 55.0]

        # Target Risiko Output
        self.z_aman = 10.0     # 10%
        self.z_waspada = 50.0  # 50%

    def evaluate(self, V, T):
        # Step 1: Fuzzifikasi
        mu_v_sedang = trimf(V, self.v_sedang)
        mu_t_hangat = trimf(T, self.t_hangat)
        mu_t_panas  = trimf(T, self.t_panas)

        # Step 2: Evaluasi Aturan (Operator MIN)
        alpha_1 = min(mu_v_sedang, mu_t_hangat)  # Rule 1: Sedang AND Hangat -> Aman
        alpha_2 = min(mu_v_sedang, mu_t_panas)   # Rule 2: Sedang AND Panas -> Waspada

        # Step 3: Defuzzifikasi (Weighted Average)
        numerator = (alpha_1 * self.z_aman) + (alpha_2 * self.z_waspada)
        denominator = alpha_1 + alpha_2

        z_star = (numerator / denominator) if denominator != 0 else 0.0

        return {
            "mu_v_sedang": mu_v_sedang,
            "mu_t_hangat": mu_t_hangat,
            "mu_t_panas": mu_t_panas,
            "alpha_1": alpha_1,
            "alpha_2": alpha_2,
            "risk_level": z_star
        }


# ==============================================================================
# EKSEKUSI SIMULASI
# ==============================================================================
if __name__ == "__main__":
    # Data Sensor Input
    V_sensor = 11.1  # Volt
    I_sensor = 2.0   # Ampere
    T_sensor = 38.0  # Celsius

    # Kondisi Awal XGBoost
    y_true = 50.0    # SoC Aktual (%)
    y_pred_0 = 60.0  # Prediksi Awal (%)

    # Run XGBoost
    xgb = XGBoostSoCEstimator(learning_rate=0.1, reg_lambda=1.0)
    res_xgb = xgb.update_prediction(y_true, y_pred_0)

    # Run Fuzzy Logic
    fuzzy = FuzzyBMSController()
    res_fuzzy = fuzzy.evaluate(V_sensor, T_sensor)

    # Cetak Output
    print("=" * 55)
    print("      HASIL SIMULASI ALGORITMA BMS (ESP32 CORE)")
    print("=" * 55)
    print(f"INPUT SENSOR : V = {V_sensor}V | I = {I_sensor}A | T = {T_sensor}°C\n")

    print("[1] PEMROSESAN XGBOOST (ESTIMASI SoC)")
    print(f"  • SoC Prediksi Awal (y^0) : {y_pred_0:.1f}%")
    print(f"  • Gradien Loss (g_i)     : {res_xgb['g_i']:.1f}")
    print(f"  • Bobot Daun (w*)        : {res_xgb['w_star']:.1f}")
    print(f"  • Hasil Estimasi Baru    : {res_xgb['y_pred_new']:.2f}%\n")

    print("[2] PEMROSESAN FUZZY LOGIC (ANALISIS RISIKO)")
    print(f"  • μ(V_Sedang)            : {res_fuzzy['mu_v_sedang']:.2f}")
    print(f"  • μ(T_Hangat)            : {res_fuzzy['mu_t_hangat']:.2f}")
    print(f"  • μ(T_Panas)             : {res_fuzzy['mu_t_panas']:.2f}")
    print(f"  • Alpha Rule 1 (Aman)    : {res_fuzzy['alpha_1']:.2f}")
    print(f"  • Alpha Rule 2 (Waspada) : {res_fuzzy['alpha_2']:.2f}")
    print(f"  • Output Risiko (z*)     : {res_fuzzy['risk_level']:.2f}%\n")

    print("[3] AKSI KONTROLER (HARDWARE CONTROL)")
    if res_fuzzy['risk_level'] < 30.0:
        aksi = "NORMAL (MOSFET Active / Full Capacity)"
    elif res_fuzzy['risk_level'] < 70.0:
        aksi = "DERATING (Kurangi Pengisian PWM / Limit Current)"
    else:
        aksi = "TRIP (Pemutusan Sakelar MOSFET)"

    print(f"  • Status Sistem          : {aksi}")
    print(f"  • Telemetri SoC          : {res_xgb['y_pred_new']:.1f}%")
    print("=" * 55)