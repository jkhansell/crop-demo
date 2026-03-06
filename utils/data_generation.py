# Cereals & Field Crops
optimal_conditions = {

    "rice": {
        "temperature": 27,
        "humidity": 85,
        "ph": 6.3,
        "rainfall": 1800,
    },

    "maize": {
        "temperature": 24,
        "humidity": 65,
        "ph": 6.0,
        "rainfall": 900,
    },

    "jute": {
        "temperature": 26,
        "humidity": 80,
        "ph": 6.5,
        "rainfall": 1600,
    },

    "cotton": {
        "temperature": 27,
        "humidity": 60,
        "ph": 6.5,
        "rainfall": 700,
    },
}

# Pulses / Legumes
optimal_conditions.update({

    "chickpea": {
        "temperature": 22,
        "humidity": 55,
        "ph": 6.5,
        "rainfall": 600,
    },

    "kidneybeans": {
        "temperature": 21,
        "humidity": 60,
        "ph": 6.2,
        "rainfall": 700,
    },

    "pigeonpeas": {
        "temperature": 26,
        "humidity": 65,
        "ph": 6.5,
        "rainfall": 900,
    },

    "mothbeans": {
        "temperature": 28,
        "humidity": 50,
        "ph": 7.0,
        "rainfall": 400,
    },

    "mungbean": {
        "temperature": 28,
        "humidity": 65,
        "ph": 6.2,
        "rainfall": 700,
    },

    "blackgram": {
        "temperature": 27,
        "humidity": 65,
        "ph": 6.5,
        "rainfall": 800,
    },

    "lentil": {
        "temperature": 20,
        "humidity": 55,
        "ph": 6.5,
        "rainfall": 500,
    },
})

# Fruits (Tropical & Subtropical)
optimal_conditions.update({

    "banana": {
        "temperature": 27,
        "humidity": 80,
        "ph": 6.0,
        "rainfall": 2000,
    },

    "mango": {
        "temperature": 27,
        "humidity": 65,
        "ph": 6.5,
        "rainfall": 1000,
    },

    "grapes": {
        "temperature": 23,
        "humidity": 55,
        "ph": 6.5,
        "rainfall": 700,
    },

    "watermelon": {
        "temperature": 26,
        "humidity": 60,
        "ph": 6.5,
        "rainfall": 600,
    },

    "muskmelon": {
        "temperature": 25,
        "humidity": 60,
        "ph": 6.5,
        "rainfall": 500,
    },

    "apple": {
        "temperature": 18,
        "humidity": 60,
        "ph": 6.5,
        "rainfall": 1000,
    },

    "orange": {
        "temperature": 24,
        "humidity": 65,
        "ph": 6.0,
        "rainfall": 1200,
    },

    "papaya": {
        "temperature": 26,
        "humidity": 75,
        "ph": 6.0,
        "rainfall": 1500,
    },

    "pomegranate": {
        "temperature": 25,
        "humidity": 55,
        "ph": 6.5,
        "rainfall": 500,
    },

    "coconut": {
        "temperature": 27,
        "humidity": 80,
        "ph": 6.2,
        "rainfall": 2000,
    },
})

# Plantation Crops

optimal_conditions.update({

    "coffee": {
        "temperature": 22,
        "humidity": 65,
        "ph": 6.0,
        "rainfall": 1800,
    }
})

import numpy as np
import pandas as pd 
import os 

# Rangos "típicos" (ajústalos si tu dataset difiere mucho)
N_ref, P_ref, K_ref = 90.0, 45.0, 45.0

# Potenciales aproximados por cultivo (t/ha) para que el yield no sea igual para todos
YMAX = {
    "rice": 6.0, "maize": 7.5, "cotton": 3.0, "jute": 2.6,
    "chickpea": 2.2, "kidneybeans": 2.4, "pigeonpeas": 2.1,
    "mothbeans": 1.6, "mungbean": 1.8, "blackgram": 1.9, "lentil": 2.0,
    "banana": 35.0, "mango": 12.0, "grapes": 14.0, "watermelon": 25.0,
    "muskmelon": 18.0, "apple": 20.0, "orange": 22.0, "papaya": 30.0,
    "pomegranate": 10.0, "coconut": 8.0, "coffee": 2.5
}

def _sat(x, x50):
    """Saturación suave en [0,1)"""
    x = max(float(x), 0.0)
    return x / (x + float(x50))

def compute_yield(row, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    crop = row["label"]
    opt = optimal_conditions[crop]

    # -----------------------
    # 1) Estrés ambiental
    # -----------------------
    dT = ((row["temperature"] - opt["temperature"]) / 6.0)**2
    dH = ((row["humidity"] - opt["humidity"]) / 10.0)**2
    dPH = ((row["ph"] - opt["ph"]) / 0.8)**2

    # Lluvia asimétrica + tolerancia por cultivo (frutas tropicales suelen tolerar más lluvia)
    rain_scale = 380.0 if crop in {"banana","coconut","rice","papaya"} else 320.0
    dr = (row["rainfall"] - opt["rainfall"]) / rain_scale
    dR = (dr**2) * (1.0 + 1.2 * (dr > 0))  # exceso castiga más

    env_stress = np.exp(-(dT + dH + dPH + dR))  # (0,1]

    # -----------------------
    # 2) Nutrientes (NPK)
    # -----------------------
    N, P, K = float(row["N"]), float(row["P"]), float(row["K"])

    # Efecto saturante individual
    fN = _sat(N, x50=0.8 * N_ref)
    fP = _sat(P, x50=0.8 * P_ref)
    fK = _sat(K, x50=0.8 * K_ref)

    # “Ley del mínimo” suave: el nutriente limitante domina
    limiting = min(fN, fP, fK)
    mean_fk = (fN + fP + fK) / 3.0
    nutrient_gain = 0.55 * limiting + 0.45 * mean_fk  # en [0,1)

    # Penalización por desbalance (ratio alejándose de 1)
    eps = 1e-6
    rNP = (N + eps) / (P + eps)
    rNK = (N + eps) / (K + eps)
    imbalance = (np.log(rNP)**2 + np.log(rNK)**2)  # 0 si balanceado
    balance_penalty = np.exp(-0.18 * imbalance)    # (0,1]

    # Penalización suave por sobre-fertilización total
    total = (N/N_ref + P/P_ref + K/K_ref)
    over_penalty = np.exp(-0.05 * max(0.0, total - 3.2)**2)

    nutrient_factor = (0.35 + 0.65 * nutrient_gain) * balance_penalty * over_penalty
    nutrient_factor = float(np.clip(nutrient_factor, 0.0, 1.1))

    # -----------------------
    # 3) Interacciones simples
    # -----------------------
    # Ejemplo: si llueve mucho y hay mucho N, aumenta riesgo (lixiviación / enfermedades) -> penaliza un poco
    interaction = 1.0
    if row["rainfall"] > opt["rainfall"] * 1.15 and N > 1.2 * N_ref:
        interaction *= 0.92

    # -----------------------
    # 4) Yield final + ruido heterocedástico
    # -----------------------
    y_max = YMAX.get(crop, 5.0)
    base = y_max * env_stress * nutrient_factor * interaction

    # ruido: más variabilidad cuando base es bajo
    sigma = 0.06 * y_max + 0.18 * (y_max - base)
    y = base + rng.normal(0.0, sigma)

    return float(np.clip(y, 0.0, None))

def generate_data():
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("ryandinh/agricultural-production-optimization")
    
    print("Path to dataset files:", path)

    df = pd.read_csv(os.path.join(path, "data (1).csv"))
    df["yield"] = df.apply(compute_yield, axis=1)

    return df

