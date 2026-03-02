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

def compute_yield(row):
    opt = optimal_conditions[row["label"]]

    d = (
        ((row["temperature"] - opt["temperature"]) / 8)**2 +
        ((row["humidity"] - opt["humidity"]) / 15)**2 +
        ((row["ph"] - opt["ph"]) / 1.2)**2 +
        ((row["rainfall"] - opt["rainfall"]) / 400)**2
    )

    # peak yield ~5 tons/hectare
    y = 5 * np.exp(-d)

    # environmental randomness
    y += np.random.normal(0, 0.25)

    return max(y, 0)

def generate_data():
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("ryandinh/agricultural-production-optimization")
    
    print("Path to dataset files:", path)

    df = pd.read_csv(os.path.join(path, "data (1).csv"))
    df["yield"] = df.apply(compute_yield, axis=1)

    return df

