import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

np.random.seed(42)

# OPCIONES
# genero:       0=hombre, 1=mujer, 2=prefiero_no_decirlo
# edad:         0=16-29, 1=30-39, 2=40-49, 3=50+
# objetivo:     0=volumen, 1=definicion, 2=mantenimiento, 3=perdida_peso
# nivel:        0=principiante, 1=intermedio, 2=avanzado
# dias:         2, 3, 4, 5
# lesion:       0=ninguna, 1=espalda, 2=rodilla, 3=hombro, 4=tobillo
# pref:         0=pesas_fuerza, 1=cardio_resistencia, 2=mixto
# peso_cat:     0=menos60kg, 1=60-75kg, 2=76-90kg, 3=91-110kg, 4=mas110kg

# RECOMENDACIONES
# 0 → Rutina Full Body Básica       + Dieta Volumen Moderado
# 1 → Rutina Hipertrofia 3 días     + Dieta Volumen Alto
# 2 → Rutina Hipertrofia 4-5 días   + Dieta Volumen Avanzado
# 3 → Rutina Cardio + Fuerza        + Dieta Definición Moderada
# 4 → Rutina HIIT + Pesas           + Dieta Definición Estricta
# 5 → Rutina Mantenimiento Activo   + Dieta Equilibrada
# 6 → Rutina Bajo Impacto           + Dieta Pérdida de Peso Suave
# 7 → Rutina Adaptada por Lesión    + Dieta según Objetivo

RECOMENDACIONES = {
    0: "Rutina Full Body Básica + Dieta Volumen Moderado",
    1: "Rutina Hipertrofia 3 días + Dieta Volumen Alto",
    2: "Rutina Hipertrofia 4-5 días + Dieta Volumen Avanzado",
    3: "Rutina Cardio + Fuerza + Dieta Definición Moderada",
    4: "Rutina HIIT + Pesas + Dieta Definición Estricta",
    5: "Rutina Mantenimiento Activo + Dieta Equilibrada",
    6: "Rutina Bajo Impacto + Dieta Pérdida de Peso Suave",
    7: "Rutina Adaptada por Lesión + Dieta según Objetivo",
}

def asignar_recomendacion(genero, edad, objetivo, nivel, dias, lesion, pref, peso_cat):
    if lesion != 0:
        return 7

    if edad == 3:
        if objetivo == 1:
            return 3
        elif objetivo == 0:
            return 0
        else:
            return 6

    
    if peso_cat == 4:
        if objetivo == 3 or objetivo == 1:
            return 6  
        elif objetivo == 2:
            return 6
        else:
            return 0  

    
    if peso_cat == 3:
        if objetivo == 3:
            return 6  
        elif objetivo == 1:
            if dias >= 4:
                return 3  
            else:
                return 3

    
    if peso_cat == 0:
        if objetivo == 0:
            if nivel == 0:
                return 1 
            else:
                return 2  
        elif objetivo == 3:
            
            return 5

    if objetivo == 0:  
        if nivel == 0:
            return 0
        elif nivel == 1:
            if edad <= 1:
                return 1
            else:
                return 0
        else:  # Avanzado
            if edad == 0:
                return 2
            elif edad == 1:
                return 1
            else:
                return 0

    elif objetivo == 1:  
        if edad <= 1:
            if dias >= 4:
                return 4  
            else:
                return 3
        else:
            return 3

    elif objetivo == 2:  
        if edad <= 1:
            return 5
        else:
            return 6

    elif objetivo == 3:  
        return 6

    return 5 


N = 5000  
datos = []

for _ in range(N):
    genero   = np.random.choice([0, 1, 2], p=[0.45, 0.45, 0.10])
    edad     = np.random.choice([0, 1, 2, 3], p=[0.35, 0.30, 0.20, 0.15])
    objetivo = np.random.randint(0, 4)
    nivel    = np.random.randint(0, 3)
    dias     = np.random.choice([2, 3, 4, 5])
    lesion   = np.random.choice([0, 0, 0, 1, 2, 3, 4])
    pref     = np.random.randint(0, 3)
    

    peso_cat = np.random.choice([0, 1, 2, 3, 4], p=[0.10, 0.35, 0.30, 0.18, 0.07])

    recomendacion = asignar_recomendacion(
        genero, edad, objetivo, nivel, dias, lesion, pref, peso_cat
    )

    if np.random.random() < 0.04:
        recomendacion = np.random.randint(0, 8)

    datos.append({
        "genero":        genero,
        "edad":          edad,
        "objetivo":      objetivo,
        "nivel":         nivel,
        "dias":          dias,
        "lesion":        lesion,
        "pref":          pref,
        "peso_cat":      peso_cat,
        "recomendacion": recomendacion
    })

df = pd.DataFrame(datos)

print("=== DATASET GENERADO ===")
print(f"Total filas: {len(df)}")
print(f"\nDistribución de recomendaciones:")
for k, v in df["recomendacion"].value_counts().sort_index().items():
    print(f"  [{k}] {RECOMENDACIONES[k]}: {v} ejemplos")

X = df[["genero", "edad", "objetivo", "nivel", "dias", "lesion", "pref", "peso_cat"]]
y = df["recomendacion"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

modelo = RandomForestClassifier(
    n_estimators=150,  
    max_depth=12,
    random_state=42
)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nRESULTADOS DEL MODELO")
print(f"Accuracy: {acc * 100:.2f}%")
print(f"\nReporte detallado:")
print(classification_report(y_test, y_pred))

print("\nIMPORTANCIA DE VARIABLES")
importancias = pd.Series(
    modelo.feature_importances_,
    index=["genero", "edad", "objetivo", "nivel", "dias", "lesion", "pref", "peso_cat"]
).sort_values(ascending=False)
for var, imp in importancias.items():
    print(f"  {var}: {imp:.4f}")

joblib.dump(modelo, "modelo_powertrack_v3.pkl")
df.to_csv("dataset_powertrack_v3.csv", index=False)

print("\nModelo guardado: modelo_powertrack_v3.pkl")
print("Dataset guardado: dataset_powertrack_v3.csv")

print("\nPRUEBAS MANUALES CON PESO")
casos = [
    {"genero": 0, "edad": 0, "objetivo": 0, "nivel": 2, "dias": 5, "lesion": 0, "pref": 0, "peso_cat": 1},
    {"genero": 1, "edad": 0, "objetivo": 0, "nivel": 2, "dias": 5, "lesion": 0, "pref": 0, "peso_cat": 1},
    {"genero": 0, "edad": 2, "objetivo": 0, "nivel": 2, "dias": 5, "lesion": 0, "pref": 0, "peso_cat": 1},
    {"genero": 0, "edad": 3, "objetivo": 0, "nivel": 2, "dias": 5, "lesion": 0, "pref": 0, "peso_cat": 1},
    {"genero": 1, "edad": 1, "objetivo": 1, "nivel": 1, "dias": 5, "lesion": 0, "pref": 1, "peso_cat": 1},
    {"genero": 0, "edad": 2, "objetivo": 1, "nivel": 1, "dias": 5, "lesion": 0, "pref": 1, "peso_cat": 1},
    {"genero": 1, "edad": 1, "objetivo": 2, "nivel": 1, "dias": 3, "lesion": 0, "pref": 2, "peso_cat": 1},
    {"genero": 0, "edad": 0, "objetivo": 0, "nivel": 1, "dias": 3, "lesion": 2, "pref": 0, "peso_cat": 1},

    {"genero": 0, "edad": 0, "objetivo": 0, "nivel": 1, "dias": 4, "lesion": 0, "pref": 0, "peso_cat": 0},
    {"genero": 1, "edad": 1, "objetivo": 3, "nivel": 0, "dias": 3, "lesion": 0, "pref": 1, "peso_cat": 3},
    {"genero": 0, "edad": 2, "objetivo": 1, "nivel": 1, "dias": 4, "lesion": 0, "pref": 2, "peso_cat": 4},
    {"genero": 0, "edad": 0, "objetivo": 3, "nivel": 0, "dias": 3, "lesion": 0, "pref": 1, "peso_cat": 0},
]

descripciones = [
    "Hombre joven avanzado volumen peso normal → espera: 2",
    "Mujer joven avanzada volumen peso normal → espera: 2",
    "Hombre 40-49 avanzado volumen peso normal → espera: 0",
    "Hombre 50+ avanzado volumen peso normal → espera: 0",
    "Mujer 30-39 definición 5 días peso normal → espera: 4",
    "Hombre 40-49 definición 5 días peso normal → espera: 3",
    "Mujer 30-39 mantenimiento peso normal → espera: 5",
    "Lesión rodilla peso normal → espera: 7",
    "Hombre joven bajo peso quiere volumen → espera: 2 (superávit agresivo)",
    "Mujer 30-39 sobrepeso quiere perder peso → espera: 6 (bajo impacto)",
    "Hombre 40-49 obesidad quiere definición → espera: 6 (bajo impacto)",
    "Hombre joven bajo peso quiere perder peso → espera: 5 (redirigir a mantenimiento)",
]

for i, (caso, desc) in enumerate(zip(casos, descripciones)):
    entrada = pd.DataFrame([caso])
    pred = modelo.predict(entrada)[0]
    esperado = int(desc.split("espera: ")[1][0])
    correcto = "biennn" if pred == esperado else "maal"
    print(f"{correcto} Caso {i+1}: {desc}")
    print(f"   Recomendación [{pred}]: {RECOMENDACIONES[pred]}\n")