from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(
    title="PowerTrack IA",
    description="Microservicio de recomendación personalizada de entrenamientos",
    version="1.0.0"
)

try:
    modelo = joblib.load("modelo_powertrack_v2.pkl")
except Exception as e:
    raise RuntimeError(f"Error al cargar el modelo: {str(e)}")


class PerfilUsuario(BaseModel):
    genero: int = Field(..., ge=0, le=2, description="0=hombre, 1=mujer, 2=prefiero_no_decirlo")
    edad: int = Field(..., ge=0, le=3, description="0=16-29, 1=30-39, 2=40-49, 3=50+")
    objetivo: int = Field(..., ge=0, le=3, description="0=volumen, 1=definicion, 2=mantenimiento, 3=perdida_peso")
    nivel: int = Field(..., ge=0, le=2, description="0=principiante, 1=intermedio, 2=avanzado")
    dias: int = Field(..., ge=2, le=5, description="Días de entrenamiento por semana")
    lesion: int = Field(..., ge=0, le=4, description="0=ninguna, 1=espalda, 2=rodilla, 3=hombro, 4=tobillo")
    pref: int = Field(..., ge=0, le=2, description="0=pesas_fuerza, 1=cardio_resistencia, 2=mixto")


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


@app.post("/predict")
def predecir(perfil: PerfilUsuario):
    try:
        datos = pd.DataFrame([{
            "genero":   perfil.genero,
            "edad":     perfil.edad,
            "objetivo": perfil.objetivo,
            "nivel":    perfil.nivel,
            "dias":     perfil.dias,
            "lesion":   perfil.lesion,
            "pref":     perfil.pref
        }])
        recomendacion = int(modelo.predict(datos)[0])
        return {
            "recomendacion": recomendacion,
            "descripcion": RECOMENDACIONES[recomendacion]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {str(e)}")


@app.get("/health")
def health():
    return {"status": "ok", "modelo": "modelo_powertrack_v2.pkl"}