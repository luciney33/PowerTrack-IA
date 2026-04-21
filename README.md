# PowerTrack IA

Microservicio de inteligencia artificial para la app PowerTrack

## Descripción
Modelo RandomForest entrenado con 3.500 registros simulados que clasifica
al usuario en uno de 8 perfiles de entrenamiento y nutrición basándose
en 7 variables de su perfil físico

## Cómo ejecutar
pip3 install -r requirements.txt

uvicorn main:app --reload

## Endpoint principal
POST http://localhost:8000/predict