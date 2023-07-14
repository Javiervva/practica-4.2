import os
import warnings

from flask import Flask, request

from app.src.models import train_model
from app.src.models.predict import predict_pipeline
from app.src.utils import database_utils

# Quitar warnings innecesarios de la salida
warnings.filterwarnings("ignore")

# inicializar la app bajo el framework Flask
app = Flask(__name__)
port = int(os.getenv("PORT", 8080))


# usando el decorador @app.route para gestionar los enrutadores (Método GET)
# ruta ráiz "/"
@app.route("/", methods=["GET"])
def root():
    """
    Función para gestionar la salida de la ruta raíz.

    Returns:
       dict.  Mensaje de salida
    """
    # No hacemos nada. Solo devolvemos info (customizable a voluntad)
    return {"Proyecto": "Mod. 4 - Ciclo de vida de modelos IA"}


@app.route("/load-dataset", methods=["POST"])
def load_dataset():
    """
    Endpoint para cargar un dataset desde un archivo CSV a una base de datos SQLite.

    Returns:
       dict.  Mensaje de confirmación.
    """
    # Cargar los datos en la base de datos SQLite
    total_records = database_utils.load_dataset_to_db()

    return {
        "message": f"Datos cargados en la base de datos SQLite exitosamente. Total registros cargados: {total_records}"}


@app.route("/train-model", methods=["GET"])
def train_model_route():
    """
    Función de lanzamiento del pipeline de entrenamiento.

    Returns:
       dict.  Mensaje de salida
    """

    # Comprobar si hay datos en la base de datos
    total_records = database_utils.check_data_in_db()

    # Si no hay datos, devolver un mensaje de error
    if total_records == 0:
        return {"Error": "No se pueden entrenar el modelo. No se han cargado datos."}

    # Lanzar el pipeline de entranamiento de nuestro modelo
    train_model.training_pipeline()

    # Se puede devolver lo que queramos (mensaje de éxito en el entrenamiento, métricas, etc.)
    return {"TRAINING MODEL": "Mod. 4 - Ciclo de vida de modelos IA"}


# ruta para el lanzar el pipeline de inferencia (Método POST)
@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Función de lanzamiento del pipeline de inferencia.

    Returns:
       dict.  Mensaje de salida (predicción)
    """

    # Obtener los datos pasados por el request
    data = request.get_json()

    # Lanzar la ejecución del pipeline de inferencia
    y_pred = predict_pipeline(data)

    return {"Predicted value": y_pred}


# main
if __name__ == "__main__":
    # ejecución de la app
    app.run(host="0.0.0.0", port=port, debug=True)
