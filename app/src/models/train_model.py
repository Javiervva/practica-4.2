import time

import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app import ROOT_DIR
from app.src.data.train.make_dataset import make_dataset
from app.src.evaluation.evaluate_model import evaluate_model
from app.src.utils.utils import load_model_config


def training_pipeline():
    """
    Función para gestionar el pipeline completo de entrenamiento
    del modelo.

    Args:
        path (str):  Ruta hacia los datos.

    Kwargs:
        model_info_db_name (str):  base de datos a usar para almacenar
        la info del modelo.
    """

    # Carga de la configuración de entrenamiento
    model_config = load_model_config()
    # variable dependiente a usar
    target = model_config["target"]
    # columnas a retirar
    cols_to_remove = model_config["cols_to_remove"]

    # timestamp usado para versionar el modelo y los objetos
    ts = time.time()

    # carga y transformación de los datos de train y test
    train_df, test_df = make_dataset(target, cols_to_remove)

    # separación de variables independientes y dependiente
    y_train = train_df[target]
    X_train = train_df.drop(columns=[target]).copy()
    y_test = test_df[target]
    X_test = test_df.drop(columns=[target]).copy()

    # definición de los transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # definición del modelo (Random Forest)
    model = RandomForestClassifier(
        n_estimators=model_config["n_estimators"],
        max_features=model_config["max_features"],
        random_state=50,
        n_jobs=-1,
    )

    # Creación de la pipeline
    pipeline = Pipeline([
        ('preprocessor', numeric_transformer),
        ('model', model)
    ])

    # Definición de los parámetros para el GridSearchCV
    params = {
        'model__n_estimators': [model_config['n_estimators']],
        'model__max_features': [model_config['max_features']],
        'model__min_samples_split': [model_config['min_samples_split']],
        'model__min_samples_leaf': [model_config['min_samples_leaf']],
        'model__max_depth': [model_config['max_depth']]
    }

    print("---> Training a model with the following configuration:")
    # Se inicia el RUN (ejecución del entrenamiento de un modelo)
    with mlflow.start_run(run_name=f"{model_config['model_name']}_{str(ts)}") as run:
        # Se crea el objeto GridSearchCV
        grid_search = GridSearchCV(pipeline, params, cv=5)
        # Ajuste del modelo con los datos de entrenamiento
        grid_search.fit(X_train, y_train)

        # Guardar el mejor modelo
        best_model = grid_search.best_estimator_

        print("------> Logging metadata in MLFlow")
        # se registran los parámetros del modelo
        mlflow.log_param("n_estimators", grid_search.best_params_['model__n_estimators'])
        mlflow.log_param("max_features", grid_search.best_params_['model__max_features'])
        mlflow.log_param("target", target)
        mlflow.log_param("cols_to_remove", cols_to_remove)

        # Evaluación del modelo utilizando los datos de prueba
        evaluate_model(best_model, X_test, y_test, ts, model_config["model_name"])
        # guardado del modelo y artifacts en MLFlow
        print(
            f"------> Saving the model {model_config['model_name']}_{str(ts)} and artifacts in MLFlow"
        )
        save_model(best_model)


def save_model(model):
    """
    Función para registrar el modelo en MLFlow

    Args:
        model: Objeto del mejor modelo entrenado.
    """

    mlflow.log_artifact(
        f"{ROOT_DIR}/models/objects/encoded_columns.pkl",
        artifact_path="model",
    )
    mlflow.log_artifact(f"{ROOT_DIR}/models/objects/imputer.pkl", artifact_path="model")

    mlflow.sklearn.log_model(
        sk_model=model,  # Aquí "model" es el mejor modelo
        artifact_path="model",
    )
