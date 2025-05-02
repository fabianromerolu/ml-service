# ml-service/train.py actualizado

import os
import sys
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

DATA_PATH = os.getenv("DATA_PATH", "data/data.csv")
MODEL_DIR = os.getenv("MODEL_DIR", "models")

def load_data(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        print(f"❌ ERROR: archivo de datos no encontrado en {path}", file=sys.stderr)
        sys.exit(1)
    return pd.read_csv(path)

def main():
    print("🔍 Cargando datos...")
    df = load_data(DATA_PATH)

    # Columnas numéricas y categóricas para predecir
    num_cols = ["edad", "estrato"]
    cat_cols = ["departamento", "municipio", "universidad", "semestre", "programa",
                "rol", "sexo", "orientacion", "identidad", "discapacidad",
                "etnia", "religion", "estado_civil", "origen"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    target_cols = [col for col in df.columns if col.startswith("target_")]
    print(f"✔️  Columnas objetivo detectadas: {target_cols}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    for col in target_cols:
        target_name = col.replace("target_", "")
        unique_vals = df[col].unique()

        # Caso Binario (0/1)
        if sorted(unique_vals) == [0, 1]:
            print(f"\n🔀 Entrenando modelo binario '{target_name}' (target = {col})")
            X_train, X_test, y_train, y_test = train_test_split(
                df.drop(columns=target_cols), df[col], test_size=0.2, random_state=42, stratify=df[col]
            )

            model = Pipeline([
                ("prep", preprocessor),
                ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
            ])

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            aucs = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
            print(f"   ▶️ CV AUC: {aucs.mean():.4f} ± {aucs.std():.4f}")

            model.fit(X_train, y_train)
            test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            print(f"   ▶️ Test AUC: {test_auc:.4f}")

            joblib.dump(model, f"{MODEL_DIR}/tov_r1_{target_name}.pkl")
            print(f"✅ Guardado: {MODEL_DIR}/tov_r1_{target_name}.pkl")

        # Caso Multi-clase (One-Hot Encoding)
        else:
            print(f"\n🔀 Entrenando sub-modelos para variable categórica '{target_name}'")
            for val in unique_vals:
                binary_target = df[col].apply(lambda x: 1 if x == val else 0)
                binary_target_name = f"{target_name}__{val}"

                print(f" 🔹 Categoría '{val}' ({binary_target_name})")
                X_train, X_test, y_train, y_test = train_test_split(
                    df.drop(columns=target_cols), binary_target, test_size=0.2, random_state=42, stratify=binary_target
                )

                model = Pipeline([
                    ("prep", preprocessor),
                    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
                ])

                aucs = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
                print(f"     ▶️ CV AUC: {aucs.mean():.4f} ± {aucs.std():.4f}")

                model.fit(X_train, y_train)
                test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                print(f"     ▶️ Test AUC: {test_auc:.4f}")

                joblib.dump(model, f"{MODEL_DIR}/tov_r1_{binary_target_name}.pkl")
                print(f"   ✅ Guardado: {MODEL_DIR}/tov_r1_{binary_target_name}.pkl")

if __name__ == "__main__":
    main()
