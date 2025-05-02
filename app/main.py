# tovV2/ml-service/app/main.py

from fastapi import FastAPI
from app.schema import InputData, DataServidor
import joblib
import pandas as pd
import os
from typing import Dict, Optional

app = FastAPI(title="TOV-R1 API")

# Directorio donde están los modelos serializados
MODEL_DIR = os.getenv("MODEL_DIR", "models")

# 1) Cargamos todos los tov_r1_*.pkl disponibles
available: Dict[str, joblib] = {}
for fn in os.listdir(MODEL_DIR):
    if fn.startswith("tov_r1_") and fn.endswith(".pkl"):
        key = fn[len("tov_r1_"):-len(".pkl")]
        available[key] = joblib.load(os.path.join(MODEL_DIR, fn))


def predict_sub(name: str, df: pd.DataFrame) -> Optional[float]:
    """
    Si existe submodelo available[name], devuelve prob*100 redondeado a 0.1,
    si no, devuelve None para indicar que hay que hacer fallback.
    """
    m = available.get(name)
    if m:
        return round(m.predict_proba(df)[0][1] * 100, 1)
    return None


@app.post("/predict", response_model=DataServidor)
async def predict(payload: InputData):
    # 2) Transformamos el input en un DataFrame de una fila
    df = pd.DataFrame([payload.dict()])

    # 3) Predicción global
    vg_pct = predict_sub("vg", df) or 0.0
    siYnoVg = {"si": vg_pct, "no": round(100 - vg_pct, 1)}

    # 4) Tipos de violencia (7 categorías)
    tipo_keys = ["fisica","psicologica","sexual","economica","patrimonial","social","vicaria"]
    uniform_tipo = round(vg_pct / len(tipo_keys), 1) if vg_pct else round(100 / len(tipo_keys), 1)
    tiposDeViolencia = {}
    for k in tipo_keys:
        val = predict_sub(f"tipo__{k}", df)
        tiposDeViolencia[k] = val if val is not None else uniform_tipo

    # 5) Frecuencia (5 categorías)
    freq_keys = ["siempre","casisiempre","puntomedio","casinunca","nunca"]
    uniform_freq = round(100 / len(freq_keys), 1)
    frecuencia = {}
    for k in freq_keys:
        val = predict_sub(f"frecuencia__{k}", df)
        frecuencia[k] = val if val is not None else uniform_freq

    # 6) Denuncias y apoyo (sí/no)
    d = predict_sub("denuncia", df)
    default_dn = 50.0
    si = d if d is not None else default_dn
    siYnoCd = {"si": si, "no": round(100 - si, 1)}

    a = predict_sub("apoyo", df)
    default_ap = 50.0
    ap = a if a is not None else default_ap
    siYnoApoyoU = {"si": ap, "no": round(100 - ap, 1)}

    # 7) Percepción (5 categorías)
    perc_keys = ["muyBuena","buena","regular","mala","muyMala"]
    uniform_perc = round(100 / len(perc_keys), 1)
    percepcion = {}
    for k in perc_keys:
        val = predict_sub(f"percepcion__{k}", df)
        percepcion[k] = val if val is not None else uniform_perc

    # 8) Semestre (10 categorías)
    sem_keys = ["primero","segundo","tercero","cuarto","quinto","sexto","septimo","octavo","noveno","decimo"]
    uniform_sem = round(100 / len(sem_keys), 1)
    semestre = {}
    for k in sem_keys:
        val = predict_sub(f"semestre__{k}", df)
        semestre[k] = val if val is not None else uniform_sem

    # 9) Programa (5 categorías)
    prog_keys = ["Derecho","ContaduriaPublica","Psicologia","IngenieriaSistemas","AdministracionEmpresas"]
    uniform_prog = round(100 / len(prog_keys), 1)
    programas = {k: (predict_sub(f"programa__{k}", df) or uniform_prog) for k in prog_keys}

    # 10) Roles (4 categorías)
    role_keys = ["Estudiante","Administrativx","Docente","Externo"]
    uniform_role = round(100 / len(role_keys), 1)
    roles = {k: (predict_sub(f"rol__{k}", df) or uniform_role) for k in role_keys}

    # 11) Rango de edad (4 categorías)
    age_keys = ["menores18","entre18y25","entre26y40","mayores40"]
    uniform_age = round(100 / len(age_keys), 1)
    rangoEdad = {k: (predict_sub(f"rangoEdad__{k}", df) or uniform_age) for k in age_keys}

    # 12) Sexo (3 categorías)
    sex_keys = ["macho","hembra","intersexual"]
    uniform_sex = round(100 / len(sex_keys), 1)
    sexos = {k: (predict_sub(f"sexo__{k}", df) or uniform_sex) for k in sex_keys}

    # 13) Orientación (5 categorías)
    oriSex_keys = ["hetero","gay","lesbiana","bisexual","otra"]
    uniform_ori = round(100 / len(oriSex_keys), 1)
    orientacionSexual = {k: (predict_sub(f"orientacion__{k}", df) or uniform_ori) for k in oriSex_keys}

    # 14) Identidad de género (3 categorías)
    idg_keys = ["hombre","mujer","otra"]
    uniform_idg = round(100 / len(idg_keys), 1)
    identidadDeGenero = {k: (predict_sub(f"identidad__{k}", df) or uniform_idg) for k in idg_keys}

    # 15) Discapacidades (5 categorías)
    dis_keys = ["fisicas","sensoriales","intelectuales","psicosociales","multiples"]
    uniform_dis = round(100 / len(dis_keys), 1)
    discapacidades = {k: (predict_sub(f"discapacidad__{k}", df) or uniform_dis) for k in dis_keys}

    # 16) Etnias (5 categorías)
    etn_keys = ["indigena","afrocolombianos","raizales","gitanos","ninguna"]
    uniform_etn = round(100 / len(etn_keys), 1)
    etnias = {k: (predict_sub(f"etnia__{k}", df) or uniform_etn) for k in etn_keys}

    # 17) Religiones (6 categorías)
    rel_keys = ["catolicismo","evangelismo","agnosticismo","ateismo","cristianismo","otra"]
    uniform_rel = round(100 / len(rel_keys), 1)
    religiones = {k: (predict_sub(f"religion__{k}", df) or uniform_rel) for k in rel_keys}

    # 18) Estado civil (5 categorías)
    ec_keys = ["soltero","casado","unionLibre","divorciado","viudo"]
    uniform_ec = round(100 / len(ec_keys), 1)
    estadoCivil = {k: (predict_sub(f"estadoCivil__{k}", df) or uniform_ec) for k in ec_keys}

    # 19) Origen (4 categorías)
    orig_keys = ["municipioLocal","otroMunicipio","otroDepartamento","otroPais"]
    uniform_orig = round(100 / len(orig_keys), 1)
    origen = {k: (predict_sub(f"origen__{k}", df) or uniform_orig) for k in orig_keys}

    # 20) Estrato (6 categorías)
    estrato_keys = [str(i) for i in range(1,7)]
    uniform_est = round(100 / len(estrato_keys), 1)
    estrato = {k: (predict_sub(f"estrato__{k}", df) or uniform_est) for k in estrato_keys}

    # 21) Devolvemos el JSON final
    return {
        "siYnoVg":           siYnoVg,
        "tiposDeViolencia":  tiposDeViolencia,
        "frecuencia":        frecuencia,
        "siYnoCd":           siYnoCd,
        "siYnoApoyoU":       siYnoApoyoU,
        "percepcion":        percepcion,
        "semestre":          semestre,
        "programas":         programas,
        "roles":             roles,
        "rangoEdad":         rangoEdad,
        "sexos":             sexos,
        "orientacionSexual": orientacionSexual,
        "identidadDeGenero": identidadDeGenero,
        "discapacidades":    discapacidades,
        "etnias":            etnias,
        "religiones":        religiones,
        "estadoCivil":       estadoCivil,
        "origen":            origen,
        "estrato":           estrato,
    }
