# ml-service/app/schema.py

from pydantic import BaseModel
from typing import Dict

class InputData(BaseModel):
    departamento: str
    municipio: str
    universidad: str
    semestre: str
    programa: str
    rol: str
    edad: int
    sexo: str
    orientacion: str
    identidad: str
    discapacidad: str
    etnia: str
    religion: str
    estado_civil: str
    origen: str
    estrato: int

class DataServidor(BaseModel):
    siYnoVg:            Dict[str, float]
    tiposDeViolencia:   Dict[str, float]
    frecuencia:         Dict[str, float]
    siYnoCd:            Dict[str, float]
    siYnoApoyoU:        Dict[str, float]
    percepcion:         Dict[str, float]
    semestre:           Dict[str, float]
    programas:          Dict[str, float]
    roles:              Dict[str, float]
    rangoEdad:          Dict[str, float]
    sexos:              Dict[str, float]
    orientacionSexual:  Dict[str, float]
    identidadDeGenero:  Dict[str, float]
    discapacidades:     Dict[str, float]
    etnias:             Dict[str, float]
    religiones:         Dict[str, float]
    estadoCivil:        Dict[str, float]
    origen:             Dict[str, float]
    estrato:            Dict[str, float]
