from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional

class MolfileRequest(BaseModel):
    molfile:str

class soap(BaseModel):
    element: str
    descriptor: str
    data: List

class bindingEnergyPrediction(BaseModel):
    modelFile: str
    data: List
    standardDeviation: List

class SpectralData(BaseModel):
    label: str = Field(None, description="Label of axis")
    data: List[float] = Field(None, description="Data of axis")
    units: str = Field(None, description="Unit of axis")

class ModelPrediction(BaseModel):
    element: str
    #soapTurbo: soap
    prediction: bindingEnergyPrediction

class SpectrumData(BaseModel):
    x: SpectralData
    y: SpectralData
    
class PredictedXPSSpectrum(BaseModel):
    bindingEnergies: List[float]
    gaussian: SpectrumData
    sigma: float = Field(None, description= "Sigma used for Gaussians")

class FullPrediction(BaseModel):
    molfile: str
    bindingEnergies: List
    spectrum: PredictedXPSSpectrum

