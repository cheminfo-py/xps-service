from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional

class MolfileRequest(BaseModel):
    molfile:str
    sigma:float = Field(1.3, description="Sigma used for Gaussians")
    
class Molfile(BaseModel):
    molfile:str

class SMILES(BaseModel):
    smiles:str
    
#class Sigma(BaseModel):
#    sigma:float    

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
    orbital: str
    #soapTurbo: soap
    prediction: bindingEnergyPrediction

class SpectrumData(BaseModel):
    x: SpectralData
    y: SpectralData
    
class PredictedXPSSpectrum(BaseModel):
    allBindingEnergies: List[float]
    gaussian: SpectrumData
    sigma: float = Field(None, description= "Sigma used for Gaussians")

class SpectralPrediction(BaseModel):
    molfile: str
    smiles: str = Field(None, description = "SMILES of structure (if given)")
    elementsIncluded: List
    elementsExcluded: List
    bindingEnergies: List
    spectrum: PredictedXPSSpectrum
    
    
class BEPrediction(BaseModel):
    molfile: str
    smiles: str = Field(None, description = "SMILES of structure (if given)")
    elementsIncluded: List
    elementsExcluded: List
    bindingEnergies: List