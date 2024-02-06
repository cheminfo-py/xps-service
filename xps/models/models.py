from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional

class Spectrum(BaseModel):
    label: str = Field(None, description="Label of axis")
    data: List[float] = Field(None, description="Data of axis")
    units: str = Field(None, description="Unit of axis")

class MolfileRequest(BaseModel):
    molfile: str


class SpectrumData(BaseModel):
    x: Spectrum
    y: Spectrum

class XPS_prediction(BaseModel):
    energies: List[float] = Field(None, description="List of binding energies in eV")
    
    intensities: List[float] = Field(None, description="List of intentsities")

    sigma: float = Field(None, description= "Sigma used for Gaussians")
    

class XPS_BE(BaseModel):
    BindingEnergies: Spectrum



class SpectrumModel(BaseModel):
    spectrum: SpectrumData
    sigma: float = Field(None, description= "Sigma used for Gaussians")