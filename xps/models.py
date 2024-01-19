from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional


class XPS_prediction(BaseModel):
    energies: List[float] = Field(None, description="List of binding energies in eV")
    
    intensities: List[float] = Field(None, description="List of intentsities")

class XPS_BE(BaseModel):
    BE: List[float] = Field(None, description="List of binding energies in eV")