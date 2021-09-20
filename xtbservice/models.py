from dataclasses import dataclass
import numpy as np
from pydantic import BaseModel, validator, Field
from typing import Optional, List, Dict
from ase import Atoms

ALLOWED_METHODS = ("GFNFF", "GFN2xTB", "GFN1xTB")


@dataclass
class OptimizationResult:
    atoms: Atoms
    forces: np.ndarray
    energy: float


class IRResult(BaseModel):
    wavenumbers: List[float] = Field(None, description="List of wavenumbers in cm^-1")
    intensities: List[float] = Field(
        None, description="List of intensities in (D/Å)^2 amu^-1"
    )
    zeroPointEnergy: float = Field(None, description="Zero point energy in a.u.")
    modes: Optional[List[dict]] = Field(
        None,
        description="List of dictionaries with the keys `number` - number of the mode (zero indexed), `displacements` - xyz file with the displacement vectors, `intensity` intensity of the mode in D/Å)^2 amu^-1, `imaginary` - true if mode is imaginary, `mostDisplaceAtoms` - sorted list of atom indices (zero indiced) according to they displacement (Euclidean norm), `mostContributingAtoms` - most contributing atoms according to a distance criterion.",
    )
    mostRelevantModesOfAtoms: Optional[Dict[int, List[int]]] = Field(
        None,
        description="Dictionary indexed with atom indices (zero indexed) and mode indices (zero indexed) as values that is most relevant for a given",
    )
    hasImaginaryFrequency: bool = Field(
        None, description="True if there is any mode with imaginary frequency"
    )


class IRRequest(BaseModel):
    smiles: Optional[str] = Field(
        None,
        description="SMILES string of input molecule. The service will add implicit hydrogens",
    )
    molFile: Optional[str] = Field(
        None,
        description="String with molfile with expanded hydrogens. The service will not attempt to add implicit hydrogens to ensure that the atom ordering is preserved.",
    )
    method: Optional[str] = Field(
        "GFNFF",
        description="String with method that is used for geometry optimization and calculation of the vibrational frequencies. Allowed values are `GFNFF`, `GFN2xTB`, and `GFN1xTB`. `GFNFF` is the computationally most inexpensive method, but can be less accurate than the xTB methods",
    )

    @validator("method")
    def method_match(cls, v):
        if not v in ALLOWED_METHODS:
            raise ValueError(f"method must be in {ALLOWED_METHODS}")
        return v

