# -*- coding: utf-8 -*-
import os

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Any

import numpy as np
from ase import Atoms
from pydantic import BaseModel, Field, validator
from rdkit import Chem




# Define transition_map dictionary
# Add more entries for other orbitals as needed
transition_map = {
    "C1s": {
        "element": "C",
        "orbital": "1s",
        "soap_filepath": os.path.abspath("xpsservice/SOAP_turbo_C1s.txt"),
        "model_filepath": os.path.abspath("xpsservice/XPS_GPR_C1s.pkl")
    },
    "O1s": {
        "element": "O",
        "orbital": "1s",
        "soap_filepath": os.path.abspath("xpsservice/SOAP_turbo_O1s.txt"),
        "model_filepath": os.path.abspath("xpsservice/XPS_GPR_O1s.pkl")
    },   
}


# Derive allowed elements from transition_map
def derive_allowed_elements(transition_map: dict) -> Set[str]:
    allowed_elements = {info["element"] for info in transition_map .values()}
    return allowed_elements

ALLOWED_ELEMENTS = derive_allowed_elements(transition_map)
ALLOWED_METHODS = ("GFNFF", "GFN2xTB", "GFN1xTB")
ALLOWED_FF = ("uff", "mmff94", "mmff94s")

#MM, bug: Not sure to be needed
class TransitionValidator(BaseModel):
    transition: str
    
    @validator("transition")
    def check_orbital(cls, value):
        if value not in transition_map:
            raise ValueError(f"Transition {value} is not allowed.")
        return value


@dataclass
class OptimizationResult:
    atoms: Atoms
    forces: np.ndarray
    energy: float


class IRResult(BaseModel):
    wavenumbers: List[float] = Field(None, description="List of wavenumbers in cm^-1")
    intensities: List[float] = Field(
        None, description="List of IR intensities in (D/Å)^2 amu^-1"
    )
    ramanIntensities: List[float] = Field(
        None,
        description="List of Raman intensities in (D/Å)^2 amu^-1, computed using Placzek and Bond Polarization (using values from Lippincott/Stuttman) approximation",
    )
    zeroPointEnergy: float = Field(None, description="Zero point energy in a.u.")
    modes: Optional[List[dict]] = Field(
        None,
        description="List of dictionaries with the keys `number` - number of the mode (zero indexed), `displacements` - xyz file with the displacement vectors, `intensity` - IR intensity of the mode in D/Å)^2 amu^-1, `ramanIntensity` - Raman intensity of mode, `imaginary` - true if mode is imaginary, `mostDisplaceAtoms` - sorted list of atom indices (zero indiced) according to they displacement (Euclidean norm), `mostContributingAtoms` - most contributing atoms according to a distance criterion.",
    )
    mostRelevantModesOfAtoms: Optional[Dict[int, List[int]]] = Field(
        None,
        description="Dictionary indexed with atom indices (zero indexed) and mode indices (zero indexed) as values that is most relevant for a given",
    )
    mostRelevantModesOfBonds: Optional[List[dict]] = Field(
        None,
        description="List of dictionaries with the key `startAtom`, `endAtom` and `mode`",
    )
    hasImaginaryFrequency: bool = Field(
        None, description="True if there is any mode with imaginary frequency"
    )
    isLinear: bool = Field(None, description="True if the molecule is linear.")
    momentsOfInertia: List[float] = Field(
        None,
        description="Moments of inertia around principal axes. For a linear molecule one only expects two non-zero components.",
    )
    hasLargeImaginaryFrequency: bool = Field(
        None,
        description="True if there is a large imaginary frequency, indicating a failed geometry optimization.",
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
        "GFN2xTB",
        description="String with method that is used for geometry optimization and calculation of the vibrational frequencies. Allowed values are `GFNFF`, `GFN2xTB`, and `GFN1xTB`. `GFNFF` is the computationally most inexpensive method, but can be less accurate than the xTB methods",
    )

    @validator("method")
    def method_match(cls, v):
        if not v in ALLOWED_METHODS:
            raise ValueError(f"method must be in {ALLOWED_METHODS}")
        return v


class ConformerRequest(BaseModel):
    smiles: Optional[str] = Field(
        None,
        description="SMILES string of input molecule. The service will add implicit hydrogens",
    )
    molFile: Optional[str] = Field(
        None,
        description="String with molfile with expanded hydrogens. The service will not attempt to add implicit hydrogens to ensure that the atom ordering is preserved.",
    )
    forceField: Optional[str] = Field(
        "uff",
        description="String with method force field that is used for energy minimization. Options are 'uff', 'mmff94', and 'mmff94s'",
    )
    rmsdThreshold: Optional[float] = Field(
        0.5, description="RMSD threshold that is used to prune conformer library."
    )
    maxConformers: Optional[int] = Field(
        1,
        description="Maximum number of conformers that are generated (after pruning).",
    )

    @validator("forceField")
    def method_match(cls, v):
        if not v in ALLOWED_FF:
            raise ValueError(f"forceField must be in {ALLOWED_FF}")
        return v


class Conformer(BaseModel):
    molFile: str = Field(
        None, description="String with molfile.",
    )
    energy: str = Field(
        None, description="Final energy after energy minimization.",
    )


class ConformerLibrary(BaseModel):
    conformers: List[Conformer]


#MM
#bug: add smiles and molfile
class XPSResult(BaseModel):
    molfile: str = Field(
        None, description = "Molfile (calculated or gived) used for the binding energy prediction"
    )
    smiles: Optional[str] = Field(
        None, description = "SMILES (if given) used for the binding energy prediction"
    )
    bindingEnergies: List[float] = Field(
        None, description="List of binding energies in eV"
    )
    standardDeviations: List[float] = Field(
        None, description="List of standard deviations of the binding energies in eV"
    )
    



# XPSRequest class supports accepting either a smiles string or a molFile string.
# When one of the fields (smiles or molFile) is provided, the other field is automatically
# converted and populated using the functions smiles2molfile and molfile2smiles.
# the inputs are checked using ALLOWED_ELEMENTS, and ALLOWED_METHODS
class XPSRequest(BaseModel):
    smiles: Optional[str] = Field(
        None,
        description="SMILES string of input molecule. The service will add implicit hydrogens",
    )
    molFile: Optional[str] = Field(
        None,
        description="String with molfile with expanded hydrogens. The service will not attempt to add implicit hydrogens to ensure that the atom ordering is preserved.",
    )
    method: Optional[str] = Field(
        "GFN2xTB",
        description="String with method that is used for geometry optimization and calculation of the XPS binding energies. Allowed values are `GFNFF`, `GFN2xTB`, and `GFN1xTB`. `GFNFF` is the computationally most inexpensive method, but can be less accurate than the xTB methods",
    )

    @validator("smiles")
    def validate_smiles(cls, v):
        if v:
            # Parse the SMILES string using RDKit
            mol = Chem.MolFromSmiles(v)
            if not mol:
                raise ValueError("Invalid SMILES string provided.")
            # Check if all elements in the SMILES string are within the allowed elements
            for atom in mol.GetAtoms():
                if atom.GetSymbol() not in ALLOWED_ELEMENTS:
                    raise ValueError(f"SMILES contains an invalid element: {atom.GetSymbol()}. Only {ALLOWED_ELEMENTS} are allowed.")
        return v

    @validator("molFile")
    def validate_molfile(cls, v):
        if v:
            # Parse the molfile string using RDKit
            mol = Chem.MolFromMolBlock(v)
            if not mol:
                raise ValueError("Invalid molfile provided.")
            # Check if all elements in the molfile are within the allowed elements
            for atom in mol.GetAtoms():
                if atom.GetSymbol() not in ALLOWED_ELEMENTS:
                    raise ValueError(f"Molfile contains an invalid element: {atom.GetSymbol()}. Only {ALLOWED_ELEMENTS} are allowed.")
        return v

    # Validator for the method field to ensure the method is within the allowed list of methods
    @validator("method")
    def validate_method(cls, v):
        if v not in ALLOWED_METHODS:
            raise ValueError(f"Method must be in {ALLOWED_METHODS}")
        return v

    