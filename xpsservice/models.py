# -*- coding: utf-8 -*-
import os

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Any

import numpy as np
from ase import Atoms
from pydantic import BaseModel, Field, validator
from rdkit import Chem

from .settings import ALLOWED_FMAX, ALLOWED_ELEMENTS, ALLOWED_FF, ALLOWED_METHODS, transition_map


#MM, bug: Not sure to be needed
class TransitionValidator(BaseModel):
    transition: str
    
    @validator("transition")
    def check_orbital(cls, value):
        if value not in transition_map:
            raise ValueError(f"Transition {value} is not allowed.")
        return value
    


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
    fmax: Optional[float] = Field(
        0.01,
        description="Maximum force admissible during the geometry optimization process using either of the selected method. Typically ranging from 0.1 to 0.0001 "
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
    
    # Validator for the fmax field to ensure the value is within the allowed range
    @validator("fmax")
    def validate_fmax(cls, v):
        if not (ALLOWED_FMAX[0] <= v <= ALLOWED_FMAX[1]):
            raise ValueError(f"fmax must be within the range {ALLOWED_FMAX}")
        return v


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


@dataclass
class OptimizationResult:
    atoms: Atoms
    forces: np.ndarray
    energy: float


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



    



    