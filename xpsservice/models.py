# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np
from ase import Atoms
from pydantic import BaseModel, Field, validator
from rdkit import Chem
from .settings import ALLOWED_FMAX, ALLOWED_ELEMENTS, ALLOWED_FF, ALLOWED_METHODS, transition_map


class TransitionValidator(BaseModel):
    transition: str
    
    @validator("transition")
    def check_orbital(cls, value):
        if value not in transition_map:
            raise ValueError(f"Transition {value} is not allowed.")
        return value


class XPSRequest(BaseModel):
    smiles: Optional[str] = Field(
        None,
        description="SMILES string of input molecule. The service will add implicit hydrogens.",
    )
    molFile: Optional[str] = Field(
        None,
        description="String with molfile with expanded hydrogens. The service will not attempt to add implicit hydrogens to ensure that the atom ordering is preserved.",
    )
    method: Optional[str] = Field(
        "GFN2xTB",
        description="Method for geometry optimization and XPS binding energies calculation. Allowed values: `GFNFF`, `GFN2xTB`, `GFN1xTB`.",
    )
    fmax: Optional[float] = Field(
        0.01,
        description="Maximum force admissible during geometry optimization. Typically ranging from 0.1 to 0.0001."
    )

    @validator("smiles")
    def validate_smiles(cls, v):
        if v:
            mol = Chem.MolFromSmiles(v)
            if not mol:
                raise ValueError("Invalid SMILES string provided.")
            for atom in mol.GetAtoms():
                if atom.GetSymbol() not in ALLOWED_ELEMENTS:
                    raise ValueError(f"SMILES contains an invalid element: {atom.GetSymbol()}. Only {ALLOWED_ELEMENTS} are allowed.")
        return v

    @validator("molFile")
    def validate_molfile(cls, v):
        if v:
            mol = Chem.MolFromMolBlock(v)
            if not mol:
                raise ValueError("Invalid molfile provided.")
            for atom in mol.GetAtoms():
                if atom.GetSymbol() not in ALLOWED_ELEMENTS:
                    raise ValueError(f"Molfile contains an invalid element: {atom.GetSymbol()}. Only {ALLOWED_ELEMENTS} are allowed.")
        return v

    @validator("method")
    def validate_method(cls, v):
        if v not in ALLOWED_METHODS:
            raise ValueError(f"Method must be in {ALLOWED_METHODS}")
        return v
    
    @validator("fmax")
    def validate_fmax(cls, v):
        if not (ALLOWED_FMAX[0] <= v <= ALLOWED_FMAX[1]):
            raise ValueError(f"fmax must be within the range {ALLOWED_FMAX}")
        return v


class Position(BaseModel):
    x: float
    y: float
    z: float

class PredictionData(BaseModel):
    binding_energy: float
    standard_deviation: float

class Prediction(BaseModel):
    atom: str
    position: Position
    prediction: Dict[str, PredictionData]  # [orbital, prediction for the orbital]

class XPSResult(BaseModel):
    molfile: str = Field(
        None, description="Molfile (calculated or given) used for the binding energy prediction."
    )
    smiles: Optional[str] = Field(
        None, description="SMILES (if given) used for the binding energy prediction."
    )
    prediction: List[Prediction] = Field(
        None, description="List of binding energies and standard deviations for every atom of the molecule and any predicted orbital."
    )


@dataclass
class OptimizationResult:
    atoms: Atoms
    forces: np.ndarray
    energy: float


class ConformerRequest(BaseModel):
    smiles: Optional[str] = Field(
        None,
        description="SMILES string of input molecule. The service will add implicit hydrogens.",
    )
    molFile: Optional[str] = Field(
        None,
        description="String with molfile with expanded hydrogens. The service will not attempt to add implicit hydrogens to ensure that the atom ordering is preserved.",
    )
    forceField: Optional[str] = Field(
        "uff",
        description="Force field for energy minimization. Options: 'uff', 'mmff94', 'mmff94s'.",
    )
    rmsdThreshold: Optional[float] = Field(
        0.5, description="RMSD threshold for pruning conformer library."
    )
    maxConformers: Optional[int] = Field(
        1,
        description="Maximum number of conformers generated (after pruning).",
    )

    @validator("forceField")
    def method_match(cls, v):
        if v not in ALLOWED_FF:
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
