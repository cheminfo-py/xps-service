# -*- coding: utf-8 -*-
import os

from csv import excel
import shutil
from functools import lru_cache
from typing import List, Tuple, Union

import numpy as np
import wrapt_timeout_decorator
import logging
import pickle

from ase import Atoms
from ase.calculators.bond_polarizability import BondPolarizability
from ase.vibrations import Infrared
from ase.vibrations.placzek import PlaczekStatic
from ase.vibrations.raman import StaticRamanCalculator
from fastapi.logger import logger
from rdkit import Chem
from scipy import spatial
from xtb.ase.calculator import XTB
from math import pi, sqrt, log
from .cache import ir_cache, ir_from_molfile_cache, ir_from_smiles_cache
from .cache import soap_cache, ml_cache
from .models import *
from .optimize import run_xtb_opt
from .settings import IMAGINARY_FREQ_THRESHOLD, MAX_ATOMS_FF, MAX_ATOMS_XTB, TIMEOUT
from .utils import (
    get_moments_of_inertia,
    hash_atoms,
    hash_object,
    molfile2ase,
    smiles2ase,
    smiles2molfile
)

#working MM
#get soap and ML model for a given transition, i.e C1s
def get_soap_and_model(transition: str):
    # Validate the transition
    TransitionValidator(transition=transition)
    
    # Retrieve the SOAP and model from the mapping
    transition_info = transition_map[transition]
    soap_key = transition_info["soap"]
    model_key = transition_info["model"]
    
    # Check the cache for SOAP configuration
    soap_config = soap_cache.get(soap_key)
    if soap_config is None:
        # Load SOAP config if not in cache (replace with your loading logic)
        soap_config = load_soap_config(transition_info)
        soap_cache[soap_key] = soap_config
    logging.info("soap xx loaded")

    # Check the cache for ML model
    ml_model = ml_cache.get(model_key)
    if ml_model is None:
        # Load ML model if not in cache (replace with your loading logic)
        ml_model = load_ml_model(transition_info)
        ml_cache[model_key] = ml_model
    logging.info("ml xx loaded")

    return soap_config, ml_model


#working MM
def load_ml_model(transition_info: Dict[str, Any]) -> Any:
    try:
        model_filepath = transition_info['model_filepath']
        with open(model_filepath, 'rb') as model_file:
            model = pickle.load(model_file)
        logging.debug(f"Loaded ML model for {transition_info['element']} {transition_info['orbital']}")
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at {model_filepath}.")
    except pickle.UnpicklingError as e:
        raise RuntimeError(f"Failed to load model file {model_filepath}: {str(e)}")


#working MM
def load_soap_config(transition_info):
    # Get the filepath for the SOAP configuration
    soap_filepath = transition_info['soap_filepath']
    
    # Check if the file exists
    if not os.path.exists(soap_filepath):
        raise FileNotFoundError(f"SOAP configuration file not found at path: {soap_filepath}")
    
    # Read the SOAP configuration file
    with open(soap_filepath, 'r') as file:
        # Read the content of the file and execute it as Python code
        exec(file.read())
        
    # Check if the variable `SOAP` is defined in the file's content
    if 'SOAP' not in locals():
        raise ValueError(f"SOAP configuration variable 'SOAP' not defined in the file: {soap_filepath}")
    
    logging.debug(f"Loaded SOAP config for {transition_info['element']} {transition_info['orbital']}")
    # Return the SOAP configuration
    return locals()['SOAP']


#working MM
def test_model_and_soap_loading(transition_map):
    # Create a list to store test results
    test_results = []
    
    logging.debug("entered test function for transition_map")
    
    # Iterate through all transitions in the transition_map
    for transition_key, transition_info in transition_map.items():
        logging.debug(f"entered loop for {transition_info['element']} {transition_info['orbital']}")
        try:
            # Load the ML model and SOAP configuration
            soap_config, ml_model = get_soap_and_model(transition_key)
        
            # Perform some basic checks
            if soap_config and ml_model:
                # If both SOAP config and ML model are loaded, the test is successful
                test_results.append((transition_key, "Success"))
            else:
                # If either the SOAP config or ML model is not loaded, the test fails
                test_results.append((transition_key, "Failure: Either SOAP config or ML model not loaded"))
        except Exception as e:
            # Capture any exceptions during loading and mark the test as failed
            test_results.append((transition_key, f"Failure: {str(e)}"))

    return test_results




def xps_from_smiles(smiles:str):
    a = 1
    return a

def xps_from_molfile(molfile:str):
    a = 1
    return a

def get_max_atoms(method):
    if method == "GFNFF":
        return MAX_ATOMS_FF
    elif method == "GFN2xTB":
        return MAX_ATOMS_XTB
    elif method == "GFN1xTB":
        return MAX_ATOMS_XTB


def ir_hash(atoms, method):
    return hash_object(str(hash_atoms(atoms)) + method)


def get_raman_spectrum(
    pz,
    modes, 
    start=0,
    end=4000,
    npts=None,
    width=4,
    type="Gaussian",
    method="standard",
    direction="central",
    normalize=False,
):
    """Get spectrum.

    The method returns wavenumbers in cm^-1 with corresponding
    absolute infrared intensity.
    Start and end point, and width of the Gaussian/Lorentzian should
    be given in cm^-1.
    normalize=True ensures the integral over the peaks to give the
    intensity.
    """
    frequencies = pz.vibrations.get_frequencies(method, direction)
    intensities = [mode['ramanIntensity'] for mode in modes]
    return get_spectrum(modes, frequencies, intensities, start, end, npts, width, type, normalize)


def get_ir_spectrum(ir, modes, start=0,
    end=4000,
    npts=None,
    width=4,
    type="Gaussian",
    method="standard",
    direction="central",
    normalize=False,):
    """Get spectrum.

    The method returns wavenumbers in cm^-1 with corresponding
    absolute infrared intensity.
    Start and end point, and width of the Gaussian/Lorentzian should
    be given in cm^-1.
    normalize=True ensures the integral over the peaks to give the
    intensity.
    """
    frequencies = ir.get_frequencies(method, direction)
    intensities = [mode['intensity'] for mode in modes]
    return get_spectrum(modes, frequencies, intensities, start, end, npts, width, type, normalize)


def fold(frequencies, intensities,
            start=800.0, end=4000.0, npts=None, width=4.0,
            type='Gaussian', normalize=False):
    """Fold frequencies and intensities within the given range
    and folding method (Gaussian/Lorentzian).
    The energy unit is cm^-1.
    normalize=True ensures the integral over the peaks to give the
    intensity.
    """

    lctype = type.lower()
    assert lctype in ['gaussian', 'lorentzian']
    if not npts:
        npts = int((end - start) / width * 10 + 1)
    prefactor = 1
    if lctype == 'lorentzian':
        intensities = intensities * width * pi / 2.
        if normalize:
            prefactor = 2. / width / pi
    else:
        sigma = width / 2. / sqrt(2. * log(2.))
        if normalize:
            prefactor = 1. / sigma / sqrt(2 * pi)

    # Make array with spectrum data
    spectrum = np.empty(npts)
    energies = np.linspace(start, end, npts)
    for i, energy in enumerate(energies):
        energies[i] = energy
        if lctype == 'lorentzian':
            spectrum[i] = (intensities * 0.5 * width / pi /
                            ((frequencies - energy)**2 +
                            0.25 * width**2)).sum()
        else:
            spectrum[i] = (intensities *
                            np.exp(-(frequencies - energy)**2 /
                                    2. / sigma**2)).sum()
    return [energies, prefactor * spectrum]



def get_spectrum(modes, frequencies, intensities,     start=0,
    end=4000,
    npts=None,
    width=4,
    type="Gaussian",

    normalize=False): 
    
    filtered_frequencies, filtered_intensities = [], []

    for freq, int, mode in zip(frequencies, intensities, modes):
        if mode['modeType'] == 'vibration':
            filtered_frequencies.append(freq.real)
            filtered_intensities.append(int)
    return fold(
        filtered_frequencies, filtered_intensities, start, end, npts, width, type, normalize
    )



def run_xtb_ir(
    atoms: Atoms, method: str = "GFNFF", mol: Union[None, Chem.Mol] = None
) -> IRResult:
    if mol is None:
        raise Exception

    this_hash = ir_hash(atoms, method)
    logger.debug(f"Running IR for {this_hash}")
    moi = atoms.get_moments_of_inertia()
    linear = sum(moi > 0.01) == 2
    result = ir_cache.get(this_hash)

    if result is None:
        logger.debug(f"IR not in cache for {this_hash}, running")
        atoms.pbc = False
        atoms.calc = XTB(method=method)

        try:
            rm = StaticRamanCalculator(atoms, BondPolarizability, name=str(this_hash))
            rm.ir = True
            rm.run()
            pz = PlaczekStatic(atoms, name=str(this_hash))
            raman_intensities = pz.get_absolute_intensities()

        except Exception as e:
            print(e)
            shutil.rmtree(str(this_hash))
            raman_intensities = None
            raman_spectrum = None

        ir = Infrared(atoms, name=str(this_hash))
        ir.run()

        zpe = ir.get_zero_point_energy()
        most_relevant_mode_for_bond = None
        bond_displacements = None
        if mol is not None:
            bond_displacements = compile_all_bond_displacements(mol, atoms, ir)
            mask = np.zeros_like(bond_displacements)
            if len(atoms) > 2:
                if linear:
                    mask[:5, :] = 1
                else:
                    mask[:6, :] = 1
                masked_bond_displacements = np.ma.masked_array(bond_displacements, mask)
            else:
                masked_bond_displacements = bond_displacements
            most_relevant_mode_for_bond_ = masked_bond_displacements.argmax(axis=0)
            most_relevant_mode_for_bond = []
            bonds = get_bonds_from_mol(mol)
            for i, mode in enumerate(most_relevant_mode_for_bond_):
                most_relevant_mode_for_bond.append(
                    {
                        "startAtom": bonds[i][0],
                        "endAtom": bonds[i][1],
                        "mode": int(mode),
                        "displacement": bond_displacements[mode][i],
                    }
                )
        displacement_alignments = [
            get_alignment(ir, n) for n in range(3 * len(ir.indices))
        ]

        mode_info, has_imaginary, has_large_imaginary = compile_modes_info(
            ir,
            linear,
            displacement_alignments,
            bond_displacements,
            bonds,
            raman_intensities,
        )

        spectrum = get_ir_spectrum(ir, mode_info)
        try:
            raman_spectrum = list(get_raman_spectrum(pz, mode_info)[1])
        except Exception as e: 
            raman_spectrum = None
        if raman_spectrum is not None:
            assert len(spectrum[0]) == len(raman_spectrum)

        result = IRResult(
            wavenumbers=list(spectrum[0]),
            intensities=list(spectrum[1]),
            ramanIntensities=raman_spectrum,
            zeroPointEnergy=zpe,
            modes=mode_info,
            hasImaginaryFrequency=has_imaginary,
            mostRelevantModesOfAtoms=get_max_displacements(ir, linear),
            mostRelevantModesOfBonds=most_relevant_mode_for_bond,
            isLinear=linear,
            momentsOfInertia=[float(i) for i in moi],
            hasLargeImaginaryFrequency=has_large_imaginary,
        )
        ir_cache.set(this_hash, result)

        shutil.rmtree(ir.cache.directory)
        ir.clean()
    return result












#################






#prefered method
@wrapt_timeout_decorator.timeout(TIMEOUT, use_signals=False)
def calculate_from_molfile(molfile, method, myhash):
    atoms, mol = molfile2ase(molfile, get_max_atoms(method)) #in utils
    opt_result = run_xtb_opt(atoms, method=method) #in optimize
    #result = run_xtb_xps(opt_result.atoms, method=method, mol=mol)
    #xps_from_molfile_cache.set(myhash, result, expire=None)
    #return result
    return "allright"


def ir_from_molfile(molfile, method):
    myhash = hash_object(molfile + method)

    result = xps_from_molfile_cache.get(myhash)

    if result is None:
        result = calculate_from_molfile(molfile, method, myhash)
    return result


@wrapt_timeout_decorator.timeout(TIMEOUT, use_signals=False)
def calculate_from_smiles(smiles, method, myhash):
    molfile = smiles2molfile(smiles)
    result = calculate_from_molfile(molfile, method, myhash)
    return result


def ir_from_smiles(smiles, method):
    myhash = hash_object(smiles + method)
    result = ir_from_smiles_cache.get(myhash)
    if result is None:
        result = calculate_from_smiles(smiles, method, myhash)
    return result



def compile_all_bond_displacements(mol, atoms, ir):
    bond_displacements = []
    for mode_number in range(3 * len(ir.indices)):
        bond_displacements.append(
            get_bond_displacements(mol, atoms, ir.get_mode(mode_number))
        )

    return np.vstack(bond_displacements)


def clean_frequency(frequencies, n):
    if frequencies[n].imag != 0:
        c = "i"
        freq = frequencies[n].imag

    else:
        freq = frequencies[n].real
        c = " "
    return freq, c


def compile_modes_info(
    ir, linear, alignments, bond_displacements=None, bonds=None, raman_intensities=None
):
    frequencies = ir.get_frequencies()
    symbols = ir.atoms.get_chemical_symbols()
    modes = []
    sorted_alignments = sorted(alignments, reverse=True)
    mapping = dict(zip(np.arange(len(frequencies)), np.argsort(frequencies)))
    third_best_alignment = sorted_alignments[2]
    has_imaginary = False
    has_large_imaginary = False
    num_modes = 3 * len(ir.indices)
    if raman_intensities is None:
        raman_intensities = [None] * num_modes
    for n in range(num_modes):
        n = int(mapping[n])
        if n < 3:
            # print("below 5", alignments[n])
            # if alignments[n] >= third_best_alignment:
            modeType = "translation"
            # else:
            #     modeType = "rotation"
        elif n < 5:
            modeType = "rotation"
        elif n == 5:
            if linear:
                modeType = "vibration"
            else:
                if alignments[n] >= third_best_alignment:
                    modeType = "translation"
                else:
                    modeType = "rotation"
        else:
            modeType = "vibration"

        f, c = clean_frequency(frequencies, n)
        if c == "i":
            has_imaginary = True
            if f > IMAGINARY_FREQ_THRESHOLD:
                has_large_imaginary = True
        mostContributingBonds = None
        if bond_displacements is not None:
            mostContributingBonds = select_most_contributing_bonds(
                bond_displacements[n, :]
            )
            mostContributingBonds = [bonds[i] for i in mostContributingBonds]
            mode = ir.get_mode(n)

        ramanIntensity = float(raman_intensities[n]) if raman_intensities[n] is not None else None 
        modes.append(
            {
                "number": n,
                "displacements": get_displacement_xyz_for_mode(
                    ir, frequencies, symbols, n
                ),
                "intensity": float(ir.intensities[n]),
                "ramanIntensity": ramanIntensity,
                "wavenumber": float(f),
                "imaginary": True if c == "i" else False,
                "mostDisplacedAtoms": [
                    int(i)
                    for i in np.argsort(np.linalg.norm(mode - mode.sum(axis=0), axis=1))
                ][::-1],
                "mostContributingAtoms": [
                    int(i) for i in select_most_contributing_atoms(ir, n)
                ],
                "mostContributingBonds": mostContributingBonds,
                "modeType": modeType,
                "centerOfMassDisplacement": float(
                    np.linalg.norm(ir.get_mode(n).sum(axis=0))
                ),
                "totalChangeOfMomentOfInteria": get_change_in_moi(ir.atoms, ir, n),
                "displacementAlignment": alignments[n],
            }
        )

    return modes, has_imaginary, has_large_imaginary


def get_max_displacements(ir, linear):
    mode_abs_displacements = []

    for n in range(3 * len(ir.indices)):
        mode_abs_displacements.append(np.linalg.norm(ir.get_mode(n), axis=1))

    mode_abs_displacements = np.stack(mode_abs_displacements)
    if linear:
        mode_abs_displacements[:5, :] = 0
    else:
        mode_abs_displacements[:6, :] = 0

    return dict(
        zip(
            ir.indices,
            [list(a)[::-1] for a in mode_abs_displacements.argsort(axis=0).T],
        )
    )


def get_alignment(ir, mode_number):
    dot_result = []

    displacements = ir.get_mode(mode_number)

    for i, displ_i in enumerate(displacements):
        for j, displ_j in enumerate(displacements):
            if i < j:
                dot_result.append(spatial.distance.cosine(displ_i, displ_j))

    return np.mean(dot_result)


def get_displacement_xyz_for_mode(ir, frequencies, symbols, n):
    xyz_file = []
    xyz_file.append("%6d\n" % len(ir.atoms))

    f, c = clean_frequency(frequencies, n)

    xyz_file.append("Mode #%d, f = %.1f%s cm^-1" % (n, float(f.real), c))

    if ir.ir:
        xyz_file.append(", I = %.4f (D/Å)^2 amu^-1.\n" % ir.intensities[n])
    else:
        xyz_file.append(".\n")

    # dict_label = xyz_file[-1] + xyz_file[-2]
    # dict_label = dict_label.strip('\n')

    mode = ir.get_mode(n)
    for i, pos in enumerate(ir.atoms.positions):
        xyz_file.append(
            "%2s %12.5f %12.5f %12.5f %12.5f %12.5f %12.5f\n"
            % (symbols[i], pos[0], pos[1], pos[2], mode[i, 0], mode[i, 1], mode[i, 2],)
        )

    xyz_file_string = "".join(xyz_file)
    return xyz_file_string


def get_displacement_xyz_dict(ir):
    symbols = ir.atoms.get_chemical_symbols()
    frequencies = ir.get_frequencies()
    modes = {}

    for n in range(3 * len(ir.indices)):
        modes[n] = get_displacement_xyz_for_mode(ir, frequencies, symbols, n)

    return modes


def select_most_contributing_atoms(ir, mode, threshold: float = 0.4):
    displacements = ir.get_mode(mode)
    relative_contribution = (
        np.linalg.norm(displacements, axis=1)
        / np.linalg.norm(displacements, axis=1).max()
    )
    res = np.where(
        relative_contribution
        > threshold * np.max(np.abs(np.diff(relative_contribution)))
    )[0]

    return res


def select_most_contributing_bonds(displacements, threshold: float = 0.4):

    if len(displacements) > 1:
        relative_contribution = displacements / displacements.sum()
        return np.where(
            relative_contribution
            > threshold * np.max(np.abs(np.diff(relative_contribution)))
        )[0]
    else:
        return np.array([0])


@lru_cache()
def get_bonds_from_mol(mol) -> List[Tuple[int, int]]:
    all_bonds = []
    for i in range(mol.GetNumBonds()):
        bond = mol.GetBondWithIdx(i)
        all_bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))

    return all_bonds


def get_bond_vector(positions, bond):
    return positions[bond[1]] - positions[bond[0]]


def get_displaced_positions(positions, mode):
    return positions + mode


def get_bond_displacements(mol, atoms, mode):
    bonds = get_bonds_from_mol(mol)
    positions = atoms.positions
    displaced_positions = get_displaced_positions(positions, mode) - mode.sum(axis=0)
    changes = []

    for bond in bonds:
        bond_displacements = np.linalg.norm(
            get_bond_vector(positions, bond)
        ) - np.linalg.norm(get_bond_vector(displaced_positions, bond))

        changes.append(np.linalg.norm(bond_displacements))

    return changes


def get_change_in_moi(atoms, ir, mode_number):
    return np.linalg.norm(
        np.linalg.norm(
            get_moments_of_inertia(
                get_displaced_positions(atoms.positions, ir.get_mode(mode_number)),
                atoms.get_masses(),
            )
        )
        - np.linalg.norm(get_moments_of_inertia(atoms.positions, atoms.get_masses()))
    )


#def load_models():

#def xps_from_molfile():

#def xps_from_smiles():
