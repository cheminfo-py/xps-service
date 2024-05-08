# -*- coding: utf-8 -*-
import os

from functools import lru_cache
from typing import List, Tuple, Union

import numpy as np
import wrapt_timeout_decorator
import logging
import pickle
import hashlib

from ase import Atoms
from quippy.descriptors import Descriptor

from .cache import soap_config_cache, soap_descriptor_cache, model_cache
from .models import *
from .optimize import run_xtb_opt
from .settings import MAX_ATOMS_FF, MAX_ATOMS_XTB, TIMEOUT, transition_map
from .utils import (
    hash_atoms,
    hash_object,
    molfile2ase,
    molfile2smiles
)


#working MM
def load_ml_model(transition_info):
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
    
    # Load the SOAP configuration from the file
    with open(soap_filepath, 'r') as file:
        soap_config = file.read()
    
    if not soap_config:
        raise ValueError(f"The SOAP configuration file at {soap_filepath} is empty or not valid.")
    
    logging.debug(f"Loaded SOAP config for {transition_info['element']} {transition_info['orbital']}")

    # Return the SOAP configuration
    return soap_config


def load_models_and_descriptors(transition_map):
    print("Entered load")
    
    #soap_config_cache.clear()
    #soap_descriptor_cache.clear()
    #model_cache.clear()
    
    for transition_key, transition_info in transition_map.items():
        try:
            
            # Load SOAP config and store in cache
            soap_config = load_soap_config(transition_info)
            soap_config_hashed_key = cache_hash(transition_key, "soap_config_cache")
            print(soap_config_hashed_key)
            soap_config_cache.set(soap_config_hashed_key, soap_config)
            logging.debug(f"SOAP config for {transition_key} loaded")
            
            # Create SOAP descriptor from soap_config and store in cache
            soap_descriptor = Descriptor(soap_config)    
            soap_descriptor_hashed_key = cache_hash(transition_key, "soap_descriptor_cache")
            soap_descriptor_cache.set(soap_descriptor_hashed_key, soap_descriptor)
            logging.debug(f"SOAP descriptor for {transition_key} loaded")
            
            # Load ML model and store in cache
            ml_model = load_ml_model(transition_info)
            ml_model_hashed_key = cache_hash(transition_key, "ml_model_cache")
            model_cache.set(ml_model_hashed_key, ml_model)
            logging.debug(f"ML model for {transition_key} loaded")
            
        except Exception as e:
            logging.error(f"Error loading data for transition {transition_key}: {str(e)}")
            continue  # Optionally skip to the next iteration on error


def check_cache_status(transition_map) -> Dict[str, Dict[str, bool]]:
    print("checking cache status")
    """
    Check the status of the cache for each transition in the transition_map.
    
    Returns:
        A dictionary where the keys are transition keys, and the values are dictionaries
        indicating the status of each cache (True if loaded, False otherwise).
    """
    logging.debug("entered check_cache_status")
    cache_status = {}

    # Iterate through each transition in the transition_map
    for transition_key in transition_map:
        # Check the status of each cache for the transition key
        soap_config_loaded = cache_hash(transition_key, "soap_config_cache") in soap_config_cache
        soap_descriptor_loaded = cache_hash(transition_key, "soap_descriptor_cache") in soap_descriptor_cache
        model_loaded = cache_hash(transition_key, "ml_model_cache") in model_cache
        
        # Store the status in the cache_status dictionary
        cache_status[transition_key] = {
            "soap_config_loaded": soap_config_loaded,
            "soap_descriptor_loaded": soap_descriptor_loaded,
            "model_loaded": model_loaded,
        }
    
    return cache_status


def has_any_cache_failure(cache_status: Dict[str, Dict[str, bool]]) -> bool:
    """
    Check if any of the cache status values are False in the given cache_status dictionary.
    """
    # Iterate through each transition and check each status
    for transition_key, status_dict in cache_status.items():
        for status_name, status_value in status_dict.items():
            if not status_value:
                # If any status is False, return True
                return True
    
    # If all statuses are True, return False
    return False


def get_max_atoms(method):
    if method == "GFNFF":
        return MAX_ATOMS_FF
    elif method == "GFN2xTB":
        return MAX_ATOMS_XTB
    elif method == "GFN1xTB":
        return MAX_ATOMS_XTB


def calculate_binding_energies(ase_mol: Atoms, transition_key):
    print(" entered calculate binding energies")
    if not isinstance(ase_mol, Atoms):
        raise TypeError(f"in calculate_binding_energies expected ase_mol to be of type Atoms, but got {type(ase_mol).__name__}")
    """
    Calculate binding energies for the atoms of a given molecule based on the loaded ML model and loaded descriptor.
    
    Parameters:
    mol: The molecule object.
    element: The element for which to calculate binding energies.
    transition_key: The specific transition_key follows the naming in transition_map from .models.py (e.g., "C1s" or "O1s").

    Returns:
    list: A list of binding energy predictions for the specified element and transition.
    """
    transition_info = transition_map[transition_key]
    element = transition_info['element']
    orbital = transition_info['orbital']

    # Check if the input transition is one of the keys in the transition_map
    if transition_key not in transition_map:
        raise KeyError(f"Transition '{transition_key}' is not a valid transition. Valid transitions are: {list(transition_map.keys())}")
    
    '''
    # Retrieve SOAP descriptor from the cache
    soap_descriptor_hashed_key = cache_hash(transition_key, "soap_descriptor_cache")
    soap_descriptor = soap_descriptor_cache.get(soap_descriptor_hashed_key)
    if soap_descriptor is None:
        logging.error(f"SOAP descriptor not found in cache for transition {transition_key}")
        return []
    '''
    
    # Retrieve ML model from the cache
    model_hashed_key = cache_hash(transition_key, "ml_model_cache")
    ml_model = model_cache.get(model_hashed_key)
    if ml_model is None:
        logging.error(f"ML model not found in cache for transition {transition_key}")
        return []
    
    '''
    # Retrieve SOAP descriptor from the cache
    soap_config_hashed_key = cache_hash(transition_key, "soap_config_cache")
    soap_config = soap_config_cache.get(soap_config_hashed_key)
    print("after soap_config")
    print(soap_config)
    soap_descriptor = Descriptor(soap_config)
    print("after soap_descriptor")
    
    


    
    SOAP = {"C": 'soap_turbo alpha_max={8 8 8} l_max=8 rcut_soft=3.7500 rcut_hard=4.2500 atom_sigma_r={0.5000 0.5000 0.5000} atom_sigma_t={0.5000 0.5000 0.5000}               atom_sigma_r_scaling={0. 0. 0.} atom_sigma_t_scaling={0. 0. 0.} radial_enhancement=1 amplitude_scaling={1. 1. 1.}               basis="poly3gauss" scaling_mode="polynomial" species_Z={1 6 8} n_species=3 central_index=2 central_weight={1. 1. 1.}               compress_mode=trivial',
    "O": 'soap_turbo alpha_max={8 8 8} l_max=8 rcut_soft=3.7500 rcut_hard=4.2500 atom_sigma_r={0.5000 0.5000 0.5000} atom_sigma_t={0.5000 0.5000 0.5000}               atom_sigma_r_scaling={0. 0. 0.} atom_sigma_t_scaling={0. 0. 0.} radial_enhancement=1 amplitude_scaling={1. 1. 1.}               basis="poly3gauss" scaling_mode="polynomial" species_Z={1 6 8} n_species=3 central_index=3 central_weight={1. 1. 1.}               compress_mode=trivial'
    }
    
    '''
    cutoff = 4.25; dc = 0.5; sigma = 0.5
    zeta = 6
    SOAP = {"C": 'soap_turbo alpha_max={8 8 8} l_max=8 rcut_soft=%.4f rcut_hard=%.4f atom_sigma_r={%.4f %.4f %.4f} atom_sigma_t={%.4f %.4f %.4f} \
               atom_sigma_r_scaling={0. 0. 0.} atom_sigma_t_scaling={0. 0. 0.} radial_enhancement=1 amplitude_scaling={1. 1. 1.} \
               basis="poly3gauss" scaling_mode="polynomial" species_Z={1 6 8} n_species=3 central_index=2 central_weight={1. 1. 1.} \
               compress_mode=trivial' % (cutoff-dc, cutoff, *(6*[sigma])),
        "O": 'soap_turbo alpha_max={8 8 8} l_max=8 rcut_soft=%.4f rcut_hard=%.4f atom_sigma_r={%.4f %.4f %.4f} atom_sigma_t={%.4f %.4f %.4f} \
               atom_sigma_r_scaling={0. 0. 0.} atom_sigma_t_scaling={0. 0. 0.} radial_enhancement=1 amplitude_scaling={1. 1. 1.} \
               basis="poly3gauss" scaling_mode="polynomial" species_Z={1 6 8} n_species=3 central_index=3 central_weight={1. 1. 1.} \
               compress_mode=trivial' % (cutoff-dc, cutoff, *(6*[sigma]))}
    
    
    # new version from here
    # Load SOAP descriptor and store in cache
            
    if element == "C":
        soap_descriptor = Descriptor(SOAP["C"])
                
    elif element == "O":
        soap_descriptor = Descriptor(SOAP["O"])
    # replaced by loading directly the models
    
    
    # Calculate the SOAP descriptor for the molecule
    desc_data = soap_descriptor.calc(ase_mol)

    # Check if the descriptor data is available
    if 'data' not in desc_data:
        logging.error(f"No descriptor data found for molecule with transition {transition_key}")
        return []

    # Get the data from the descriptor object
    desc_molecule = desc_data['data']
    
    # Predict binding energies using the ML model
    be, std = ml_model.predict(desc_molecule, return_std=True)
    logging.info(f'Predicted {len(be)} binding energies for element {element}, orbital {orbital}, defined in {transition_key}')

    # Return a list of binding energy predictions
    return list(zip(be, std))


def run_xps_calculations(ase_mol: Atoms) -> dict:
    if not isinstance(ase_mol, Atoms):
        raise TypeError(f"in run_xps_calculation, expected ase_mol to be of type Atoms, but got {type(ase_mol).__name__}")

    # Dictionary to store predictions by element and transition
    be_predictions = {}

    # Iterate through the keys (transitions) in transition_map
    for transition_key in transition_map.keys():
        # Retrieve the element and orbital from the transition info
        transition_info = transition_map[transition_key]
        element = transition_info['element']

        # Check if the element is present in the ASE molecule
        if element in ase_mol.symbols:
            # Calculate binding energies for the element and transition
            predictions = calculate_binding_energies(ase_mol, transition_key)

            # Store the predictions in the dictionary using the transition_key as the key
            be_predictions[transition_key] = predictions
        else:
            logging.warning(f"No model found for element {element} in transition {transition_key}")

    return be_predictions


@wrapt_timeout_decorator.timeout(TIMEOUT, use_signals=False)
def calculate_from_molfile(molfile, method) -> XPSResult:

    # Convert molfile to smiles and ASE molecule
    smiles = molfile2smiles(molfile)
    ase_mol, mol = molfile2ase(molfile, get_max_atoms(method))
    
    if not isinstance(ase_mol, Atoms):
        raise TypeError(f"in calculate_from_molfile, expected ase_mol to be of type Atoms, but got {type(ase_mol).__name__}")
    
    # Optimize the ASE molecule
    opt_result = run_xtb_opt(ase_mol, fmax=0.2, method=method)
    opt_ase_mol = opt_result.atoms
    if not isinstance(opt_ase_mol, Atoms):
        raise TypeError(f"After xtb optimization, expected ase_mol to be of type Atoms, but got {type(ase_mol).__name__}")
    
    # Run XPS calculations
    be_predictions = run_xps_calculations(opt_ase_mol)
    
    # Extract binding energies and standard deviations
    binding_energies = []
    standard_deviations = []
    for transition_key, predictions in be_predictions.items():
        energies, stds = zip(*predictions)
        binding_energies.extend(list(energies))
        standard_deviations.extend(list(stds))

    # Create an instance of XPSResult
    xps_result = XPSResult(
        molfile=molfile,
        smiles=smiles,
        bindingEnergies=binding_energies,
        standardDeviations=standard_deviations
    )

    return xps_result




##################################
##################################



def ir_hash(atoms, method):
    return hash_object(str(hash_atoms(atoms)) + method)

#returns a hash to be used in for the cache
def cache_hash(transition_key, cache_type):
    
    # Concatenate the transition key and cache type
    input_string = transition_key + cache_type
    
    # Hash the combined string using SHA-256
    hash_object = hashlib.sha256(input_string.encode('utf-8'))
    
    # Convert the hash to a hexadecimal string
    hashed_key = hash_object.hexdigest()
    
    return hashed_key