from rdkit import Chem
from rdkit.Chem import AllChem

from ase.io import read
from quippy.descriptors import Descriptor

import logging
import pickle
import numpy as np

from xps.models.models import *
import os
cutoff = 5; dc = 0.5; sigma = 0.5
zeta = 6
SOAP = {"C": 'soap_turbo alpha_max={8 8 8} l_max=8 rcut_soft=%.4f rcut_hard=%.4f atom_sigma_r={%.4f %.4f %.4f} atom_sigma_t={%.4f %.4f %.4f} \
               atom_sigma_r_scaling={0. 0. 0.} atom_sigma_t_scaling={0. 0. 0.} radial_enhancement=1 amplitude_scaling={1. 1. 1.} \
               basis="poly3gauss" scaling_mode="polynomial" species_Z={1 6 8} n_species=3 central_index=2 central_weight={1. 1. 1.} \
               compress_mode=trivial' % (cutoff-dc, cutoff, *(6*[sigma])),
        "O": 'soap_turbo alpha_max={8 8 8} l_max=8 rcut_soft=%.4f rcut_hard=%.4f atom_sigma_r={%.4f %.4f %.4f} atom_sigma_t={%.4f %.4f %.4f} \
               atom_sigma_r_scaling={0. 0. 0.} atom_sigma_t_scaling={0. 0. 0.} radial_enhancement=1 amplitude_scaling={1. 1. 1.} \
               basis="poly3gauss" scaling_mode="polynomial" species_Z={1 6 8} n_species=3 central_index=3 central_weight={1. 1. 1.} \
               compress_mode=trivial' % (cutoff-dc, cutoff, *(6*[sigma]))}


def get_gaussians(values, sigma, limit = 2) -> SpectrumData:
    def g(BE_sweep, BE_max, sigma_):
        G = 1/(sigma_*np.sqrt(2*np.pi)) * np.exp(-(BE_sweep-BE_max)**2 / (2*sigma_**2))
        new_y= np.array(G)
        return new_y

    # Create a range of x values for the plot
    x = np.linspace(min(values) - limit, max(values) + limit, 1000)
    logging.info(f'n points in spectra = {len(x)}')

    gaussian=0
    for val in values:
        gaussian += g(x,val,sigma)

    return SpectrumData(
        x = SpectralData(
            label = "Binding Energies",
            data = list(x),
            units = 'eV'
        ),
        y = SpectralData(
            label = "Intensities",
            data = list(gaussian),
            units = 'Relative'
        )
    )

def get_all_BEs(predictions: ModelPrediction) -> list:
    all_BE = []
    for pred in predictions:
        for be in pred.prediction.data:
            all_BE.append(be)
    return all_BE

def be_to_spectrum(be:bindingEnergyPrediction,sigma= 0.35, limit = 2) -> PredictedXPSSpectrum:
    all_BEs = get_all_BEs(be)

    spectra_gauss  = get_gaussians(all_BEs, sigma, limit = limit)

    return PredictedXPSSpectrum(
        allBindingEnergies = all_BEs,
        gaussian = spectra_gauss,
        sigma = sigma
    )

def soap_to_BE(soap:soap, element:str, orbital:str = '1s') -> bindingEnergyPrediction:
    '''Searches for the relevant model and predict the binding energy for the given element and orbital'''
    model_file = f'xps/MLmodels/XPS_GPR_{element}{orbital}.pkl'

    model = pickle.load(open(model_file, 'rb'))
    logging.info('Model loaded')

    be, std = model.predict(soap, return_std = True)
    logging.info(f'{len(be)} predictions')
    return bindingEnergyPrediction(
        modelFile = model_file,
        data = list(be),
        standardDeviation = list(std)
    )

def molfile_to_xyz(molfile:str):
    '''From molfile to ASE Atoms object'''
    
    logging.info('molfile to xyz')
    temp_file = 'temp2.mol'
    with open(temp_file, 'w+') as f:
        f.write(molfile) #write molfile to temporary file
    mol = Chem.MolFromMolFile(temp_file) # Read temp file into RDKit molecule
    mol = Chem.AddHs(mol)# post-process molecule
    AllChem.EmbedMolecule(mol)
    Chem.MolToMolFile(mol, temp_file)    # Write RDKit molecule to a temporary file
    molecule = read(temp_file) # Read the temporary file into ASE Atoms object
    return molecule

def xyz_to_soap_turbo(mol, element) -> soap:
    '''Create soap turbo descriptor'''
    desriptor =Descriptor(SOAP[element])

    atoms = []
    descMol = desriptor.calc(mol) #descriptor for each atom

    if 'data' in descMol:
        desc_data = descMol['data'] #get the data from the descriptor object if exist
        for atom in desc_data:
            atoms.append(atom)
    return soap(
        element = element,
        descriptor = SOAP[element],
        data = atoms
    )

def molfile_to_BE(molfile:str) -> list:
    '''
    From Molfile to predicted Binding Energies
    '''
    logging.info(molfile)

    not_present = []
    be_predictions = []
    mol = molfile_to_xyz(molfile)

    for element in ['C', 'O']:
        logging.info(element)
        if (element in mol.symbols):
            orbital = '1s'
            soaps = xyz_to_soap_turbo(mol, element=element)
            be = soap_to_BE(soaps.data, element, orbital=orbital)

            model_prediction = ModelPrediction(
                element = element,
                orbital= orbital,
                #soapTurbo = soaps,
                prediction = be
            )
            be_predictions.append(model_prediction)

        else:
            not_present.append(element)
    logging.info(f"{len(be_predictions)} predictions in total, {len(not_present)} no model found")
    return be_predictions

    
def SMILES_to_molfile(smiles:str) -> MolfileRequest:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    Chem.MolToMolFile(mol, 'temp.mol')    # Write RDKit molecule to a temporary file
    with open('temp.mol', 'r+') as f:
        content = f.readlines()
    content = ''.join(content)

    return MolfileRequest(
        molfile = content
    )