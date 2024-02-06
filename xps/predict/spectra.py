import numpy as np
import logging

from xps.predict.BE import smiles_to_BE, molfile_to_BE

logging.basicConfig(level=logging.INFO)

def get_gaussians(values, sigma, limit = 2):
    def g(BE_sweep, BE_max, sigma_):
        logging.info(type(sigma_))
        logging.info(type(sigma_))

        G = 1/(sigma_*np.sqrt(2*np.pi)) * np.exp(-(BE_sweep-BE_max)**2 / (2*sigma_**2))
        new_y= np.array(G)
        logging.info('Creating gaussian')

        return new_y

    # Create a range of x values for the plot
    x = np.linspace(min(values) - limit, max(values) + limit, 1000)
    logging.info(f'n points in spectra = {len(x)}')

    gaussian=0
    for val in values:
        gaussian += g(x,val,sigma)
    return x, gaussian


def molfile_to_spectrum(molfile:str, sigma = 0.35, limit = 2):
    binding_energies = molfile_to_BE(molfile)
    logging.info(f'Binding Energies OK: n = {len(binding_energies)}')
    BEs, intensities  = get_gaussians(binding_energies, sigma, limit = limit)
    return BEs, intensities


def smiles_to_spectrum(smiles, sigma = 0.35, limit =2):
    binding_energies = smiles_to_BE(smiles)
    logging.info(f'Binding Energies OK: n = {len(binding_energies)}')
    BEs, intensities  = get_gaussians(binding_energies, sigma, limit = limit)
    #logging.info(type(BEs), type(intensities))
    return BEs, intensities