from fastapi import FastAPI, HTTPException
import logging

from xps.models.models import XPS_prediction, XPS_BE
from xps.predict.BE import smiles_to_BE
from xps.predict.spectra import smiles_to_spectrum

logging.basicConfig(level=logging.INFO)


app = FastAPI(
    title="XPS webservice",
    description="Offers xps predictions.",
    #version=__version__,
    contact={"name": "Cheminfo", "email": "admin@cheminfo.org",},
    license_info={"name": "MIT"},
)


@app.post("/spectrum", 
         response_model=XPS_prediction
         )
def predict_xps_spectrum(smiles: str, sigma = 0.35):
    try:
        sigma = float(sigma)
        logging.info(f'Predicting spectrum of {smiles}')
        energies, intensities = smiles_to_spectrum(smiles, sigma)
        xps = XPS_prediction(
            energies=  list(energies),
            intensities = list(intensities),
            sigma = sigma
        )
    except Exception as e:
        raise HTTPException(
                status_code=422,
                detail=f"Error in predicting spectra({e})",
            )
    return xps



@app.post("/BE", 
         response_model=XPS_BE
         )
def predict_BE(smiles: str):
    logging.info(f'getting binding ergies of {smiles}')

    try:
        binding_energies = smiles_to_BE(smiles)
        xps = XPS_BE(
            BE = list(binding_energies)
        )
    except Exception as e:
        raise HTTPException(
                status_code=422,
                detail=f"Error in predicting binding energies ({e})`",
            )
    return xps