from fastapi import FastAPI, HTTPException
import logging

from xps.models.models import XPS_prediction, XPS_BE, SpectrumModel, SpectrumData, Spectrum
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


@app.post("/v0/spectrum", 
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

@app.post("/v1/spectrum", 
         response_model=SpectrumModel
         )
def predict_xps_spectrum(smiles: str, sigma = 0.35):
    try:
        sigma = float(sigma)
        logging.info(f'Predicting spectrum of {smiles}')
        energies, intensities = smiles_to_spectrum(smiles, sigma)

        xps = SpectrumModel(spectrum = SpectrumData(
            x = Spectrum(label = "Energies",
                         data = list(energies),
                         units = "eV"),
            y = Spectrum(label = "Intensities",
                         data= list(intensities),
                         units = "Relative")),
            sigma = sigma
        )
    except Exception as e:
        raise HTTPException(
                status_code=422,
                detail=f"Error in predicting spectra({e})",
            )
    return xps



@app.post("/v1/bindingEnergies", 
         response_model=XPS_BE
         )
def predict_BE(smiles: str):
    logging.info(f'getting binding ergies of {smiles}')

    try:
        binding_energies = smiles_to_BE(smiles)
        xps = XPS_BE(
            BindingEnergies = Spectrum(label = "Intensities",
                                       data = list(binding_energies),
                                       units = "Relative")
        )
    except Exception as e:
        raise HTTPException(
                status_code=422,
                detail=f"Error in predicting binding energies ({e})`",
            )
    return xps