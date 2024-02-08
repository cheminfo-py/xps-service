from fastapi import FastAPI, HTTPException
import logging

from xps.models.models import *
from xps.predict.xps_predictions import molfile_to_BE, be_to_spectrum

logging.basicConfig(level=logging.INFO)


app = FastAPI(
    title="XPS webservice",
    description="Offers xps predictions.",
    #version=__version__,
    contact={"name": "Cheminfo", "email": "admin@cheminfo.org",},
    license_info={"name": "MIT"},
)

@app.post("test", 
         )
def test(test):
    return test


@app.post("/v1/fromMolfile", 
         response_model=FullPrediction
         )
def fromMolfile(molfile: MolfileRequest, sigma = 0.35):
    logging.info(f'Request: {molfile.molfile}')
    try:
        be = molfile_to_BE(molfile.molfile)
        pred_spectrum = be_to_spectrum(be, sigma=sigma)

    except Exception as e:
        raise HTTPException(
                status_code=422,
                detail=f"Error in predicting spectra ({e})",
            )
    logging.info(be)
    return FullPrediction(
        molfile = molfile.molfile,
        bindingEnergies = be,
        spectrum = pred_spectrum
    )
