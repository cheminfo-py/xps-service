from fastapi import FastAPI, HTTPException
import logging

from xps.models.models import *
from xps.predict.xps_predictions import molfile_to_BE, be_to_spectrum, SMILES_to_molfile, get_atoms

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="XPS webservice",
    description="Offers xps predictions.",
    #version=__version__,
    contact={"name": "Cheminfo", "email": "admin@cheminfo.org",},
    license_info={"name": "MIT"},
)

@app.post("/test", 
         )
def test(test):
    return test


@app.get("/ping")
def ping():
    return {"message": "pong"}


@app.post("/v1/fromMolfile", 
         response_model=FullPrediction
         )
def fromMolfile(molfile: MolfileRequest, sigma = 0.35):
    logging.info(f'Request: {molfile.molfile}')
    try:
        included, excluded = get_atoms(molfile.molfile)
        be = molfile_to_BE(molfile.molfile)
        pred_spectrum = be_to_spectrum(be, sigma=molfile.sigma)

    except Exception as e:
        raise HTTPException(
                status_code=422,
                detail=f"Error in predicting spectra ({e})",
            )
    return FullPrediction(
        molfile = molfile.molfile,
        elementsIncluded = included,
        elementsExcluded = excluded,
        bindingEnergies = be,
        spectrum = pred_spectrum
    )


@app.post("/v1/fromSMILES", 
         response_model = FullPrediction
         )
def fromSMILES(smiles: SMILES, sigma = 0.35):
    logging.info(f'Request: {smiles.smiles}')
    try:
        molfile = SMILES_to_molfile(smiles.smiles)
        included, excluded = get_atoms(molfile.molfile)
        be = molfile_to_BE(molfile.molfile)
        pred_spectrum = be_to_spectrum(be, sigma=sigma)

    except Exception as e:
        raise HTTPException(
                status_code=422,
                detail=f"Error in predicting spectra ({e})",
            )
    return FullPrediction(
        molfile = molfile.molfile,
        smiles = smiles.smiles,
        elementsIncluded = included,
        elementsExcluded = excluded,
        bindingEnergies = be,
        spectrum = pred_spectrum
    )
    
# Define a simple FastAPI function that returns a list
@app.get("/simple_list", response_model=ListResponse)
def simple_list():
    # Return a list of integers as an example
    return ListResponse(items=[1, 2, 3, 4, 5])

@app.get("/test_simple", response_model=SimpleResponse)
def test_simple():
    # Define the response data
    data = [1, 2, 3, 4, 5]

    # Return the response
    return SimpleResponse(
        message="Simple function response",
        data=data
    )
