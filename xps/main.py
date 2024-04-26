from fastapi import FastAPI, HTTPException
import logging

from xps.models.models import *
from xps.predict.xps_predictions import molfile_to_BE, be_to_spectrum, SMILES_to_molfile, get_atoms

#logging.basicConfig(level=logging.INFO)
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


app = FastAPI(
    title="XPS webservice",
    description="Offers xps predictions.",
    #version=__version__,
    contact={"name": "Cheminfo", "email": "admin@cheminfo.org",},
    license_info={"name": "MIT"},
)

@app.get("/ping")
def ping():
    return {"message": "backtopong"}

@app.get("/docs")
def ping():
    return {"Implemented": "/SpectrumfromMolfile, /BEfromMolfile, /SpectrumfromSMILES, /BEfromSMILES" }


@app.post("/SpectrumfromMolfile", 
         response_model = SpectralPrediction
         )
def SpectrumfromMolfile(molfile: Molfile, sigma = 1.3):
    logging.info(f'Request: {molfile.molfile}')
    try:
        included, excluded = get_atoms(molfile.molfile)
        be = molfile_to_BE(molfile.molfile)
        pred_spectrum = be_to_spectrum(be, sigma)

    except Exception as e:
        raise HTTPException(
                status_code=422,
                detail=f"Error in predicting spectrum ({e})",
            )
    return SpectralPrediction(
        molfile = molfile.molfile,
        elementsIncluded = included,
        elementsExcluded = excluded,
        bindingEnergies = be,
        spectrum = pred_spectrum
    )
    
@app.post("/BEfromMolfile", 
         response_model = BEPrediction
         )
def BEfromMolfile(molfile: Molfile):
    logging.info(f'Request: {molfile.molfile}')
    try:
        included, excluded = get_atoms(molfile.molfile)
        be = molfile_to_BE(molfile.molfile)

    except Exception as e:
        raise HTTPException(
                status_code=422,
                detail=f"Error in predicting binding energies ({e})",
            )
    return SpectralPrediction(
        molfile = molfile.molfile,
        elementsIncluded = included,
        elementsExcluded = excluded,
        bindingEnergies = be
    )


@app.post("/SpectrumfromSMILES", response_model=SpectralPrediction)
def SpectrumfromSMILES(smiles: SMILES, sigma = 1.3):
    # Log request information
    logging.info(f"Received request: SMILES = {smiles.smiles}, sigma = {sigma}")
    
    try:
        # Convert SMILES to molfile
        molfile_request = SMILES_to_molfile(smiles.smiles)
        molfile = molfile_request.molfile
        
        # Get included and excluded elements
        included, excluded = get_atoms(molfile)
        
        # Calculate binding energies
        be_predictions = molfile_to_BE(molfile)
        
        # Generate predicted XPS spectrum
        predicted_spectrum = be_to_spectrum(be_predictions, sigma = sigma)
        
        #
        # Create and return the full prediction response
        response = SpectralPrediction(
            molfile = molfile,
            smiles = smiles.smiles,
            elementsIncluded = included,
            elementsExcluded = excluded,
            bindingEnergies = be_predictions,
            spectrum = predicted_spectrum
        )
        
        # Log successful completion
        logging.info("Returning full prediction response.")
        return response

    except ValidationError as ve:
        # Handle input validation errors
        logging.error(f"Input validation error: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {ve}")

    except Exception as e:
        # Handle unexpected errors
        logging.error(f"An error occurred during processing: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")
    
    
#MM
@app.post("/BEfromSMILES", response_model=BEPrediction)
def BEfromSMILES(smiles: SMILES):
    # Log request information
    logging.info(f"Received request: SMILES = {smiles.smiles}")
    
    try:
        # Convert SMILES to molfile
        molfile_request = SMILES_to_molfile(smiles.smiles)
        molfile = molfile_request.molfile
        logging.info("Successfully converted SMILES to molfile.")
        
        # Get included and excluded elements
        included, excluded = get_atoms(molfile)
        logging.info(f"Included elements: {included}, Excluded elements: {excluded}")
        
        # Calculate binding energies
        be_predictions = molfile_to_BE(molfile)
        logging.info(f"Calculated binding energies: {be_predictions}")
        
        # Create and return the full prediction response
        response = BEPrediction(
            molfile = molfile,
            smiles = smiles.smiles,
            elementsIncluded = included,
            elementsExcluded = excluded,
            bindingEnergies = be_predictions
        )
        
        # Log successful completion
        logging.info("Returning full prediction response.")
        return response

    except ValidationError as ve:
        # Handle input validation errors
        logging.error(f"Input validation error: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {ve}")

    except Exception as e:
        # Handle unexpected errors
        logging.error(f"An error occurred during processing: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")