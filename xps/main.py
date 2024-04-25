from fastapi import FastAPI, HTTPException
import logging

from xps.models.models import *
from xps.predict.xps_predictions import molfile_to_BE, be_to_spectrum, SMILES_to_molfile, get_atoms, smiles_to_BE, xtb_opt_from_ase

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

@app.post("/test", 
         )
def test(test):
    return test


@app.get("/ping")
def ping():
    return {"message": "backtopong"}


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


@app.post("/v1/fromSMILES", response_model=FullPrediction)
def fromSMILES(smiles: SMILES, sigma=0.35):
    # Log request information
    logging.info(f"Received request: SMILES = {smiles.smiles}, sigma = {sigma}")
    
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
        
        # Generate predicted XPS spectrum
        predicted_spectrum = be_to_spectrum(be_predictions, sigma=sigma)
        logging.info("Generated predicted XPS spectrum.")
        
        #
        # Create and return the full prediction response
        response = FullPrediction(
            molfile=molfile,
            smiles=smiles.smiles,
            elementsIncluded=included,
            elementsExcluded=excluded,
            bindingEnergies=be_predictions,
            spectrum=predicted_spectrum
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


    
    
# Define a simple FastAPI function that returns a list
@app.get("/simple_list", response_model=ListResponse)
def simple_list():
    # Return a list of integers as an example
    return ListResponse(items=[1, 2, 3, 4, 5])    
    
@app.get("/test_simple", response_model=SimpleResponse)
def test_simple():
    # Define the response data
    data = [1, 2, 3, 4, 5, 6]

    # Return the response
    return SimpleResponse(
        message="Simple function response",
        data=data
    )
    


@app.post("/predict_be/")
async def predict_be(request: BERequest) -> BEResponse:
    smiles = request.smiles
    try:
        # Calculate binding energies
        response = smiles_to_BE(smiles)
        return response
    except Exception as e:
        # If any error occurs, return a 400 HTTP status code
        raise HTTPException(status_code=400, detail=str(e))

#working
@app.post("/get_molfile/")
async def get_molfile(request: BERequest) -> MolfileRequest:
    smiles = request.smiles
    try:
        # Calculate binding energies
        response = SMILES_to_molfile(smiles)
        return response
    except Exception as e:
        # If any error occurs, return a 400 HTTP status code
        raise HTTPException(status_code=400, detail=str(e))
