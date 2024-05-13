# -*- coding: utf-8 -*-
import os

from fastapi.logger import logger
from typing import Dict, List, Optional, Set, Any


#Conformer generation method
ALLOWED_FF = ("uff", "mmff94", "mmff94s")
MAX_ATOMS_FF = int(os.getenv("MAX_ATOMS_FF", 100))


#XTB optimization method
ALLOWED_METHODS = ("GFNFF", "GFN2xTB", "GFN1xTB")
MAX_ATOMS_XTB = int(os.getenv("MAX_ATOMS_XTB", 100))
ALLOWED_FMAX = (0.000001, 0.1)


#timeout for the overall calculation
TIMEOUT = int(os.getenv("TIMEOUT", 100))


# Define transition_map dictionary(list of orbitals for which a photoelectron emission is calculated)
# Several maps possible. Adjust which one to load in xpsservice
transition_map = {
    "C1s": {
        "element": "C",
        "orbital": "1s",
        "soap_config_filepath": os.path.abspath("SOAP_configs/soap_config_C1s.pkl"),
        "model_filepath": os.path.abspath("ML_models/XPS_GPR_C1s_xtb.pkl")
    },
    "O1s": {
        "element": "O",
        "orbital": "1s",
        "soap_config_filepath": os.path.abspath("SOAP_configs/soap_config_O1s.pkl"),
        "model_filepath": os.path.abspath("ML_models/XPS_GPR_O1s_xtb.pkl")
    }
}


# Derive allowed elements from transition_map
def derive_allowed_elements(transition_map: dict) -> Set[str]:
    allowed_elements = {info["element"] for info in transition_map .values()}
    return allowed_elements


#From transition map
ALLOWED_ELEMENTS = derive_allowed_elements(transition_map)


logger.info(
    f"Settings: MAX_ATOMS_XTB: {MAX_ATOMS_XTB}, TIMEOUT: {TIMEOUT}, ALLOWED_ELEMENTS: {ALLOWED_ELEMENTS}, ALLOWED_METHODS: {ALLOWED_METHODS}, ALLOWED_FMAX: {ALLOWED_FMAX}"
)
