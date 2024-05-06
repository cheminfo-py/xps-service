# -*- coding: utf-8 -*-
import os

from diskcache import Cache

CACHE_DIR = os.getenv("CACHEDIR", "xpscache")

# 2 ** 30 = 1 GB

ir_cache = Cache(CACHE_DIR, size_limit=2 ** 30, disk_min_file_size=0)
ir_from_smiles_cache = Cache(CACHE_DIR, size_limit=2 ** 30, disk_min_file_size=0)
ir_from_molfile_cache = Cache(CACHE_DIR, size_limit=2 ** 30, disk_min_file_size=0)

opt_cache = Cache(CACHE_DIR, size_limit=2 ** 30, disk_min_file_size=0)
conformer_cache = Cache(CACHE_DIR, size_limit=2 ** 30, disk_min_file_size=0)


soap_config_cache = Cache(CACHE_DIR, size_limit=2 ** 30, disk_min_file_size=0)
soap_descriptor_cache = Cache(CACHE_DIR, size_limit=2 ** 30, disk_min_file_size=0)
model_cache = Cache(CACHE_DIR, size_limit=2 ** 30, disk_min_file_size=0)

xps_from_molfile_cache = Cache(CACHE_DIR, size_limit=2 ** 30, disk_min_file_size=0)


if __name__ == "__main__":
    
    ir_cache.clear()
    ir_from_smiles_cache.clear()
    ir_from_molfile_cache.clear()

    opt_cache.clear()
    conformer_cache.clear()
    soap_config_cache.clear()
    soap_descriptor_cache.clear()
    model_cache.clear()
    xps_from_molfile_cache.clear()
