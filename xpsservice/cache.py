# -*- coding: utf-8 -*-
import os

from diskcache import Cache

CACHE_DIR = os.getenv("CACHEDIR", "xpscache")

# 2 ** 30 = 1 GB

opt_cache = Cache(CACHE_DIR, size_limit=2 ** 30, disk_min_file_size=0)
conformer_cache = Cache(CACHE_DIR, size_limit=2 ** 30, disk_min_file_size=0)
soap_config_cache = Cache(CACHE_DIR, size_limit=2 ** 30, disk_min_file_size=0)
model_cache = Cache(CACHE_DIR, size_limit=2 ** 30, disk_min_file_size=0)


if __name__ == "__main__":
   
    opt_cache.clear()
    conformer_cache.clear()
    soap_config_cache.clear()
    model_cache.clear()