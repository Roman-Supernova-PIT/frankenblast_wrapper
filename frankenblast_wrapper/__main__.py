#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 16:23:05 2026

@author: hmfausey
"""

import argparse
import run_frankenblast
 
def main():
    parser = argparse.ArgumentParser(
        description="Run Frankenblast on a single galaxy"
    )
     
    parser.add_argument("--diaobject-id", required=True, type=int)
    parser.add_argument("--provenance-id", required=True, type=int)
    parser.add_argument("--galid", required=True, type=int)
    parser.add_argument("--field", required=True)
     
    parser.add_argument("--ra", required=True, type=float)
    parser.add_argument("--dec", required=True, type=float)
     
    parser.add_argument("--redshift", required=True, type=float)
     
    parser.add_argument("--magnitudes", required=True)
    parser.add_argument("--magerrs", required=True)
     
    parser.add_argument("--lsst-data", required=True)
    parser.add_argument("--subaru-data", required=True)
     
    parser.add_argument(
        "--filetype",
        default="parquet",
        choices=["parquet", "csv"],
    )
     
    args = parser.parse_args()
     
    run_frankenblast.run(
        diaobject_id=args.diaobject_id,
        provenance_id=args.provenance_id,
        galID=args.galid,
        field=args.field,
        RA=args.ra,
        Dec=args.dec,
        magnitudes=args.magnitudes,
        magerrs=args.magerrs,
        LSST_data=args.lsst_data,
        subaru_data=args.subaru_data,
        redshift=args.redshift,
        filetype=args.filetype,
    )
 
if __name__ == "__main__":
    main()
    