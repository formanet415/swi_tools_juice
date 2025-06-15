# swi_tools_juice
Python library for calibrating and analyzing data from the Submillimetre Wave Instrument (SWI) onboard ESA's JUICE spacecraft, developed during a Paris Observatory M1 internship at LIRA.

## Overview

This repository contains tools for:
- Loading and analyzing SWI Level 1 (L1) calibrated data
- Estimating beam parameters from limb and mapping observations
- Performing system temperature analysis on L01B data (limited capabilities)
- Comparing observations with analytical antenna models

## Repository Contents

### `switools.py`
The main analysis library. It provides:
- Tools for beam fitting (Gaussian, analytical, error function)
- Systematic extraction of FWHM, pointing offsets, and defocus
- Plotting functions for continuum channels and derived profiles
- Model comparison routines (Gaussian vs. Bessel-based)

### `swincloadobsid.py`
A standalone utility for loading **L01B raw data**.  
It is more limited than `switools.py` and intended mainly for:
- Basic handling of uncalibrated SWI observations
- Estimating system temperature using hot/cold load measurements
