# L1 handling library
import xarray as xr
import numpy as np
import os
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import erf
from matplotlib.patches  import Circle
from matplotlib import ticker
import matplotlib.pyplot as plt
from scipy import constants as spc
c_kmps = spc.speed_of_light/1e3
import matplotlib.dates as mdates
from scipy.special import jv  # Bessel functions of the first kind
from scipy.integrate import quad  # Numerical integration

"""
Author: Tomáš Formanek (tomas.formanek@obspm.fr)
Paris Observatory, 2025
Supervised by Raphaël Moreno
"""


def load(obsid, version='max'):
    """
    Loads L1 data for a given obsID
    input:
        obsid: obsID to load
    output:
        ds: xarray dataset with L1 data
    """
    # look for the L1 data file in DR-SWI/database-L1/L1_<obsid>*.nc
    # load the file with xarray
    absoluteorrelativedirectory = 'DR-SWI/database-L1/'
    if version=='max':
        version=''

    ext = version+'.nc'
    if obsid==228:
        ext = 'TAC.nc' # it is likely this can be removed at some point in the future
    fnames = os.listdir(absoluteorrelativedirectory)
    fnames = [f for f in fnames if f.startswith('L1_') and f.endswith(ext)]
    fnames = [f for f in fnames if str(obsid) in f]
    if len(fnames) == 0:
        raise ValueError(f'No L1 data found for obsID {obsid}')
    elif len(fnames) > 1:
        #raise ValueError(f'Multiple L1 data files found for obsID {obsid}: {fnames}')
        # choose the first one and print a warning
        print(f'Warning: Multiple L1 data files found for obsID {obsid}: {fnames}')
        print(f'Using {fnames[0]}')
        
    
    fname = fnames[0]

    ds = xr.open_dataset(os.path.join(absoluteorrelativedirectory, fname))
    print("Loaded",fname)
    return ds

def get_SWI_BORESIGHT_TCA(ds,R=None):
    """
    Gets the SWI boresight Target Center Angle (TCA) - developed for v0.1 data
    input:
        ds: xarray dataset with L1 data
        R: radius of the target body (default: 1737.4 km for the Moon)
    output:
        tca: boresight target center angle, unit: degrees
        A/2: boresight angular radius, unit: degrees
    """
    if not R:
        if ds.TARGET.data[0] == 'MOON':
            R = 1737.4 # IAU radius (JPL)
        elif ds.TARGET.data[0] == 'EARTH':
            R = 6371.01 # Vol. mean radius (JPL)
    TP = ds.TP_ALTITUDE.data # defined if boresight misses the target
    ALT = ds.ALTITUDE.data
    thetaOFF = np.rad2deg(np.arcsin((R+TP)/(R+ALT))) # based on TP
    SBE = ds.SWI_BORESIGHT_EMI.data
    thetaON = np.rad2deg( np.arcsin(R*(np.sin(np.deg2rad(180-SBE)))/(R+ALT)) ) # Sine law
    # use thetaOFF if boresight misses the target and thetaON if boresight hits the target
    tca = np.where(np.isnan(thetaOFF), thetaON, thetaOFF)

    A = ds.ANGULAR.data
    
    return tca, A/2

def get_observation_summary(obsids=[375, 376, 377, 378]):
    """
    Prints a LaTeX table of the observation summary with parameters:
    Target, Date & Time [UTC], Distance [km], Angular [arcmin], Polar angle [deg], Phase angle [deg], Radial vel. [km/s], LO1 [GHz], LO2 [GHz], T_ON [s], 
    AT midpoint [arcmin], CT midpoint [arcmin], SUB SC LAT [deg], SUB SC LON [deg], SUB SUN LAT [deg], SUB SUN LON [deg], map resolution, sapling [arcmin]
    input:
        obsids: list of obsids to load
    output:
        None - prints the LaTeX table to stdout
    """

    # it makes sense to have the columns be obsids and the rows be the parameters
    # We start by making a skeleton with the first column containing the parameter names
    import pandas as pd
    
    data = {
        'ObsID': [r'Target', r'Date', r'Start time (UTC)', 'Duration (s)', r'Distance (km)', r"Angular ($\mathrm{'}$)", r"\acrshort{PSR}", r'Polar angle ($^\circ$)', r'Phase angle ($^\circ$)', 
                  r'Radial vel. (km/s)', r'LO1 (GHz)', r'LO2 (GHz)',r'Molec1',r'Molec2', r'$\mathrm{T_{ON}}$ (s)', r'$\mathrm{T_{ON}}$ CCH (s)', r"AT OFFSET ($^\circ$)", r"CT OFFSET ($^\circ$)",
                  r"SUB SC LAT ($^\circ$)", r'SUB SC LON ($^\circ$)', r'SUB SUN LAT ($^\circ$)', r'SUB SUN LON ($^\circ$)', 
                    r"Map dimension", r"AT Stepsize ($\mathrm{'}$)", r"CT Stepsize ($\mathrm{'}$)"]
    }
    # Now we will fill in the data for each obsid
    for obsid in obsids:
        ds = load(obsid)
        # Get the data for each parameter
        target = ds.TARGET.data[0]
        fulldate = ds.DATES.data[0]
        # split the date into date and time
        date = fulldate.split('T')[0]
        time = fulldate.split('T')[1]
        # remove decimal point from time
        time = time.split('.')[0]
        duration = float(ds.DURATION.data[0])*1e-9
        distance = f"${(ds.DISTANCE.data[0]/1000):.0f}"+r"\times10^3$" # formatted string with 2 decimal places
        angular = float(ds.ANGULAR.data[0]*60) # convert to arcmin
        psr = angular / 8.6 # value from NECP
        polar_angle = float(ds.POLAR_ANG.data[0])
        phase_angle = float(ds.PHASE.data[0])
        radial_velocity = float(ds.VEL_RADIAL.data[0])
        # convert to formatted string with 3 decimal places
        radial_velocity = "{:.3f}".format(radial_velocity)
        lo1 = float(ds.LO1.data[0])
        lo2 = float(ds.LO2.data[0])
        molec1 = ds.MOLEC1.data[0]
        import re
        pattern = r'T-[\w\+\-]+-\d+[UL]+'
        # find all matches
        matches = re.findall(pattern, molec1)
        # keep the first match
        if len(matches) > 0:
            molec1 = matches[0][2:]
        else:
            molec1 = 'unknown'
            print(molec1)
        print(molec1)
        # now we need to find the second molecule
        molec2 = ds.MOLEC2.data[0]
        # find all matches
        matches = re.findall(pattern, molec2)
        # keep the first match
        if len(matches) > 0:
            molec2 = matches[0][2:]
        else:
            print(molec2)
            molec2 = 'unknown'
            
        t_on = float(ds.T_ON.data[0]) *1e-9 # convert to seconds - formatted string with 0 decimal places
        t_on = "{:.1f}".format(t_on)
        try:
            t_on_CCH = float(ds.T_ON_CCH.data[0]) *1e-9 # convert to seconds - formatted string with 0 decimal places
        except:
            t_on_CCH = np.nan
        t_on_CCH = "{:.3f}".format(t_on_CCH)
        ATmid = float(ds.AT_OFFSET_OBS.data[0])
        CTmid = float(ds.CT_OFFSET_OBS.data[0])
        SUB_SC_LAT = float(ds.SUB_SC_LAT.data[0])
        SUB_SC_LON = float(ds.SUB_SC_LON.data[0])
        SUB_SUN_LAT = float(ds.SUB_SUN_LAT.data[0])
        SUB_SUN_LON = float(ds.SUB_SUN_LON.data[0])
        # Map resolution and sampling
        try:
            MapRes = ds.MAP_DIM.data[0]
            SamplingAT = float(ds.MAP_AT_STEPSIZE.data[0])
            SamplingCT = float(ds.MAP_CT_STEPSIZE.data[0])
        except:
            MapRes = np.nan
            SamplingAT = np.nan
            SamplingCT = np.nan
        


        # Add the data to the dictionary
        data[obsid] = [target, date, time, duration, distance, angular, psr, polar_angle, phase_angle, radial_velocity, lo1, lo2, molec1, molec2, t_on, t_on_CCH, ATmid, CTmid,
                      SUB_SC_LAT, SUB_SC_LON, SUB_SUN_LAT, SUB_SUN_LON, MapRes, SamplingAT, SamplingCT]

    # print the LaTeX table
    df = pd.DataFrame(data)
    # convert the dataframe to a LaTeX table, keep the formatting
    latex_table = df.to_latex(index=False, float_format="%.2f", escape=False)
    # print the LaTeX table
    print(latex_table)

def Gauss(x, A, mu, sigma):
    """
    Gaussian function
    input:
        x: independent variable
        A: amplitude of the Gaussian
        mu: mean of the Gaussian
        sigma: standard deviation of the Gaussian
    output:
        Gaussian function evaluated at x
    """
    return A/(sigma*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu)/sigma)**2)

def get_sigma_threestencil(x, y, z, hd, hs, sigma_x, sigma_y, sigma_z, sigma_hd=0, sigma_hs=0):
    """
    Calculates the error propagation for the three point stencil method
    input:
        x: first point value
        y: second point value
        z: third point value
        hd: distance between the first and second point
        hs: distance between the second and third point
        sigma_x: error of the first point
        sigma_y: error of the second point
        sigma_z: error of the third point
        sigma_hd: error of the distance between the first and second point
        sigma_hs: error of the distance between the second and third point
    output:
        sigma: error of the three point stencil method
    """
    # Petr Kos error propagation software
    return np.sqrt(hd**2*sigma_z**2/(hs**2*(hd + hs)**2) + sigma_hd**2*((2*hd*y - 2*hd*z)/(hd*hs*(hd + hs)) - (-hd**2*z + hs**2*x + y*(hd**2 - hs**2))/(hd*hs*(hd + hs)**2) - (-hd**2*z + hs**2*x + y*(hd**2 - hs**2))/(hd**2*hs*(hd + hs)))**2 + sigma_hs**2*((2*hs*x - 2*hs*y)/(hd*hs*(hd + hs)) - (-hd**2*z + hs**2*x + y*(hd**2 - hs**2))/(hd*hs*(hd + hs)**2) - (-hd**2*z + hs**2*x + y*(hd**2 - hs**2))/(hd*hs**2*(hd + hs)))**2 + hs**2*sigma_x**2/(hd**2*(hd + hs)**2) + sigma_y**2*(hd**2 - hs**2)**2/(hd**2*hs**2*(hd + hs)**2))

def threepointstencil(x, y, sy):
    """
    Calculate derivative using the three point stencil method - uncertainties included
    input:
        x: x axis values
        y: y axis values
        sy: error of the y axis values
    output:
        dydx: derivative of y with respect to x (smaller in dimension by 2)
        sdydx: error of the derivative (smaller in dimension by 2)
    """
    # Ensure sy is an array of the same size as x
    if isinstance(sy, (int, float)):
        sy = np.full_like(x, sy, dtype=float)
    dydx = np.zeros(y.size-2)
    sdydx = np.zeros(y.size-2)
    for i in range(dydx.size):
        hs = x[i+1]-x[i]
        hd = x[i+2]-x[i+1]
        dydx[i] = (hs**2*y[i+2] + (hd**2-hs**2)*y[i+1] - hd**2*y[i])/(hs*hd*(hs+hd))
        sdydx[i] = get_sigma_threestencil(y[i], y[i+1], y[i+2], hd, hs, sy[i], sy[i+1], sy[i+2])
    return dydx, sdydx

def determineSNR(ds):
    """
    Based on the integration and bandwidth, this function determines if CCH or integrated CTS has a better SNR
    input:
        ds: xarray data structure with L1 data
    output:
    Returns:
        DeltaTCCH1 and DeltaTCCH2: estimated RMS noise for CCH channels
        betterSNR: a string indicating which channel is expected to have a better SNR ("CCH" or "CTS")
        Prints out the estimated RMS noise for both channels and a recommendation
    """
    try:
        DeltatCCH = float(ds.T_ON_CCH.data[0]) * 1e-9
    except:
        DeltatCCH = float(ds.T_ON.data[0]) * 1e-9*0
    DeltatCTS = float(ds.T_ON.data[0]) * 1e-9

    DeltanuCCH = 5e9
    DeltanuCTS = 1e9
    approxTSYS1 = 1300  # for channel 1
    approxTSYS2 = 3000  # for channel 2
    try:
        if np.max(ds.CCH1)<1:
            approxTSYS1=1 # v02 data uncalibrated L1
            approxTSYS2=1 # v02 data uncalibrated L1
    except:
        # this probably means CCH1 is not a field or empty
        # we dont care
        pass

    DeltaTCCH1 = 2 * approxTSYS1 / np.sqrt(DeltanuCCH * DeltatCCH)
    DeltaTCTS1 = 2 * approxTSYS1 / np.sqrt(DeltanuCTS * DeltatCTS)
    DeltaTCCH2 = 2 * approxTSYS2 / np.sqrt(DeltanuCCH * DeltatCCH)
    DeltaTCTS2 = 2 * approxTSYS2 / np.sqrt(DeltanuCTS * DeltatCTS)

    # Print out results and recommendation for channel 1
    print(f"Estimated RMS noise for CCH (Channel 1): {DeltaTCCH1:.2f} K")
    print(f"Estimated RMS noise for CTS (Channel 1): {DeltaTCTS1:.2f} K")
    if DeltaTCCH1 < DeltaTCTS1:
        print("CCH is expected to have a better SNR for Channel 1.")
        return DeltaTCCH1, DeltaTCCH2, "CCH"
    else:
        print("CTS is expected to have a better SNR for Channel 1.")
        return DeltaTCTS1, DeltaTCTS2, "CTS"

    

def fit_main_beam_flyby(ds, start, end, nskyindexes=15, plotstuff=False, verbose=True, useCTS=False, refLO1 = 562.95, refLO2 = 1119.33, usesurrounding = (15,10)):
    """
    Fit the main beam for flyby observations.
    input:
        ds: xarray dataset with L1 data
        start: start index for data selection
        end: end index for data selection
        nskyindexes: number of sky indexes to use for the fit (default: 15)
        plotstuff: plot the data and the fit (default: False)
        verbose: print the fit parameters (default: True)
    output:
        FWHPCCH1: FWHM of the Gaussian fit for CCH1
        FWHPCCH2: FWHM of the Gaussian fit for CCH2
        sigmaFWHMCCH1: sigma of the Gaussian fit for CCH1
        sigmaFWHMCCH2: sigma of the Gaussian fit for CCH2
    """

    # Inaccurate, use 
    # TCA, A2 = get_SWI_BORESIGHT_TCA(ds)
    TCA = np.sqrt((ds.AT.data)**2 + (ds.CT.data)**2)
    A2 = ds.ANGULAR.data/2
    
    # Determine theoretically, which should have a better SNR:
    if verbose: 
        determineSNR(ds)

    if useCTS:
        CH1 = np.mean(ds.CTS1.data, axis=1)
        CH2 = np.mean(ds.CTS2.data, axis=1)
        choice = 'CTS'
    else:
        CH1 = ds.CCH1.data
        CH2 = ds.CCH2.data
        choice = 'CCH'
    
    # crop the values to only take into account records between start and end
    # This allows this function to be called by the 1D map (2D OTF still) - to be implemented
    variables = [TCA, A2, CH1, CH2] 
    TCA, A2, CH1, CH2 = map(lambda x: x[start:end], variables)

    sortidx = np.argsort(TCA)
    TCA = TCA[sortidx]
    A2 = A2[sortidx]
    theta = (TCA - A2)*60 # convert to arcmin
    CH1 = CH1[sortidx]
    CH2 = CH2[sortidx]

    plotparams = {'type': choice, 'channel': '1', 'LO': ds.LO1.data[0], 'title': 'SWA LEGA', 'obsid': ds.OBSID.data[0]}
    FWHPCCH1, sigmaFWHPCCH1 = erfit_main_beam(CH1, theta, nskyindexes=nskyindexes, plotstuff=plotstuff, verbose=verbose, plotparams=plotparams, usesurrounding = usesurrounding[0])
    plotparams = {'type': choice, 'channel': '2', 'LO': ds.LO2.data[0], 'title': 'SWA LEGA', 'obsid': ds.OBSID.data[0]}    
    FWHPCCH2, sigmaFWHPCCH2 = erfit_main_beam(CH2, theta, nskyindexes=nskyindexes, plotstuff=plotstuff, verbose=verbose, plotparams=plotparams, usesurrounding = usesurrounding[1])

    # print the results
    factor1 = ds.LO1.data[0]/refLO1
    factor2 = ds.LO2.data[0]/refLO2
    
    print("= ObsID:", ds.OBSID.data[0],"============")
    print("FWHM CCH1: {:.5f} ± {:.5f}".format(FWHPCCH1, sigmaFWHPCCH1))
    print("FWHM CCH2: {:.5f} ± {:.5f}".format(FWHPCCH2, sigmaFWHPCCH2))
    print("- Rescaled: ---------------")
    print("FWHM CCH1: {:.5f} ± {:.5f}".format(FWHPCCH1*factor1, sigmaFWHPCCH1*factor1))
    print("FWHM CCH2: {:.5f} ± {:.5f}".format(FWHPCCH2*factor2, sigmaFWHPCCH2*factor2))
    print("===========================")

    return FWHPCCH1, FWHPCCH2, sigmaFWHPCCH1, sigmaFWHPCCH2

def smooth_1D_disk(x, x0, R, A, epsilon):
    """
    Smooth 1D disk function for fitting main beam data.
    input:
        x: independent variable (angular offset in arcmin)
        x0: center position of the disk
        R: radius of the disk in arcmin
        A: amplitude of the disk (on target antenna temperature)
        epsilon: smoothing parameter
    output:
        Evaluated value at x
    """
    r = np.abs(x - x0)
    return A / np.abs(1 + np.exp((r - R) / epsilon))

def fit_main_beam_1D(ds, verbose = False, choice="CCH", plotstuff=False, lim=150, usesurrounding=[10,5]):
    """
    Fit the main beam on 2D OTF data which consists of 1D scans.

    input:
        ds: xarray dataset containing L1 data, including angular offsets, continuum data, and other relevant parameters.
        verbose: print detailed information during the fitting process (default: False).
        choice: data type to use for fitting, either "CCH" or "CTS" (default: "CCH").
        plotstuff: whether to generate plots for the fitting process (default: False).
        lim: limit for AT or CT axis in arcmin (default: 150).

    output:
        FWHM1: Full Width at Half Maximum (FWHM) for CCH1 for positive and negative scans.
        FWHM2: Full Width at Half Maximum (FWHM) for CCH2 for positive and negative scans.
        sigma_FWHM1: Uncertainty in FWHM for CCH1 for positive and negative scans.
        sigma_FWHM2: Uncertainty in FWHM for CCH2 for positive and negative scans.
        x0_CH1: Fitted center position for CH1.
        sigma_x0_CH1: Uncertainty in the fitted center position for CH1.
        x0_CH2: Fitted center position for CH2.
        sigma_x0_CH2: Uncertainty in the fitted center position for CH2.
        LO1: Local oscillator freq in GHz
        LO2: Local oscillator freq in GHz  
    """

    if verbose: 
        determineSNR(ds)

    if choice=="CTS":
        CH1 = np.mean(ds.CTS1.data, axis=1)
        CH2 = np.mean(ds.CTS2.data, axis=1)
    else:
        CH1 = ds.CCH1.data
        CH2 = ds.CCH2.data
        choice = 'CCH'

    # figure out if this is an AT or CT scan
    meanDeltaAT = np.median(np.diff(ds.AT.data))
    meanDeltaCT = np.median(np.diff(ds.CT.data))
    
    if np.abs(meanDeltaAT)>np.abs(meanDeltaCT):
        # AT scan
        scan = "AT"
        DeltaTheta = meanDeltaAT
        signedTheta = ds.AT.data
    else:
        # CT scan
        scan = "CT"
        DeltaTheta = meanDeltaCT
        signedTheta = ds.CT.data

    TCA, A2 = get_SWI_BORESIGHT_TCA(ds)

    FWHMCCH1, FWHMCCH2, sigmaFWHMCCH1, sigmaFWHMCCH2 = map(lambda _: np.zeros(2), range(4))
    fit_params_CCH1, fit_params_CCH2 = [], []  # To store all fit parameters
    # divide the scan and fit both parts, exclude outer most two records
    mask1 = signedTheta > 0
    for i, mask in enumerate([mask1, ~mask1]):
        sort = np.argsort(TCA[mask])
        theta = (TCA[mask][sort] - A2[mask][sort]) * 60  # arcmin
        C1, C2 = map(lambda x: x[mask][sort]/np.max(x), [CH1, CH2])

        try:
            #plt.figure()
            FWHMCCH1[i], sigmaFWHMCCH1[i], fit1, perr1 = erfit_main_beam(C1, theta, nskyindexes=15, normalise=False, plotstuff=False, verbose=verbose, usesurrounding=usesurrounding[0], returnallfitparams=True)
            #plt.show()
            #plt.figure()
            FWHMCCH2[i], sigmaFWHMCCH2[i], fit2, perr2 = erfit_main_beam(C2, theta, nskyindexes=15, normalise=False, plotstuff=False, verbose=verbose, usesurrounding=usesurrounding[1], returnallfitparams=True)
            #plt.show()
            fit_params_CCH1.append((fit1, perr1))
            fit_params_CCH2.append((fit2, perr2))

        except:
            print(f"Sampling too low for obsid {ds.OBSID.data[0]}")
            continue
        # Fit center position using smooth_1D_disk


    # Initial guesses for the fit parameters: x0, R, A, epsilon
    initial_guess = [0, 60*A2[0], np.max(CH1), 1]

    # Define bounds for the fit parameters: [x0_min, R_min, A_min, epsilon_min], [x0_max, R_max, A_max, epsilon_max]
    bounds = ([-200, 0, 0, 0], [200, 3*A2[0]*60, 50000, 6])

    fitstart = 1
    fitend = -2
    # Fit for CH1
    sorted_indices = np.argsort(signedTheta)
    print(initial_guess)
    popt_CH1, pcov_CH1 = curve_fit(
        smooth_1D_disk, 
        signedTheta[sorted_indices][fitstart:fitend]*60, 
        CH1[sorted_indices][fitstart:fitend], 
        p0=initial_guess, 
        bounds=bounds
    )
    x0_CH1, R_CH1, A_CH1, epsilon_CH1 = popt_CH1
    perr_CH1 = np.sqrt(np.diag(pcov_CH1))

    # Update initial guess for CH2
    initial_guess[2] = np.max(CH2)

    # Fit for CH2
    popt_CH2, pcov_CH2 = curve_fit(
        smooth_1D_disk, 
        signedTheta[np.argsort(signedTheta)][fitstart:fitend]*60, 
        CH2[np.argsort(signedTheta)][fitstart:fitend], 
        p0=initial_guess, 
        bounds=bounds
    )
    x0_CH2, R_CH2, A_CH2, epsilon_CH2 = popt_CH2
    perr_CH2 = np.sqrt(np.diag(pcov_CH2))
    
    # Print the results
    print("Fit results for CH1:")
    print(f"x0 = {x0_CH1:.2f} ± {perr_CH1[0]:.2f}, R = {R_CH1:.2f} ± {perr_CH1[1]:.2f}, A = {A_CH1:.2f} ± {perr_CH1[2]:.2f}, epsilon = {epsilon_CH1:.2f} ± {perr_CH1[3]:.2f}")
    print("Fit results for CH2:")
    print(f"x0 = {x0_CH2:.2f} ± {perr_CH2[0]:.2f}, R = {R_CH2:.2f} ± {perr_CH2[1]:.2f}, A = {A_CH2:.2f} ± {perr_CH2[2]:.2f}, epsilon = {epsilon_CH2:.2f} ± {perr_CH2[3]:.2f}")

    if plotstuff:
        # Simple plotter for the data and the fitted model
        plt.figure(figsize=(8, 6))
        plt.plot(signedTheta*60, CH1, 'o', label=f"{choice}1 Data", color="blue")
        plt.plot(signedTheta*60, CH2, 'o', label=f"{choice}2 Data", color="green")
        plt.plot(signedTheta[np.argsort(signedTheta)]*60, smooth_1D_disk(signedTheta[np.argsort(signedTheta)]*60, *popt_CH1), label=f"{choice}1 Fit", color="red", linestyle="--")
        plt.plot(signedTheta[np.argsort(signedTheta)]*60, smooth_1D_disk(signedTheta[np.argsort(signedTheta)]*60, *popt_CH2), label=f"{choice}2 Fit", color="orange", linestyle="--")
        plt.xlabel(r"$\theta$ (')")
        plt.ylabel(r"$T_{RJ}/T_{sys}$")
        plt.title(f"Fit of the center position of 1D scan, ObsID: {ds.OBSID.data[0]}")
        plt.legend()
        plt.grid()
        plt.savefig(f"Figures/CenterPosition1DScan_{scan}_{int(ds.OBSID.data[0])}.png", dpi=300, bbox_inches='tight')
        plt.show()

    # Print the results
    print("= ObsID: {:.0f} {:2s} ===========".format(ds.OBSID.data[0],scan))
    print(f"FWHM {choice}"+"(positive scan): {:.5f} ± {:.5f}".format(FWHMCCH1[0], sigmaFWHMCCH1[0]))
    print(f"FWHM {choice}"+"(negative scan): {:.5f} ± {:.5f}".format(FWHMCCH1[1], sigmaFWHMCCH1[1]))
    print(f"FWHM {choice}"+"(positive scan): {:.5f} ± {:.5f}".format(FWHMCCH2[0], sigmaFWHMCCH2[0]))
    print(f"FWHM {choice}"+"(negative scan): {:.5f} ± {:.5f}".format(FWHMCCH2[1], sigmaFWHMCCH2[1]))
    print("===========================")

    if plotstuff:
        # Plotting - use the plot continuum with returning axes and then, overplot the fitted model
        axes = plot_continuum(ds, vs=scan, choice=choice, lim=lim, dontshow=True)

        # Overplot the fitted model for Ch1 and Ch2
        for channel, fit_params, ax in zip(["CCH1", "CCH2"], [fit_params_CCH1, fit_params_CCH2], axes):
            for i, (fit, _) in enumerate(fit_params):
                # Generate a uniform range of theta values for plotting
                fitrange = usesurrounding[channel==["CCH1", "CCH2"]]
                uniform_theta = np.linspace(-fitrange, +fitrange, 1000)
                # Compute the fitted model using the error function
                fitted_model = errorfunct(uniform_theta, *fit)
                # Plot the fitted model
                ax.plot((-1)**i*(uniform_theta+A2[0]*60), fitted_model, label=f"Fitted Model {channel} ({['$+$','$-$'][i]})", linestyle="--")

        # Add legends and FWHM text
        for i, ax in enumerate(axes):
            FWHM_positive = FWHMCCH1[0] if i == 0 else FWHMCCH2[0]
            sigmaFWHM_positive = sigmaFWHMCCH1[0] if i == 0 else sigmaFWHMCCH2[0]
            FWHM_negative = FWHMCCH1[1] if i == 0 else FWHMCCH2[1]
            sigmaFWHM_negative = sigmaFWHMCCH1[1] if i == 0 else sigmaFWHMCCH2[1]
            ax.text(0.35, 0.4, rf"FWHM$_+$ = ({FWHM_positive:.1f}$\pm${sigmaFWHM_positive:.1f})'"+f"\n"+
                                 rf"FWHM$_-$ = ({FWHM_negative:.1f}$\pm${sigmaFWHM_negative:.1f})'",
                    transform=ax.transAxes, fontsize=10, verticalalignment='top', 
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
            
        # Fix suptitle
        plt.subplots_adjust(top=0.9)  # Adjust the top padding to move the suptitle closer to the plot

        # Add legends and save the plot
        axes[0].legend(loc=8,fontsize=9)
        axes[1].legend(loc=8,fontsize=9)
        plt.savefig(f"Figures/Fitted1DScan_{scan}_{int(ds.OBSID.data[0])}.png", dpi=300, bbox_inches='tight')
        plt.show()
    

    return FWHMCCH1, FWHMCCH2, sigmaFWHMCCH1, sigmaFWHMCCH2, x0_CH1, perr_CH1[0], x0_CH2, perr_CH2[0], scan, ds.LO1.data[0], ds.LO2.data[0]


def fit_main_beam_2D(ds, nskyindexes=15, plotstuff=False, verbose=True, usesurrounding=(10,5)):
    """
    Fit the main beam on 2D OTF data
    input:
        ds: xarray dataset with L1 data
        nskyindexes: number of sky indexes to use for the fit (default: 15)
        plotstuff: plot the data and the fit (default: False)
        verbose: print the fit parameters (default: True)
    output:
        FWHPCCH1: FWHM of the Gaussian fit for CCH1
        FWHPCCH2: FWHM of the Gaussian fit for CCH2
        sigmaFWHMCCH1: sigma of the Gaussian fit for CCH1
        sigmaFWHMCCH2: sigma of the Gaussian fit for CCH2
    """

    # Determine theoretically, which should have a better SNR:
    if verbose: 
        determineSNR(ds)
    choice='CCH'

    # we use the entire dataset for the fit
    A2 = ds.ANGULAR.data/2
    deltas_file = os.path.join('pointing_offsets', f'deltas_{int(ds.OBSID.data[0])}.txt')
    if verbose: 
        print(deltas_file)
    if os.path.exists(deltas_file):
        # read the file as a pandas dataframe
        df = pd.read_csv(deltas_file, sep=',')
    else:
        raise ValueError(f'File {deltas_file} does not exist. Please run the script to create it.')
    
    TCA1 = np.sqrt((ds.AT.data-df.DeltaAT0CCH1[0]/60)**2 + (ds.CT.data-df.DeltaCT0CCH1[0]/60)**2)
    TCA2 = np.sqrt((ds.AT.data-df.DeltaAT0CCH2[0]/60)**2 + (ds.CT.data-df.DeltaCT0CCH2[0]/60)**2)
    # sort data by TCA
    sortidx1 = np.argsort(TCA1)
    sortidx2 = np.argsort(TCA2)
    TCA1 = TCA1[sortidx1]
    TCA2 = TCA2[sortidx2]
    A21 = A2[sortidx1]
    A22 = A2[sortidx2]
    theta1 = (TCA1 - A21)*60 # convert to arcmin
    theta2 = (TCA2 - A22)*60 # convert to arcmin
    CCH1 = ds.CCH1.data[sortidx1]
    CCH2 = ds.CCH2.data[sortidx2]

    plotparams = {'type': choice, 'channel': '1', 'LO': ds.LO1.data[0], 'title': 'SWA 2D Map', 'obsid': ds.OBSID.data[0]}
    FWHPCCH1, sigmaFWHPCCH1 = erfit_main_beam(CCH1, theta1, nskyindexes=nskyindexes, plotstuff=plotstuff, verbose=verbose, plotparams=plotparams, usesurrounding=usesurrounding[0])
    plotparams = {'type': choice, 'channel': '2', 'LO': ds.LO2.data[0], 'title': 'SWA 2D Map', 'obsid': ds.OBSID.data[0]}    
    FWHPCCH2, sigmaFWHPCCH2 = erfit_main_beam(CCH2, theta2, nskyindexes=nskyindexes, plotstuff=plotstuff, verbose=verbose, plotparams=plotparams, usesurrounding=usesurrounding[1])
    # print the results
    print("==========================")
    print("ObsID:", ds.OBSID.data[0])
    print("FWHM CCH1: {:.5f} ± {:.5f}".format(FWHPCCH1, sigmaFWHPCCH1))
    print("FWHM CCH2: {:.5f} ± {:.5f}".format(FWHPCCH2, sigmaFWHPCCH2))
    print("===========================")

    return FWHPCCH1, FWHPCCH2, sigmaFWHPCCH1, sigmaFWHPCCH2



def errorfunct(x, A, mu, sigma, offset = 0):
    """
    Error function for the Gaussian fit
    input:
        x: x axis values
        A: amplitude of the Gaussian
        mu: center of the Gaussian
        sigma: width of the Gaussian
        offset: this takes into account the calibration, and possibly sidelobe contribution not described by the gaussian main beam model
    output:
        y: y axis values
    """
    return 0.5*(1 + erf((mu-x)/(sigma*np.sqrt(2))))*A+offset

def erfit_main_beam(CCH, theta, nskyindexes=15, plotstuff=False, verbose=True, initial_width=8, usesurrounding=30, plotparams=None, returnallfitparams=False, normalise=True):
    """
    Fit the main beam using the error function.

    Parameters:
        CCH (numpy.ndarray): Normalized CCH data.
        theta (numpy.ndarray): Angular offset data in arcmin.
        nskyindexes (int): Number of sky indexes to use for estimating noise level (default: 15).
        plotstuff (bool): Whether to plot the data and the fit (default: False).
        verbose (bool): Whether to print
        usesurrounding: arcminutes around the center to be considered in the fit
        plotparams: parameters of the plot if the plotting is set to true
        returnallfitparams (bool): Whether to return the full fit parameters and their uncertainties (default: False).
    output:
        FWHPCCH: FWHM of the Gaussian fit
        sigmaFWHMCCH: sigma of the Gaussian fit
        (optional) fit: Full fit parameters
        (optional) perr: Uncertainties of the fit parameters
    """

    # we normalise the data 
    if normalise: CCH = CCH/np.max(CCH)
    # find the approximate center of the beam (CCH = 0.5)
    argidx = np.argmin(np.abs(CCH-0.5))
    theta0 = theta[argidx]
    # only use the surrounding points for the fit based on arcmin range
    crop_mask = (theta >= theta0 - usesurrounding) & (theta <= theta0 + usesurrounding)
    cropCCH = CCH[crop_mask]
    croptheta = theta[crop_mask]
    # determine the resolution in theta at the limb: theta range/sum of mask
    theta_range = 2 * usesurrounding  # arcmin^-1
    period = theta_range / np.sum(crop_mask)
    if verbose: print(f"Period in theta at the limb: {period:.2f} arcmin")


    # convert initial width to sigma
    sigma0 = initial_width/(2*np.sqrt(2*np.log(2)))
    # fit the data with error function
    initial_guess = [1, theta0, sigma0,0]
    nskyindexes = -nskyindexes
    fit, pcov = curve_fit(errorfunct, croptheta, cropCCH, sigma=np.std(CCH[nskyindexes:]), p0=initial_guess)
    # Extract fitted parameters, including uncertainties
    A, mu, sigma,offs = fit
    perr = np.sqrt(np.diag(pcov))
    sA, smu, ssigma, soffs = perr
    # convert sigma to FWHM
    FWHMCCH = 2*np.sqrt(2*np.log(2))*sigma
    sFWHMCCH = 2*np.sqrt(2*np.log(2))*ssigma
    # if plotstuff
    if plotstuff:
        #print(plotparams)
        if plotparams is None:
            plotparams = {'type': 'CCH', 'channel': '', 'LO': '', 'title': 'SWA', 'obsid': 0}
            #print('Plot parameters not provided')
        plt.plot(theta-mu, CCH, 'o', label=f'{plotparams["type"]}{plotparams["channel"]}')
        plt.plot(croptheta-mu, cropCCH, 'o', label='Used for fit')
        smooththeta = np.linspace(np.min(theta), np.max(theta), 2000)
        plt.plot(smooththeta-mu, errorfunct(smooththeta, A, mu, sigma, offs), label='Error function fit')
        plt.legend()
        plt.xlabel(r"$\theta$ ($\mathrm{'}$)")
        plt.ylabel(f'{plotparams["type"]}{plotparams["channel"]} normalised')
        plt.title(f'{plotparams["title"]}, ObsID: {int(plotparams["obsid"])}, LO{plotparams["channel"]} = {plotparams["LO"]:.2f} GHz')
        plt.xlim(-usesurrounding*3, usesurrounding*3)
        # Display FWHM ± sigma as text inside the figure
        plt.text(usesurrounding / 2, 0.55, f"FWHM = {FWHMCCH:.1f} ± {sFWHMCCH:.1f}", fontsize=10, color='black')
        # Save the figure
        plt.savefig(f"Figures/ErrorFunctionFit_{int(plotparams['obsid'])}_{plotparams['type']}{plotparams['channel']}.png", dpi=300, bbox_inches='tight')
        plt.show()

    if returnallfitparams:
        # Warning: Ensure that the `returnallfitparams` flag is handled correctly to avoid potential parsing errors.
        # If `returnallfitparams` is True, the function will return additional parameters (fit and perr).
        # Ensure that the calling code is prepared to handle the different return values based on this flag.
        return FWHMCCH, sFWHMCCH, fit, perr
    return FWHMCCH, sFWHMCCH
    

def full_beam_pattern(ds, nskyindexes=15, start = 0, end = -1, method="threepointstencil",choice="CCH", plotGauss = True, savefig=True):
    """
    This function differentiates the continuum channel data (or integrated CTS) with respect to angle. The data is plotted with uncertainties in dB scale. Fit of the beam pattern and sidelobes to be implemented.
    input:
        ds: xarray dataset with L1 data
        nskyindexes: number of sky indexes to use for the fit (default: 15)
        start: index to start the fit (default: 0)
        end: index to end the fit (default: -1)
        method: method to use for the differentiation (default: "threepointstencil")
        choice: "CCH" or "CTS" to select the data type to plot (default: "CCH")
        plotGauss: whether to plot the Gaussian fit (default: True)
        savefig: whether to save the figure (default: True)
    output:
        None - plots of the beam pattern and sidelobes
    """

    # to avoid runtime warnings
    import warnings
    warnings.filterwarnings("ignore")

    # Check if choice is correct and obtain the sT uncertainty (SNR)
    sT1, sT2, correctchoice = determineSNR(ds)
    #print(sT)
    #print(correctchoice)
    if choice == "CCH":
        CH1 = ds.CCH1.data[start:end]
        CH2 = ds.CCH2.data[start:end]
    elif choice == "CTS":
        CH1 = np.mean(ds.CTS1.data[start:end], axis=1)
        CH2 = np.mean(ds.CTS2.data[start:end], axis=1)
    else:
        raise ValueError("Invalid choice. Use 'CCH' or 'CTS'.")
    
    if correctchoice != choice:
        print(f"Warning: The SNR was determined for {correctchoice}, but the choice is {choice}. Adjusting choice to {correctchoice}.")
        choice = correctchoice
        print("mfw when 'the illusion of choice'")

    deltas_file = os.path.join('pointing_offsets', f'deltas_{int(ds.OBSID.data[0])}.txt')
    if os.path.exists(deltas_file):
        # read the file as a pandas dataframe
        df = pd.read_csv(deltas_file, sep=',')
    else:
        # create a dataframe with the same columns as the deltas file
        print("Offsets unavailable. This might or might not be an issue")
        df = pd.DataFrame(columns=['DeltaAT0CCH1', 'DeltaCT0CCH1', 'DeltaAT0CCH2', 'DeltaCT0CCH2'])
        # fill the dataframe with zeros
        df.loc[0] = [0, 0, 0, 0]
    
    TCA1 = np.sqrt((ds.AT.data-df.DeltaAT0CCH1[0]/60)**2 + (ds.CT.data-df.DeltaCT0CCH1[0]/60)**2)
    TCA2 = np.sqrt((ds.AT.data-df.DeltaAT0CCH2[0]/60)**2 + (ds.CT.data-df.DeltaCT0CCH2[0]/60)**2)
    TCA1 = TCA1[start:end]*60 # cut the data to the start index + convert to arcmin
    TCA2 = TCA2[start:end]*60 # cut the data to the start index + convert to arcmin
    A2 = ds.ANGULAR.data[start:end]/2*60
    
    # Calculate theta
    theta1 = (TCA1 - A2)
    theta2 = (TCA2 - A2)
    if ds.OBSID.data[0] == 228:
        TAC = ds.SWI_BORESIGHT_TAC.data[start:end]*60
        theta1 = TAC - A2
        theta2 = TAC - A2
        print("Using accurate TAC")
    
    # The sorting causes issues (variations across disk) - Do not uncomment unless you know what you're doing
    #CH1 = CH1[np.argsort(theta1)]
    #theta1 = theta1[np.argsort(theta1)]
    #CH2 = CH2[np.argsort(theta2)]
    #theta2 = theta2[np.argsort(theta2)]

    # Differentiate the data
    if method == "threepointstencil":
        dCH1dtheta, sdCH1dtheta = threepointstencil(theta1, CH1, sT1)
        dCH2dtheta, sdCH2dtheta = threepointstencil(theta2, CH2, sT2)
    elif method == "twopointstencil":
        raise NotImplementedError
    else:
        raise NotImplementedError
    
    # save the differentiated data to a file
    diff_data = {
        'theta1': theta1,
        'dCH1dtheta': dCH1dtheta,
        'sdCH1dtheta': sdCH1dtheta,
        'theta2': theta2,
        'dCH2dtheta': dCH2dtheta,
        'sdCH2dtheta': sdCH2dtheta
    }
    # save ascii
    diff_file = os.path.join('differentiated', f'diff_data_{int(ds.OBSID.data[0])}.txt')
    if not os.path.exists('differentiated'):
        os.makedirs('differentiated')
    np.savetxt(diff_file, 
               np.column_stack((theta1[1:-1], dCH1dtheta, sdCH1dtheta, theta2[1:-1], dCH2dtheta, sdCH2dtheta)), 
               header='theta1 dCH1dtheta sdCH1dtheta theta2 dCH2dtheta sdCH2dtheta', 
               fmt='%f')
    print(f"Saved differentiated data to {diff_file}")


    
    # commented for now
    #theta1 = theta1 - theta1[np.argmax(dCH1dtheta)]
    #theta2 = theta2 - theta2[np.argmax(dCH2dtheta)]
    
    # Fit of Channel 1
    # dimensionless parameters
    lamb=spc.c/(ds.LO1.data[0]*1e9)
    d=29/100 #/1.1 # 30cm dish
    u = np.sin(np.deg2rad(theta1[1:-1]/60))*d*np.pi/lamb

    # Fit the beam pattern model to the data
    initial_guess = [0.0, 2, 1.0]  # Initial guess for tau and beta
    bounds = ([-1000, 0, 0.8], [1000, np.inf, 1.2])  # Bounds for tau and beta
    y=-dCH1dtheta/np.max(-dCH1dtheta)
    ymask = (u > 0) | ((u < 0) & (10*np.log10(y) > -13))
    popt, pcov, id, msg, fi = curve_fit(beam_pattern_model_experimental, u[ymask], y[ymask], sigma=sdCH1dtheta[ymask], p0=initial_guess, bounds=bounds, full_output=True)
    #print(id)
    print(msg)
    #print(fi)
    center1, beta_fit1, uscale1 = popt
    center_err1, beta_err1, uscale_err1 = np.sqrt(np.diag(pcov))
    theta0 = np.rad2deg(np.arcsin(center1/d/np.pi*lamb))*60
    theta1 = theta1-theta0
    popt, pcov = curve_fit(Gauss, theta1[1:-1][ymask], y[ymask], sigma = sdCH1dtheta[ymask], p0=[8.,0.,3.])
    amp1, x01, sig1 = popt
    samp1, sx01, ssig1 = np.sqrt(np.diag(pcov))
    
    convertbetatophysical = lamb/(2*np.pi*(1-np.cos(np.deg2rad(45))))
    # Print the fitted parameters
    print("Channel 1 fit:")
    print(f"Fitted beta: {beta_fit1:.3f} ± {beta_err1:.3f}")
    delta1 = beta_fit1 * convertbetatophysical
    errdelta1 = beta_err1 * convertbetatophysical
    print(f"Axial defocus in physical units: {delta1*1e3:.3f} ± {errdelta1*1e3:.3f} mm")
    print(f"Fitted uscale: {uscale1:.3f} ± {uscale_err1:.3f}")
    print(f"Fitted Gaussian parameters for Channel 1:")
    print(f"Amplitude = {amp1:.3f} ± {samp1:.3f}")
    print(f"Center = {x01:.3f} ± {sx01:.3f}")
    print(f"Sigma = {sig1:.3f} ± {ssig1:.3f}")
    FWHM1 = 2*np.sqrt(2*np.log(2))*sig1
    sFWHM1 = 2*np.sqrt(2*np.log(2))*ssig1
    print(f"FWHM = {FWHM1:.3f} ± {sFWHM1:.3f}")

    # Plot the fitted model
    fmtheta1 = np.linspace(-30,30,1000)
    u = np.sin(np.deg2rad(fmtheta1/60))*d*np.pi/lamb
    fitted_model = beam_pattern_model_experimental(u, 0, beta_fit1, uscale1)
    fitted_model1 = fitted_model#/np.max(fitted_model)
    
    # Fit2
    lamb=spc.c/(ds.LO2.data[0]*1e9)
    u2 = np.sin(np.deg2rad(theta2[1:-1]/60))*d*np.pi/lamb
    # Fit the beam pattern model to the data
    initial_guess = [center1, 2]  # Initial guess for tau and beta
    # bounds same as before
    bounds = ([-1000, 0], [1000, np.inf]) 
    y=-dCH2dtheta/np.max(-dCH2dtheta)
    ymask = (u2 > 0) | ((u2 < 0) & (10*np.log10(y) > -10))
    # plot the masked data( simple log scale )
    #plt.plot(u2[ymask], 10 * np.log10(y[ymask]), 'o', label='Masked data')
    #plt.xlabel('u')
    #plt.ylabel('10 * log10(y)')
    #plt.title('Masked data in log scale')
    #plt.legend()
    #plt.show()

    popt, pcov, id, msg, fi = curve_fit(beam_pattern_model_experimental, u2[ymask], y[ymask], sigma=sdCH2dtheta[ymask], p0=initial_guess, bounds=bounds, full_output=True)
    #print(id)
    print(msg)
    #print(fi)
    #center2, beta_fit2, uscale2 = popt
    #center_err2, beta_err2, uscale_err2 = np.sqrt(np.diag(pcov))
    center2, beta_fit2 = popt
    center_err2, beta_err2 = np.sqrt(np.diag(pcov))
    theta0 = np.rad2deg(np.arcsin(center2/d/np.pi*lamb))*60
    theta2 = theta2-theta0
    popt, pcov = curve_fit(Gauss, theta2[1:-1][ymask], y[ymask], sigma = sdCH2dtheta[ymask], p0=[8.,0.,1.5])
    amp2, x02, sig2 = popt
    samp2, sx02, ssig2 = np.sqrt(np.diag(pcov))

    convertbetatophysical = lamb/(2*np.pi*(1-np.cos(np.deg2rad(45))))
    # Print the fitted parameters
    print("Channel 2 fit:")
    print(f"Fitted beta: {beta_fit2:.3f} ± {beta_err2:.3f}")
    delta1 = beta_fit1 * convertbetatophysical
    errdelta1 = beta_err1 * convertbetatophysical
    print(f"Axial defocus in physical units: {delta1*1e3:.3f} ± {errdelta1*1e3:.3f} mm") 
    #print(f"Fitted uscale: {uscale2:.3f} ± {uscale_err2:.3f}")
    print(f"Fitted Gaussian parameters for Channel 1:")
    print(f"Amplitude = {amp2:.3f} ± {samp2:.3f}")
    print(f"Center = {x02:.3f} ± {sx02:.3f}")
    print(f"Sigma = {sig2:.3f} ± {ssig2:.3f}")
    FWHM2 = 2*np.sqrt(2*np.log(2))*sig2
    sFWHM2 = 2*np.sqrt(2*np.log(2))*ssig2
    print(f"FWHM = {FWHM2:.3f} ± {sFWHM2:.3f}")

    # Plot the fitted model
    fmtheta2 = np.linspace(-30,30,1000)
    u2 = np.sin(np.deg2rad(fmtheta2/60))*d*np.pi/lamb
    fitted_model = beam_pattern_model_experimental(u2, 0, beta_fit2, uscale1)
    fitted_model2 = fitted_model#/np.max(fitted_model)
    
    # error plot in dB using errorplot_in_dB()
    fig, axs = plt.subplots(1,2,figsize=(12,4))
    errorplot_in_dB(axs[0], theta1[1:-1], -dCH1dtheta, sdCH1dtheta, label='Data')
    axs[0].plot(fmtheta1, 10 * np.log10(fitted_model1), label="Fitted sidelobe model", linestyle="--", marker = '', color="red")
    if plotGauss:
        axs[0].plot(fmtheta1, 10 * np.log10(Gauss(fmtheta1,amp1,x01,sig1)), label="Fitted Gaussian", linestyle=":", marker = '', color="green")
    axs[0].set_title(f"Beam pattern {choice}1 ObsID: {ds.OBSID.data[0]:.0f}")
    axs[0].set_xlabel(r"$\theta$ (')")
    axs[0].set_ylabel(r"$|\mathrm{d}T_{A}/\mathrm{d}\theta|$ (dB)")
    axs[0].set_ylim(-37,1.5)
    axs[0].set_xlim(-60,60)
    axs[0].grid(linestyle='--', alpha=0.5)
    # Add annotations to the leftmost and rightmost ticks without changing existing tickmarks
    axs[0].annotate("Target side", xy=(axs[0].get_xlim()[0], axs[0].get_ylim()[0]), 
                    xytext=(axs[0].get_xlim()[0] +100, axs[0].get_ylim()[0] + 5), 
                    textcoords="offset points", ha="center", va="bottom", fontsize=10, color="black")
    axs[0].annotate("Sky side", xy=(axs[0].get_xlim()[1], axs[0].get_ylim()[0]), 
                    xytext=(axs[0].get_xlim()[1] -100, axs[0].get_ylim()[0] + 5), 
                    textcoords="offset points", ha="center", va="bottom", fontsize=10, color="black")
    axs[0].legend(loc=2)
    # Add text box with fit parameters (beta and FWHM) in the top right corner
    if plotGauss:
        axs[0].text(0.95, 0.95, 
                f"   β = ({beta_fit1:.2f} ± {beta_err1:.2f}) \nFWHM = ({FWHM1:.2f} ± {sFWHM1:.2f})'", 
                transform=axs[0].transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', 
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
    else:
        axs[0].text(0.95, 0.95, 
                f"   β = ({beta_fit1:.2f} ± {beta_err1:.2f})", 
                transform=axs[0].transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', 
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

    

    
    errorplot_in_dB(axs[1], theta2[1:-1], -dCH2dtheta, sdCH2dtheta, label='Data')
    axs[1].plot(fmtheta2, 10 * np.log10(fitted_model2), label="Fitted sidelobe model", linestyle="--", marker = '', color="red")
    if plotGauss:
        axs[1].plot(fmtheta2, 10 * np.log10(Gauss(fmtheta2,amp2,x02,sig2)), label="Fitted Gaussian", linestyle=":", marker = '', color="green")
    axs[1].set_title(f"Beam pattern {choice}2 ObsID: {ds.OBSID.data[0]:.0f}")
    axs[1].set_xlabel(r"$\theta$ (')")
    axs[1].set_ylabel(r"$|\mathrm{d}T_{A}/\mathrm{d}\theta|$ (dB)")
    axs[1].set_ylim(-37,1.5)
    axs[1].set_xlim(-60,60)
    axs[1].grid(linestyle='--', alpha=0.5)
    # Add annotations to the leftmost and rightmost ticks without changing existing tickmarks
    axs[1].annotate("Target side", xy=(axs[0].get_xlim()[0], axs[0].get_ylim()[0]), 
                    xytext=(axs[0].get_xlim()[0] +100, axs[0].get_ylim()[0] + 5), 
                    textcoords="offset points", ha="center", va="bottom", fontsize=10, color="black")
    axs[1].annotate("Sky side", xy=(axs[0].get_xlim()[1], axs[0].get_ylim()[0]), 
                    xytext=(axs[0].get_xlim()[1] -100, axs[0].get_ylim()[0] + 5), 
                    textcoords="offset points", ha="center", va="bottom", fontsize=10, color="black")
    axs[1].legend(loc=2)
    if plotGauss:
        axs[1].text(0.95, 0.95, 
                f"   β = ({beta_fit2:.2f} ± {beta_err2:.2f}) \nFWHM = ({FWHM2:.2f} ± {sFWHM2:.2f})'", 
                transform=axs[1].transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', 
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
    else:
        axs[1].text(0.95, 0.95, 
                f"   β = ({beta_fit2:.2f} ± {beta_err2:.2f})", 
                transform=axs[1].transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', 
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))


    plt.tight_layout()
    obsid = int(ds.OBSID.data[0])  # Extract the ObsID
    
    if savefig:
        plt.savefig(f"Figures/BeamPattern_{choice}_ObsID_{obsid}_{start}_{end}.png", dpi=300, bbox_inches='tight')
    plt.show()

    # separate plot of channel 2 with 10x sdCH2dtheta (it is underestimated)
    #plt.figure(figsize=(6, 4))
    #errorplot_in_dB(plt.gca(), theta2[1:-1], -dCH2dtheta, 10 * sdCH2dtheta, label='CCH2 (5x error)')
    #plt.title(f"Beam pattern CCH2 ObsID: {ds.OBSID.data[0]:.0f} (10x error)")
    #plt.xlabel(r"$\theta$ (')")
    #plt.ylabel(r"$|\mathrm{d}T_{RJ}/\mathrm{d}\theta|$ (K/ ')")
    #plt.ylim(-25, 1.5)
    #plt.grid(linestyle='--', alpha=0.5)
    #plt.tight_layout()
    #plt.show()



def errorplot_in_dB(ax, x, y, sy, label=''):
    """
    This function converts the linear (power) y, and sy to dB and plots this out in the ax axis
    input:
        ax: matplotlib axis to plot in
        x: x values
        y: y values (linear)
        sy: y uncertainties (linear)
        label: label for the plot (default: '')
    output:
        None - plots the data in dB scale
    """
    sy=sy/np.max(y)
    y=y/np.max(y)
    # Convert to dB
    y_dB = 10 * np.log10(y)
    # unsymmetric errorbars for sy - log10(y+-sy)
    sposit = 10 * np.log10(y + sy)  # positive error
    snegat = 10 * np.log10(y - sy)  # negative error
    snegat = y_dB-snegat
    sposit = sposit-y_dB
    # set snegat to 100 where it is nan or +-inf
    snegat[np.isnan(snegat) | np.isinf(snegat)] = 100
    # Plot with error bars in dB
    ax.errorbar(x, y_dB, yerr=[snegat, sposit], fmt='o', label=label, capsize=3)

    # plot the negative values in gray
    ny_dB = 10 * np.log10(-y)
    nsposit = 10 * np.log10(-y + sy)
    nsnegat = 10 * np.log10(-y - sy)
    nsnegat = ny_dB-nsnegat
    nsposit = nsposit-ny_dB
    nsnegat[np.isnan(nsnegat) | np.isinf(nsnegat)] = 100
    ax.errorbar(x, ny_dB, yerr=[nsnegat, nsposit], fmt='o', label=label + " (negative)", capsize=3, alpha=0.5, color='gray')


def F(r, tau):
    """
    Compute the edge taper formula.
    input:
        r (float or np array): The radial distance or scaling factor.
        tau (float): The tapering parameter, controlling the strength of the taper.
    output:
        float: The computed edge taper value.
    """
    return 1-(1-tau)*r**2 # comments longer than the implementation lol, I am so commited to documentation

def beam_pattern_model(u, center, beta, edgetaper=17):
    """
    Following Baars, this function is the model containing the edge taper and the axial defocusing. We need to integrate numerically to obtain the value
    physical axial misalignment is less or equal to 900 um$
    """
    tau = 10**(-edgetaper/20) # edgetaper default: 17 dB edge taper
    nintsteps = 400
    r = np.linspace(0,1,nintsteps)
    rip1mri = np.diff(r)
    u=u-center
    Integral = np.zeros_like(u)
    for i in range(np.size(u)):
        f = F(r,tau)*jv(0,u[i]*r)*np.exp(-1j*beta*r**2)*r
        Integral[i] = np.sum(rip1mri*(f[:-1]+f[1:])/2)
    return np.abs(Integral)**2/np.max(np.abs(Integral)**2)

def Fr(r, tau):
    """
    Compute the edge taper formula.
    input:
        not sure, ask Baars
    output:
        float: The computed edge taper value.
    """
    return 1-tau*r**2 

def beam_pattern_model_experimental(u, center, beta,scaleu = 0.913,edgetaper=17):
    """
    Following Baars blindly
    """
    tau = 10**(-edgetaper/20) # edgetaper default: -12 dB edge taper
    nintsteps = 400
    r = np.linspace(0,1,nintsteps)
    rip1mri = np.diff(r)
    u=u-center
    u=u*scaleu
    Integral1 = np.zeros_like(u)
    Integral2 = np.zeros_like(u)
    for i in range(np.size(u)):
        f1 = 1.6*Fr(r,1-tau)*jv(0,u[i]*r)*np.sin(beta*r**2)*r
        f2 = 1.6*Fr(r,1-tau)*jv(0,u[i]*r)*np.cos(beta*r**2)*r
        Integral1[i] = np.sum(rip1mri*(f1[:-1]+f1[1:])/2)
        Integral2[i] = np.sum(rip1mri*(f2[:-1]+f2[1:])/2)
    Itot = 4*(Integral1**2+Integral2**2)
    return Itot/np.max(Itot)

def plot_continuum(ds, vs="index", choice="CCH", lim=150, dontshow=False):
    """
    Plot the continuum vs index or vs time UTC.

    Parameters:
        ds: xarray dataset with L1 data
        vs: "index" to plot against record index or "time" to plot against UTC time (default: "index", other options: "time", "AT", "CT")
        choice: "CCH" or "CTS" to select the data type to plot (default: "CCH")
        lim: Limit for AT or CT axis in arcmin (default: 150)
        dontshow: If True, return the axes instead of showing the plot (default: False)
    """
    if choice == "CCH":
        CH1 = ds.CCH1.data
        CH2 = ds.CCH2.data
    elif choice == "CTS":
        CH1 = np.mean(ds.CTS1.data, axis=1)
        CH2 = np.mean(ds.CTS2.data, axis=1)
    else:
        raise ValueError("Invalid choice. Use 'CCH' or 'CTS'.")
    
    CH1, CH2 = map(lambda x: x/np.max(x), [CH1, CH2])

    if vs == "index":
        x = np.arange(len(CH1))
        xlabel = "index"
    elif vs == "time":
        x = pd.to_datetime(ds.DATES.data)
        xlabel = "time (UTC)"
    elif vs == "AT":
        x = ds.AT.data * 60  # arcmin
        xlabel = r"AT ($\mathrm{'}$)"
        lims = [-lim, lim]
    elif vs == "CT":
        x = ds.CT.data * 60  # arcmin
        xlabel = r"CT ($\mathrm{'}$)"
        lims = [-lim, lim]
    else:
        raise ValueError("Invalid 'vs' parameter. Use 'index', 'time', 'AT', or 'CT'.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{choice} ObsID: {ds.OBSID.data[0]:.0f}")
    
    # Channel 1 subplot
    axes[0].plot(x, CH1, label=f"{choice}1", color="black", linestyle="", marker='.')
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(f"{choice}1 (normalised)")
    axes[0].set_title(f"LO1: {ds.LO1.data[0]:.2f} GHz")
    if vs in "ATCT":
        axes[0].set_xlim(lims[0], lims[1])
    if vs == "time":
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[0].legend(loc=8)
    axes[0].grid()
    
    # Channel 2 subplot
    axes[1].plot(x, CH2, label=f"{choice}2", color="black", linestyle="", marker='.')
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(f"{choice}2 (normalised)")
    axes[1].set_title(f"LO2: {ds.LO2.data[0]:.2f} GHz")
    if vs in "ATCT":
        axes[1].set_xlim(lims[0], lims[1])
    if vs == "time":
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[1].legend(loc=8)
    axes[1].grid()
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if dontshow:
        return axes
    else:
        plt.show()

def plot_averaged_2D_spectra(ds, lines={}, offset=0.5, verbose=False, noplot=False, ylim1 = [None, None], ylim2 = [None, None]):
    """
    Plot the averaged 2D spectra
    input:
        ds: xarray dataset with L1 data
        offset: how far beyond the angular radius of the target to look (default: 0.5) unit = beamwidth
    output:
        None - plots the averaged 2D spectra
    Used beamwidth
    FWHM CCH1: 9.0 ' (approximate value)
    FWHM CCH2: 4.5 ' (approximate value)
    """
    
    # For 2D maps we recenter
    A2 = ds.ANGULAR.data/2
    deltas_file = os.path.join('pointing_offsets', f'deltas_{int(ds.OBSID.data[0])}.txt')
    if verbose: 
        print(deltas_file)
    if os.path.exists(deltas_file):
        # read the file as a pandas dataframe
        df = pd.read_csv(deltas_file, sep=',')
        TCA1 = np.sqrt((ds.AT.data-df.DeltaAT0CCH1[0]/60)**2 + (ds.CT.data-df.DeltaCT0CCH1[0]/60)**2)
        TCA2 = np.sqrt((ds.AT.data-df.DeltaAT0CCH2[0]/60)**2 + (ds.CT.data-df.DeltaCT0CCH2[0]/60)**2)
    else:
        TCA1 = np.sqrt((ds.AT.data)**2 + (ds.CT.data)**2)
        TCA2 = np.sqrt((ds.AT.data)**2 + (ds.CT.data)**2)
        #print( Warning(f"File {deltas_file} does not exist. Please run the script to create it."))
        
    
    sortidx1 = np.argsort(TCA1)
    sortidx2 = np.argsort(TCA2)
    theta1 = (TCA1 - A2)*60 # convert to arcmin
    theta2 = (TCA2 - A2)*60 # convert to arcmin
    theta1 = theta1[sortidx1]
    theta2 = theta2[sortidx2]
    CTS1 = ds.CTS1.data[sortidx1,:]
    CTS2 = ds.CTS2.data[sortidx2,:]
    width1 = 9
    width2 = 4.5
    mask1 = theta1 < width1*offset
    mask2 = theta2 < width2*offset
    # limb only
    #mask1 = np.logical_and(theta1 < width1/2, -theta1 < width1/2)
    #mask2 = np.logical_and(theta2 < width2/2, -theta2 < width2/2)
    avgCTS1 = np.mean(CTS1[mask1], axis=0)
    avgCTS2 = np.mean(CTS2[mask2], axis=0)
    lsb1 = ds.LSB_FREQ_CTS1.data
    usb1 = ds.USB_FREQ_CTS1.data
    lsb2 = ds.LSB_FREQ_CTS2.data
    usb2 = ds.USB_FREQ_CTS2.data
    usb2[usb2<0] = np.nan
    
    #print(mask1)
    #print(mask2)

    if not noplot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(f'Averaged {ds.TARGET.data[0]} spectrum, ObsID: {int(ds.OBSID.data[0])}')
        ax1.set_title("LO1: %.1f GHz" % ds.LO1.data[0])
        ax1.plot(lsb1[10:-10], avgCTS1[10:-10],label="CTS1",color='black')
        #plt.title("Channel 1")
        #ax1.set_ylim(140,174)
        ax1.set_xlabel("LSB Frequency (GHz)")
        ax1.set_ylabel("$T_{RJ}$ (K)")
        ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax1b = ax1.twiny()
        ax1b.set_xlabel("USB Frequency (GHz)")
        ax1b.plot(usb1[10:-10], avgCTS1[10:-10],alpha=0)
        ax1b.xaxis.set_inverted(True)
        ax1b.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))


        ax2.plot(lsb2[10:-10], avgCTS2[10:-10], label='CTS2',color='black')
        ax2.set_title("LO2: %.1f GHz" % ds.LO2.data[0])
        #ax2.set_ylim(150,205)
        ax2.set_xlabel("LSB Frequency (GHz)")
        ax2.set_ylabel("$T_{RJ}$ (K)")
        ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax2b = ax2.twiny()
        ax2b.set_xlabel("USB Frequency (GHz)")
        ax2b.plot(usb2[10:-10], avgCTS2[10:-10],alpha=0)
        ax2b.xaxis.set_inverted(True)
        ax2b.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
       
        if ylim1[0]:
            ax1b.set_ylim(ylim1[0], ylim1[1])
        if ylim2[0]:
            ax2b.set_ylim(ylim2[0], ylim2[1])

        ax1.spines['top'].set_color('red')
        ax1b.spines['top'].set_color('red')
        ax2.spines['top'].set_color('red')
        ax2b.spines['top'].set_color('red')

        ax1.spines['bottom'].set_color('blue')
        ax1b.spines['bottom'].set_color('blue')
        ax2.spines['bottom'].set_color('blue')
        ax2b.spines['bottom'].set_color('blue')

        # Parse the input lines
        for line, freq in lines.items():
            # Get the molecule name from the line
            molecule = line.split('|')[0]
            dopplerfreq = freq-np.mean(ds.VEL_RADIAL.data)/c_kmps*freq
            print(np.mean(ds.VEL_RADIAL.data)/c_kmps*freq)
            freq=dopplerfreq
            
            # Determine the channel based on the frequency range
            if lsb1.min() <= freq <= usb1.max():
                channel = 'CH1'
                sideband = 'USB' if freq > lsb1.max() else 'LSB'
            elif lsb2.min() <= freq <= usb2.max():
                channel = 'CH2'
                sideband = 'USB' if freq > lsb2.max() else 'LSB'
            else:
                continue

            
            
            if sideband == 'USB':
                if channel == 'CH1':
                    ax1b.axvline(freq, color='r', linestyle='--', label=molecule)
                else:
                    ax2b.axvline(freq, color='r', linestyle='--', label=molecule)
            else:
                if channel == 'CH1':
                    ax1.axvline(freq, color='b', linestyle='--', label=molecule)
                else:
                    ax2.axvline(freq, color='b', linestyle='--', label=molecule)
                
        ax1.legend(loc=1)
        ax1b.legend(loc=2)
        ax2.legend(loc=1)
        ax2b.legend(loc=2)

            


        plt.savefig(f"Figures/Averaged2DSpectra_{int(ds.OBSID.data[0])}.png", dpi=300, bbox_inches='tight')
        plt.show()

        # ds = st.load(375)
        # st.plot_averaged_2D_spectra(ds,{"H2O": 1207.63873})
        # ds = st.load(377)
        # st.plot_averaged_2D_spectra(ds,{"HF": (1232.47627),"O3": (1220.474835)})


def plot_line_mcontinuum_2D(ds, linename, linefreq, linewidth, ch, sb, pole=None, skipdebugplots=False, limitextent = [-np.inf, np.inf], ax=None):
    """
    Plot the 2D map of the line minus continuum for a given dataset.

    Parameters:
        ds (xarray.Dataset): The dataset containing the spectral data.
        linename (str): The name of the spectral line to be plotted.
        linefreq (float): The frequency of the spectral line in GHz.
        linewidth (float): The width of the spectral line in GHz.
        ch (int): The channel index to be used for the analysis (1 or 2).
        sb (str): The sideband ('LSB' or 'USB') to be used for the analysis.
        pole (float, optional): The pole angle in degrees for additional plotting (default: None).
        skipdebugplots (bool, optional): Whether to skip generating debug plots (default: False).
        limitextent (list, optional): The limit extent of the frequency range to be considered for the analysis. 
            It should be a list with two elements, [lower_limit, upper_limit]. 
            The default value is [-np.inf, np.inf], which means the entire frequency range is considered.
        ax (matplotlib.axes.Axes, optional): The axes to plot on. If None, a new figure and axes will be created (default: None).
    Returns:
        None: This function generates a plot and does not return any value.
    """

    # Doppler shift of the line frequency
    linefreq = linefreq-ds.VEL_RADIAL.data[0]/c_kmps*linefreq

    # set default matplotlib fontsize to 14
    plt.rcParams.update({'font.size': 14})

    # Implementation: start by applying ATCT offset
    A2 = ds.ANGULAR.data / 2
    deltas_file = os.path.join('pointing_offsets', f'deltas_{int(ds.OBSID.data[0])}.txt')
    if os.path.exists(deltas_file):
        # Read the file as a pandas dataframe
        df = pd.read_csv(deltas_file, sep=',')
    else:
        raise ValueError(f"File {deltas_file} does not exist. Please run the script to create it.")

    TCA1 = np.sqrt((ds.AT.data - df.DeltaAT0CCH1[0] / 60) ** 2 + (ds.CT.data - df.DeltaCT0CCH1[0] / 60) ** 2)
    TCA2 = np.sqrt((ds.AT.data - df.DeltaAT0CCH2[0] / 60) ** 2 + (ds.CT.data - df.DeltaCT0CCH2[0] / 60) ** 2)
    if ch==1:
        TCA = TCA1
        freq = ds.LSB_FREQ_CTS1.data if sb == "LSB" else ds.USB_FREQ_CTS1.data
        specs = ds.CTS1.data
    else:
        TCA = TCA2
        freq = ds.LSB_FREQ_CTS2.data if sb == "LSB" else ds.USB_FREQ_CTS2.data
        specs = ds.CTS2.data

    # create a mask for the line and for outside the line
    line_mask = (freq >= (linefreq - linewidth / 2)) & (freq <= (linefreq + linewidth / 2))
    outside_mask = ~line_mask & (freq >= limitextent[0]) & (freq <= limitextent[1])

    # Make a folder within Figures to store debug plots
    debug_folder = f"Figures/debug_plots_obsid_{int(ds.OBSID.data[0])}"
    os.makedirs(debug_folder, exist_ok=True)

    # for each record in CTS apply the mask: outside, fit with a linear function. inside, subtract the linear function and take average
    line_minus_continuum = []
    for idx, spec in enumerate(specs):
        # Fit a linear function to the outside mask
        outside_freq = freq[outside_mask]
        outside_spec = spec[outside_mask]
        coeffs = np.polyfit(outside_freq, outside_spec, 1)
        linear_fit = np.polyval(coeffs, freq)

        # Subtract the linear function from the spectrum
        corrected_spec = spec - linear_fit

        # Take the average of the line region
        line_avg = np.mean(corrected_spec[line_mask])
        line_minus_continuum.append(line_avg)

        # Plot the spectrum for the current pixel
        if skipdebugplots==False:
            plt.figure(figsize=(10, 6))
            plt.plot(freq, spec, label='Original spectrum', color='blue')
            plt.plot(freq, linear_fit, label='Linear fit (continuum)', color='red', linestyle='--')
            plt.plot(freq, corrected_spec, label='Corrected spectrum', color='green')
            plt.axvspan(linefreq - linewidth / 2, linefreq + linewidth / 2, color='yellow', alpha=0.3, label='Line Region')
            plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            plt.xlabel('Frequency (GHz)')
            plt.ylabel(r'$T_{RJ}$ (K)')
            plt.title(f'Spectrum for record {idx + 1} | AT: {ds.AT.data[idx]:.2f}, CT: {ds.CT.data[idx]:.2f} | Line: {linename}',fontsize=16)
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(debug_folder, f'spectrum_record_{idx + 1}_{linename}.jpg'))
            plt.close()

    line_minus_continuum = np.array(line_minus_continuum)

    # plot the 2D map of line minus continuum

    # Create a scatter plot for the line_minus_continuum data
    AT = ds.AT.data
    CT = ds.CT.data

    if ax:
        savefig=True
    else:
        savefig=False
        fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(AT, CT, c=line_minus_continuum, cmap='viridis', s=200)
    plt.colorbar(scatter, ax=ax,label=f'{linename} Line - Continuum (K)')
    ax.set_xlabel('AT  (arcmin)')
    ax.set_ylabel('CT  (arcmin)')
    
    ax.set_title(f'2D Plot of {linename} Line - Continuum', fontsize=16)


    if pole:
        r=A2[0]
        pole = 90-pole
    else: 
        r=A2[0]
        pole = 90-np.mean(ds.POLAR_ANG.data) 
    
    ax.plot([np.cos(np.deg2rad(pole))*r,-np.cos(np.deg2rad(pole))*r],[np.sin(np.deg2rad(pole))*r,-np.sin(np.deg2rad(pole))*r],color='black',linestyle='--')
    ax.add_patch(Circle((0,0), A2[0], edgecolor='red', facecolor='none', lw=1.5))
    ax.set_aspect(1)
    if savefig:
        plt.savefig(f"Figures/2DLineContinuum_{int(ds.OBSID.data[0])}_{linename}.jpg", dpi=300, bbox_inches='tight')
        plt.show()

    # st.plot_line_mcontinuum_2D(ds, "H2O", 1207.63873, 0.2, 2, "LSB")

def LineAbsorption(x,offset,A, mu, sigma):    
    """
    Gaussian function for line absorption with an additional offset (compensating what is likely an instrument instablility still evolving).
    input:
        x: frequency values
        offset: offset value
        A: amplitude of the Gaussian
        mu: mean (center frequency) of the Gaussian
        sigma: standard deviation (width) of the Gaussian
    output:
        Gaussian function evaluated at x + offset
    """
    return offset+Gauss(x, A, mu, sigma)

def plot_doppler_shift_2D(ds, linename, linefreq, linewidth, ch, sb, pole=None, skipdebugplots=False):
    """
    Fits a Gaussian to the spectral line for each pixel in the dataset and calculates 
    the Doppler shift in km/s by comparing the fitted line position with the given 
    line frequency.
    input:
        ds: xarray dataset with L1 data
        linename: name of the spectral line to be analyzed
        linefreq: rest frequency of the spectral line in GHz
        linewidth: expected width of the spectral line in GHz
        ch: channel index to be used for the analysis (1 or 2)
        sb: sideband ('LSB' or 'USB') to be used for the analysis
        pole: pole angle in degrees for additional plotting (default: None)
        skipdebugplots: whether to skip generating debug plots (default: False)
    output:
        None - generates a 2D plot of the Doppler shift
    """

    
    # Doppler shift of the line frequency
    linefreq = linefreq - ds.VEL_RADIAL.data[0] / c_kmps * linefreq

    # Implementation: start by applying ATCT offset
    A2 = ds.ANGULAR.data / 2
    deltas_file = os.path.join('pointing_offsets', f'deltas_{int(ds.OBSID.data[0])}.txt')
    if os.path.exists(deltas_file):
        # Read the file as a pandas dataframe
        df = pd.read_csv(deltas_file, sep=',')
    else:
        raise ValueError(f"File {deltas_file} does not exist. Please run the script to create it.")

    TCA1 = np.sqrt((ds.AT.data - df.DeltaAT0CCH1[0] / 60) ** 2 + (ds.CT.data - df.DeltaCT0CCH1[0] / 60) ** 2)
    TCA2 = np.sqrt((ds.AT.data - df.DeltaAT0CCH2[0] / 60) ** 2 + (ds.CT.data - df.DeltaCT0CCH2[0] / 60) ** 2)
    if ch == 1:
        TCA = TCA1
        freq = ds.LSB_FREQ_CTS1.data if sb == "LSB" else ds.USB_FREQ_CTS1.data
        specs = ds.CTS1.data
    else:
        TCA = TCA2
        freq = ds.LSB_FREQ_CTS2.data if sb == "LSB" else ds.USB_FREQ_CTS2.data
        specs = ds.CTS2.data

    # Create a mask for the line region
    line_mask = (freq >= (linefreq - linewidth / 2)) & (freq <= (linefreq + linewidth / 2))

    # Make a folder within Figures to store debug plots
    debug_folder = f"Figures/debug_plots_obsid_{int(ds.OBSID.data[0])}"
    os.makedirs(debug_folder, exist_ok=True)

    doppler_shifts = []
    sigma_doppler_shifts = []
    for idx, spec in enumerate(specs):
        # Fit a Gaussian to the spectral line
        line_freqs = freq[line_mask]
        line_spec = spec[line_mask]

        # Initial guess for Gaussian parameters: amplitude, offset, mean, stddev
        initial_guess = [np.max(line_spec), -np.max(line_spec) + np.min(line_spec), linefreq, linewidth / 20]
        bounds = ([0, -np.inf, linefreq - linewidth / 2, 0], [np.inf, np.inf, linefreq + linewidth / 2, np.inf])

        try:
            popt, pcov = curve_fit(LineAbsorption, line_freqs, line_spec, p0=initial_guess, bounds=bounds)
            fitted_center = popt[2]  # Extract the fitted line center
            doppler_shift = (fitted_center - linefreq) / linefreq * c_kmps  # Convert to km/s
            # Extract the uncertainty of the fitted line center
            perr = np.sqrt(np.diag(pcov))  # Standard deviation errors on the parameters
            fitted_center_uncertainty = perr[2]  # Uncertainty of the fitted line center
            sigma_doppler_shift = (fitted_center_uncertainty / linefreq) * c_kmps  # Convert to km/s
            # Plot the fit result for debugging
            if skipdebugplots==False:
                plt.figure(figsize=(10, 6))
                plt.plot(line_freqs, line_spec, 'o', label='Data', color='blue')
                plt.plot(line_freqs, LineAbsorption(line_freqs, *popt), label='Fit', color='red')
                plt.axvline(linefreq, color='green', linestyle='--', label='Line + mean relative motion')
                plt.axvline(fitted_center, color='orange', linestyle='--', label='Fitted center')
                plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
                plt.xlabel('Frequency (GHz)')
                plt.ylabel(r'$T_{RJ}$ (K)')
                plt.title(f'Fit for record {idx + 1} | AT: {ds.AT.data[idx]:.2f}, CT: {ds.CT.data[idx]:.2f} | Doppler Shift: {doppler_shift:.2f} ± {sigma_doppler_shift:.2f} km/s', fontsize=16)
                plt.legend()
                plt.grid()
                plt.savefig(os.path.join(debug_folder, f'absorption_fit_rec_{idx + 1}.jpg'))
                plt.close()

        except RuntimeError:
            doppler_shift = np.nan  # If the fit fails, assign NaN
            sigma_doppler_shift = np.inf
            
        doppler_shifts.append(doppler_shift)
        sigma_doppler_shifts.append(sigma_doppler_shift)
        

    doppler_shifts = np.array(doppler_shifts)
    sigma_doppler_shifts = np.array(sigma_doppler_shifts)
    # if the point is outside A/2, assign NaN
    doppler_shifts[TCA > A2[0]] = np.nan
    sigma_doppler_shifts[TCA > A2[0]] = np.nan

    # Calculate the mean uncertainty of the values within A/2
    mean_uncertainty = np.nanmean(sigma_doppler_shifts)
    print(f"Mean uncertainty of Doppler shifts within A/2: {mean_uncertainty:.2f} km/s")

    # Plot the 2D map of Doppler shifts

    AT = ds.AT.data*60 # arcmin
    CT = ds.CT.data*60 # arcmin

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(AT, CT, c=doppler_shifts*1000, cmap='coolwarm', s=200, vmin=-400, vmax=400)
    plt.colorbar(scatter, label=f'{linename} Velocity (m/s)')
    plt.xlabel('AT (arcmin)')
    plt.ylabel('CT (arcmin)')
    plt.title(f'Doppler shift 2D map, {linename}', fontsize=16)
    plt.gca().add_patch(Circle((0, 0), A2[0]*60, edgecolor='red', facecolor='none', lw=1.5))
    r=A2[0]*60

    if pole:
        pole = 90-pole
    else: 
        pole = 90-np.mean(ds.POLAR_ANG.data) 
    
    plt.plot([np.cos(np.deg2rad(pole))*r,-np.cos(np.deg2rad(pole))*r],
    [np.sin(np.deg2rad(pole))*r,-np.sin(np.deg2rad(pole))*r],
    color='black',linestyle='--')

    plt.gca().set_aspect(1)
    plt.savefig(f"Figures/2DDopplerShift_{int(ds.OBSID.data[0])}.jpg", dpi=300, bbox_inches='tight')
    plt.show()

    # Plot the uncertainty of the doppler shift (same plot but showing the uncertainty)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(AT, CT, c=sigma_doppler_shifts*1000, cmap='viridis', s=200)
    plt.colorbar(scatter, label=f'{linename} Velocity Uncertainty (m/s)')
    plt.xlabel('AT (arcmin)')
    plt.ylabel('CT (arcmin)')
    plt.title(f'Doppler shift uncertainty 2D map, {linename}', fontsize=16)
    plt.gca().add_patch(Circle((0, 0), A2[0]*60, edgecolor='red', facecolor='none', lw=1.5))
    if pole:
        plt.plot([np.cos(np.deg2rad(pole)) * r, -np.cos(np.deg2rad(pole)) * r],
                 [np.sin(np.deg2rad(pole)) * r, -np.sin(np.deg2rad(pole)) * r], color='black', linestyle='--')
    plt.gca().set_aspect(1)
    plt.savefig(f"Figures/2DDopplerShiftUncertainty_{int(ds.OBSID.data[0])}.jpg", dpi=300, bbox_inches='tight')
    plt.show()

    polar_angle_full = ds.POLAR_ANG.data
    # print("used polar angle: ",polar_angle_full)
    # rotate by minus the polar angle
    x = np.cos(np.deg2rad(-polar_angle_full))*AT + np.sin(np.deg2rad(-polar_angle_full))*CT
    y = -np.sin(np.deg2rad(-polar_angle_full))*AT + np.cos(np.deg2rad(-polar_angle_full))*CT
    # save ascii x, y, doppler
    doppler_data = np.column_stack((x, y, doppler_shifts, sigma_doppler_shifts))
    np.savetxt(f"Figures/DopplerShift_{int(ds.OBSID.data[0])}_{linename}.txt", doppler_data, header="x y doppler_shift sigma_doppler_shift", fmt='%.6f')

    # importlib.reload(st)
    # ds = st.load(375)
    # st.plot_doppler_shift_2D(ds, "H2O", 1207.63873, 0.05, 2, "LSB")



def get_forward_efficiency(ds,doppler=1., plotstuff=True, index = 112 ):
    """
    This function determines the model spectrum for a given tuning. We start by extracting the tuning from ds and then resample the model to have a direct comparison.
    input:
        ds - xarray with swi data
        doppler - doppler factor to apply to the model (default: 1.0) ??? - Keep this at 1, this is the correct value 
        but when there is quickly changing geometry this might be inaccurate... thus the "temporary" fix to adjust if really needed
        plotstuff - if True, plot the results (default: True)
        index - index of the record to plot (default: 112)
    output:
        model1 - model T for ch1
        model2 - model T for ch2
    """
    # Load model
    model1 = np.loadtxt("Model/earth-nadir-0600-chris.dat")
    model2 = np.loadtxt("Model/earth-nadir-1200-chris.dat")
    
    # Interpolate channel 1
    lf1 = ds.LSB_FREQ_CTS1
    uf1 = ds.USB_FREQ_CTS1
    model1freq = (model1[:, 0] / 1e9) + ds.VEL_RADIAL.data[0]/c_kmps*(model1[:, 0] / 1e9)*doppler
    limodel1 = np.interp(lf1, model1freq, model1[:, 1])
    uimodel1 = np.interp(uf1, model1freq, model1[:, 1])
    imodel1 = uimodel1 / 2 + limodel1[::-1] / 2

    # Interpolate channel 2
    lf2 = ds.LSB_FREQ_CTS2
    uf2 = ds.USB_FREQ_CTS2
    model2freq = (model2[:, 0] / 1e9) + ds.VEL_RADIAL.data[0]/c_kmps*(model2[:, 0] / 1e9)*doppler
    limodel2 = np.interp(lf2, model2freq, model2[:, 1])
    uimodel2 = np.interp(uf2, model2freq, model2[:, 1])
    imodel2 = uimodel2 / 2 + limodel2[::-1] / 2
    
    if plotstuff:
         # Defined the index
        obsid = int(ds.OBSID.data[0])  # Extract the ObsID

        # Plot for Channel 2
        plt.figure()
        plt.plot(uf2, ds.CTS2[index, :], label="Observed CTS2")
        plt.plot(uf2, imodel2, label="Model CTS2")
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("$T_{RJ}$ (K)")
        plt.title(f"Channel 2 - ObsID: {obsid}, Index: {index}")
        plt.legend()
        plt.savefig(f"Figures/ForwardEfficiency_Channel2_ObsID_{obsid}_Index_{index}.png", dpi=300, bbox_inches='tight')
        plt.show()

        # Plot for Channel 1
        plt.figure()
        plt.plot(uf1, ds.CTS1[index, :], label="Observed CTS1")
        plt.plot(uf1, imodel1, label="Model CTS1")
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("$T_{RJ}$ (K)")
        plt.title(f"Channel 1 - ObsID: {obsid}, Index: {index}")
        plt.legend()
        plt.savefig(f"Figures/ForwardEfficiency_Channel1_ObsID_{obsid}_Index_{index}.png", dpi=300, bbox_inches='tight')
        plt.show()

        # Plot the ratios of Forward efficiency for the first channel
        ratio1 = ds.CTS1[index, :] / imodel1
        mean_ratio1 = np.mean(ratio1)
        plt.figure()
        plt.plot(uf1, ratio1, label="Channel 1")
        plt.axhline(mean_ratio1, color='r', linestyle='--', label=f"Mean: {mean_ratio1:.2f}")
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Forward efficiency")
        plt.title(f"Forward Efficiency - Channel 1 - ObsID: {obsid}, Index: {index}")
        plt.legend()
        plt.savefig(f"Figures/ForwardEfficiency_Ratio_Channel1_ObsID_{obsid}_Index_{index}.png", dpi=300, bbox_inches='tight')
        plt.show()

        # Plot the ratios of Forward efficiency for the second channel
        ratio2 = ds.CTS2[index, :] / imodel2
        mean_ratio2 = np.mean(ratio2)
        plt.figure()
        plt.plot(uf2, ratio2, label="Channel 2")
        plt.axhline(mean_ratio2, color='r', linestyle='--', label=f"Mean: {mean_ratio2:.2f}")
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Forward efficiency")
        plt.title(f"Forward Efficiency - Channel 2 - ObsID: {obsid}, Index: {index}")
        plt.legend()
        plt.savefig(f"Figures/ForwardEfficiency_Ratio_Channel2_ObsID_{obsid}_Index_{index}.png", dpi=300, bbox_inches='tight')
        plt.show()

    
