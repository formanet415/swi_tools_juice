import xarray as xr
import numpy as np
import os

# Set the directory path below (line ~61)

def load(obsid,dir):
    """
    Loads L1 data for a given obsID
    input:
        obsid: obsID to load
        dir: directory where the L1 files are stored
    output:
        ds: xarray dataset with L1 data
    """
    # look for the L1 data file in <dir>/L1_<obsid>*.nc
    # load the file with xarray
    fnames = os.listdir(dir)
    fnames = [f for f in fnames if f.startswith('L1_') and f.endswith('.nc')]
    fnames = [f for f in fnames if str(obsid) in f]
    if len(fnames) == 0:
        raise ValueError(f'No L1 data found for obsID {obsid}')
    elif len(fnames) > 1:
        raise ValueError(f'Multiple L1 data files found for obsID {obsid}: {fnames}')
    fname = fnames[0]
    ds = xr.open_dataset(os.path.join(dir, fname))
    return ds

def get_SWI_BORESIGHT_TCA(ds,R=None):
    """
    Get the SWI boresight Target Center Angle (TCA)
    input:
        ds: xarray dataset with L1 data
        R: radius of the target body (default: 1737.4 km for the Moon)
    output:
        tca: boresight target center angle
        A/2: boresight angular radius
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


import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
if __name__ == "__main__":
    dir='DR-SWI/database-L1/'   # <--------- Replace with your directory
    obsid=228                   # <--------- Replace with your obsid
    ds=load(obsid,dir)

    tca, A2 = get_SWI_BORESIGHT_TCA(ds) # Our formula
    
    atct = np.sqrt(ds.AT.data**2+ds.CT.data**2)

    time = [datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%f') for x in ds.DATES.data.tolist() ]
    myFmt = mdates.DateFormatter('%H:%M')

    plt.figure()
    plt.plot(time,tca)
    plt.plot(time,A2)
    plt.xlabel('Time GMT')
    plt.ylabel(r'Angle ($^\circ$)')
    
    plt.plot(time,atct,linestyle='--')
    plt.legend(['TCA','Angular/2',r'$\sqrt{AT^2+CT^2}$'],loc='upper left')
    plt.title('L1 SWI obsid %d LEGA (outbound Moon)'%obsid)
    plt.twinx()
    plt.plot(time,ds.CCH1.data,linestyle='--',color='purple',alpha=0.5)
    plt.ylabel('CCH1 (K)')
    plt.legend(['CCH1'],loc='upper right')
    plt.savefig('Figures/TCA_%d.png'%obsid)
    plt.gca().xaxis.set_major_formatter(myFmt)
    plt.show()