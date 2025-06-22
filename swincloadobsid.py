# Python library for loading L01B SWI data 
# The goal of this library is to load raw SWI data and perform calibration to physical units.

# Unless the goal of your work is to refine the data calibration, the use of this library is discouraged, 
# use the switools library instead which works for L1 data.

# Version log:
# V0.1      21.2.2025   Basic netcdf loader
# V0.1.1    28.2.2025   Basic calibrated observations
# V0.2      3.3.2025    Module_v0.txt implementation - module 2
# V0.2.1    6.3.2025    Module_v0.txt implementation - modules 3, 4 and unfinished 5, minor bug fixes
# V0.2.2    8.3.2025    Module 5 implementation, bug fixes, datadir moved to a separate file
# V0.2.3    10.3.2025   Module 5 improvements, bug fixes
# V0.3      June 2025   Several bugfixes have been implemented and figure routines have been improved.


# Recommended use:
# import swincloadobsid as sd
# mydata = sd.load(obsid)

# Do not forget to adjust the data directory for the loader

# Imports
import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates

# we follow the structure from module_v0.txt

def getindexes(sidx):
    sidx = sidx.split(',')
    idx = []
    for sid in sidx:
        if '-' in sid:
            lims = [int(x) for x in sid.split('-')]
            for i in range(lims[0], lims[1]):
                idx.append(i)
        else:
            idx.append(int(sid))
    return idx


# Since we need to perform calibration later on, we load the data into a SWIData object.
class SWIDataLoader:
    def __init__(self, obsid):
        # Do not forget to adjust the data directory for the loader!
        # load directory path from a config file
        with open('datapath.txt', 'r') as f:
            self.ddir = f.readline()
        print("Data directory set to", self.ddir)
        # self.ddir = "/home/formanek/Documents/SWI_internship/DR-SWI/database/" # todo: remove this line
        if type(obsid)==int:
            self.obsid = obsid
            self.load(obsid)
        else:
            raise Exception("Incorrect format of ObsID: "+str(obsid)+"\n ObsID should be an integer.")
    
    def load(self, obsid):
        for file in os.listdir(self.ddir):
            try:
                cobsid = int(file[0:5])
            except:
                continue
            if cobsid == obsid:
                fname = file
                break
        else:
            print('Obsid', obsid, 'not found')
            return -1

        self.data = xr.open_dataset(self.ddir+fname)
        print("Loaded file located at "+self.ddir+fname)
        self.fname = fname
        
        # Observatonal table (from telemetry information)
        self.get_obs_table()
        self.nrecs = len(self.data.GMT.data)
        self.fname = fname
        self.cts_counts_to_continuum() # we need this to get the estimate of best sky indexes - needed by most modules
        self.time = datetime.strptime(self.data.GMT.data[0], '%Y-%m-%dT%H:%M:%S.%f')

    def get_obs_table(self):
        tmln=self.data.TMLN_SCRIPT
        ckeys = ['CMD', 'IDX1', 'IDX2', 'INTCYCLES', 'COMB', 'FLM', 'AT', 'CT', 'BIAS1', 'SHIFT1', 'K1', 'LO1', 'MODEC', 'MODES', 'MODEA', 'BIAS2', 'SHIFT2', 'K2', 'LO2', 'CCH_INTCYCLES']
        data = {}
        n = len(tmln)
        for i, col in enumerate(ckeys):
            data[col] = [tmln[j][i].data.tolist() for j in range(n)]
        self.obstable = pd.DataFrame(data=data)

    def comb_cts_fit(self, obsid_comb, comb_index):
        # Placeholder for the actual comb fitting logic
        # Example output based on module_v0.txt
        freq_calib = lambda x: 6496.291429 - 0.099271 * x
        return freq_calib

    def plot_L01B_comb_cts_counts_fit_vs_comb_window(self):
        # Placeholder for the plotting logic
        pass


    def cts_counts_to_continuum(self, index_list = None):
        # This method integrates the counts per cycle over the frequency range and provides the total signal for each record.
        # We return a tuple with receiver 1 and receiver 2 data, an on/off index list and a quality flag list.
        # Input parameters:
        # index_list - list of indexes OFF/ON/COMB. If None, this is determined from the data.
        # Output:
        # ccts = average frequency cts channels versus  index_list ; ccts = [ccts1, ccts2]
        # onoff = on/off index list
        # qflag = quality flag list

        self.receiver = self.data.CTSID.data
        ctsmat = self.data.CTS_RAW.data
        # There are some inconsistencies in the data skeleton - we do not care it isn't a big issue an is fixed in L1 data (I hope)
        try:
            bitshift = self.data.MODEON_SHFT.data # Shift 
            # print("Data skeleton inconsistency:", self.fname, "uses", "MODEON_SHFT")
        except:
            bitshift = self.data.MODE_ON_SHFT.data # Shift
            # print("Data skeleton inconsistency:", self.fname, "uses", "MODE_ON_SHFT") # this seems to be the expected behaviour
        tint = self.data.T_ON.data # Integration 
        tintcomb = self.data.INT_COMB.data  
        scale = 2**bitshift/tint
        scale_comb = 2**bitshift/tintcomb
        mat=ctsmat.copy()
        n=mat.shape[0]
        factor = np.concatenate((scale_comb*np.ones(2), scale*np.ones(n-2)), axis=0).reshape(n, 1) # shape so that each row corresponds to a record.
        self.cts_mat = factor*mat
        self.cts1 = self.cts_mat[self.receiver==1,:]
        self.cts2 = self.cts_mat[self.receiver==2,:]

        # average over the frequency channels
        self.ccts1 = np.nanmean(self.cts1, axis=1)
        self.ccts2 = np.nanmean(self.cts2, axis=1)
        self.ccts = [self.ccts1, self.ccts2]

        # determine the on/off index list
        if index_list:
            self.onoff = index_list
        else:
            # determine the mean value of all records except the first two (comb for receiver 1 and 2)
            meancts = np.mean(self.ccts1[2:]), np.mean(self.ccts2[2:])
            # determine the on/off mask for receiver 1 and 2, then compare them
            self.onoff1 = np.where(self.ccts1 > meancts[0], 1, 0.5) #
            self.onoff2 = np.where(self.ccts2 > meancts[1], 1, 0.5) #
            if hasattr(self, 'best_sky_list'):
                self.onoff1[self.onoff1==0] = 0.5
                self.onoff2[self.onoff2==0] = 0.5
                self.onoff1[np.array(self.best_sky_list, dtype=bool)] = 0
                self.onoff2[np.array(self.best_sky_list, dtype=bool)] = 0
            else:
                self.onoff1 = np.where(self.ccts1 > meancts[0], 1, 0)
                self.onoff2 = np.where(self.ccts2 > meancts[1], 1, 0)
            # if they agree at a given element, we keep the value and set quality flag = 1, otherwise we set the quality flag = 0
            self.onoff = np.where(self.onoff1 == self.onoff2, self.onoff1, 0.5)
            self.qflag = np.where(self.onoff1 == self.onoff2, 1, 0)
            # set first observation to -1
            self.onoff[0] = -1
            self.qflag[0] = -1
            self.onoff[self.obstable["FLM"]=="HOT"] = -2
            self.qflag[self.obstable["FLM"]=="HOT"] = -2
            #print(self.obstable["FLM"])
            #print(self.obstable["FLM"].shape)
            #print(self.qflag.shape)
            
        # Default guess for the best sky indexes: the off indexes below the off average + excluding the first and last record of each continous interval
        self.best_sky_list = np.zeros(len(self.onoff))
        self.best_sky_list[np.logical_and(np.logical_and(self.onoff==0, self.qflag==1), self.ccts1<np.mean(self.ccts1[self.onoff==0]))] = 1
        # use diff to find the start and end of the continous intervals
        diff = np.diff(self.best_sky_list) # we need to add a zero at the beginning to get the correct indexes
        diff = np.insert(diff, 0, 0)
        self.best_sky_list[np.logical_or(diff==1, diff==-1)] = 0
        
        return self.ccts, self.onoff, self.qflag
            

    def plot_L01B_onoff_ccts_counts_vs_index_list(self, index_list = None, observation_mode = 'Nadir_stare', manualbestsky = False, versus='index', limbcrossing=None):
        # Run the cts_counts_to_continuum method first and then plot the results. the on/off index list is used to plot the data.
        # Comb, on and off data are plotted using different colors and markers.
        # grid is set to True, xlabel is set to "Index", ylabel is set to "Counts per cycle"
        # The titles are set to "Receiver 1" and "Receiver 2"
        # the suptitle includes the name of the file and date of the observation
        # there are two subplots next to each other, one for receiver 1 and one for receiver 2
        # The legend is set to "Comb", "On", "Off"
        # the average (excluding comb) is plotted as a horizontal line

        plt.rcParams.update({'font.size': 14})
        
        ccts, onoff, qflag = self.cts_counts_to_continuum(index_list)
        fig, axs = plt.subplots(1, 2, figsize=(15, 4.7))
        fig.suptitle('L01B data ObsID='+str(self.obsid) + ' ' + str(self.data.GMT.data[0][0:22]), fontsize=16)
        axs[0].set_title('Receiver 1, BIAS = '+str(self.obstable['BIAS1'][0])+', LO1='+str(self.obstable['LO1'][0])+' GHz', fontsize=16)
        axs[1].set_title('Receiver 2, BIAS = '+str(self.obstable['BIAS2'][0])+', LO2='+str(self.obstable['LO2'][0])+' GHz', fontsize=16)

        # x axis - index number
        if versus=='index':
            x = np.arange(len(onoff))
        else:
            x = np.array([datetime.strptime(self.data.GMT.data[i], '%Y-%m-%dT%H:%M:%S.%f') for i in range(0, len(onoff)*2, 2)])
            
        axs[0].plot(x[onoff!=-2], ccts[0][onoff!=-2], 'ko')
        axs[1].plot(x[onoff!=-2], ccts[1][onoff!=-2], 'ko')
        axs[0].plot(x[onoff==-1], ccts[0][onoff==-1], 'mo', label='Comb')
        axs[0].plot(x[onoff==-2], ccts[0][onoff==-2], 'r^', label='Hot')
        axs[0].plot(x[onoff==0], ccts[0][onoff==0], 'o', color='deepskyblue', label='Sky')
        axs[0].plot(x[onoff==1], ccts[0][onoff==1], 'go', label='On')
        axs[1].plot(x[onoff==-1], ccts[1][onoff==-1], 'mo', label='Comb')
        axs[1].plot(x[onoff==-2], ccts[1][onoff==-2], 'r^', label='Hot')
        axs[1].plot(x[onoff==0], ccts[1][onoff==0], 'o', color='deepskyblue', label='Sky')
        axs[1].plot(x[onoff==1], ccts[1][onoff==1], 'go', label='On')

        # plot limb crossing as vertical dashed line if it is, 
        if limbcrossing:
            # convert string to datetime
            limbcrossing = datetime.strptime(limbcrossing, '%Y-%m-%dT%H:%M:%S')
            axs[0].axvline(limbcrossing, color='magenta', linestyle='--', label='Limb crossing')
            axs[1].axvline(limbcrossing, color='magenta', linestyle='--', label='Limb crossing')

        axs[0].set_xlabel('Index')
        axs[1].set_xlabel('Index')
        # if versus not 'index', set the x axis to date format
        

        if versus != 'index':
            myFmt = mdates.DateFormatter('%H:%M')
            axs[0].xaxis.set_major_formatter(myFmt)
            axs[1].xaxis.set_major_formatter(myFmt)
            axs[0].set_xlabel('Time (UTC)')
            axs[1].set_xlabel('Time (UTC)')
            
        
        #axs[0].axhline(np.mean(ccts[0][1:]), color='black', linestyle='-.', label='Mean')
        #axs[1].axhline(np.mean(ccts[1][1:]), color='black', linestyle='-.', label='Mean')
        # left middle
        axs[0].legend(loc = 'center left')
        axs[1].legend(loc = 'center left')
        axs[0].grid(True)
        axs[1].grid(True)
        
        axs[0].set_ylabel('Counts per cycle')
        axs[1].set_ylabel('Counts per cycle')

        if observation_mode in ['Nadir_stare', 'Limb_stare']: 
            # plot the average of ons and offs including the standard deviation
            meanon1 = np.mean(ccts[0][onoff==1])
            meanoff1 = np.mean(ccts[0][onoff==0])
            meanon2 = np.mean(ccts[1][onoff==1])
            meanoff2 = np.mean(ccts[1][onoff==0])
            stdon1 = np.std(ccts[0][onoff==1])
            stdoff1 = np.std(ccts[0][onoff==0])
            stdon2 = np.std(ccts[1][onoff==1])
            stdoff2 = np.std(ccts[1][onoff==0])
            # draw a line at the average of the on and off +- std, no labels
            #axs[0].axhline(meanon1, color='green', linestyle='--')
            #axs[0].axhline(meanon1+stdon1, color='green', linestyle=':')
            #axs[0].axhline(meanon1-stdon1, color='green', linestyle=':')
            #axs[0].axhline(meanoff1, color='blue', linestyle='--')
            #axs[0].axhline(meanoff1+stdoff1, color='blue', linestyle=':')
            #axs[0].axhline(meanoff1-stdoff1, color='blue', linestyle=':')
            #axs[1].axhline(meanon2, color='green', linestyle='--')
            #axs[1].axhline(meanon2+stdon2, color='green', linestyle=':')
            #axs[1].axhline(meanon2-stdon2, color='green', linestyle=':')
            #axs[1].axhline(meanoff2, color='blue', linestyle='--')
            #axs[1].axhline(meanoff2+stdoff2, color='blue', linestyle=':')
            #axs[1].axhline(meanoff2-stdoff2, color='blue', linestyle=':')
        
        plt.tight_layout()
        plt.savefig(self.fname + '_onoff_ccts_counts_vs_index_list.jpg', dpi=300, bbox_inches='tight')
        plt.show()

        if manualbestsky:
            # Prompt the user to select the best sky indexes - if empty, leave it as is
            print("Total number of records:", len(onoff))
            sidx = input("Enter the list of indexes to be used for TSYS calibration (e.g.: 1,2,5-10): ")
            if not sidx:
                return
            
            # save a mask for the best sky indexes
            self.best_sky_list = np.zeros(len(onoff))
            self.best_sky_list[getindexes(sidx)] = 1
            return self.best_sky_list
    
    def plot_FS_quicklook(self):
        """
        Generates quicklook plots for the Frequency Switch (FS) data.

        This function averages the spectra corresponding to the first two unique local oscillator (LO) frequencies
        from the observation table, excluding the comb signals. It then computes the difference between the two
        averaged signals for both receivers and creates two sets of plots:
        1. Subtracted signals for each receiver.
        2. Overlaid signals for both LO frequencies for each receiver.

        The resulting plots are saved as JPG files.
        """
        
        # Get unique local oscillator frequencies
        lo1s = np.unique(self.obstable.LO1)
        lo2s = np.unique(self.obstable.LO2)

        # Average the spectra corresponding to LO1=lo1s[0] (excluding comb signals)
        lomask1 = np.logical_and(self.obstable.LO1 == lo1s[0], 
                                 self.obstable.COMB == '0')
        cts1lo1 = np.mean(self.cts1[lomask1], axis=0)
        cts2lo1 = np.mean(self.cts2[lomask1], axis=0)

        # Average the spectra corresponding to LO1=lo1s[1] (excluding comb signals)
        lomask2 = np.logical_and(self.obstable.LO1 == lo1s[1],
                                 self.obstable.COMB == '0')
        cts1lo2 = np.mean(self.cts1[lomask2], axis=0)
        cts2lo2 = np.mean(self.cts2[lomask2], axis=0)

        # Compute the difference between the two signals for both receivers
        y1 = cts1lo1 - cts1lo2
        y2 = cts2lo1 - cts2lo2

        # Create subplots for the subtracted signals
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('Quicklook ' + self.fname, fontsize=16)
        axs[0].set_title('Receiver 1, subtracted LOs: ' + str(lo1s[0]) + ' - ' + str(lo1s[1]) + ' (GHz)', fontsize=16)
        axs[1].set_title('Receiver 2, subtracted LOs: ' + str(lo2s[0]) + ' - ' + str(lo2s[1]) + ' (GHz)', fontsize=16)

        # Set x axis - frequency channels and plot the subtracted signals
        x = np.arange(len(y1))
        axs[0].plot(x, y1, 'black')
        axs[1].plot(x, y2, 'black')
        axs[0].grid(True)
        axs[1].grid(True)
        axs[0].set_xlabel('Frequency channel')
        axs[1].set_xlabel('Frequency channel')
        axs[0].set_ylabel('Counts per cycle')
        axs[1].set_ylabel('Counts per cycle')
        plt.tight_layout()
        plt.savefig(self.fname + '_quicklook.jpg', dpi=300, bbox_inches='tight')
        plt.show()

        # Create subplots for both LOs overlaid
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('Quicklook ' + self.fname, fontsize=16)
        axs[0].set_title('Receiver 1, both LOs: ' + str(lo1s[0]) + ' & ' + str(lo1s[1]) + ' (GHz)', fontsize=16)
        axs[1].set_title('Receiver 2, both LOs: ' + str(lo2s[0]) + ' & ' + str(lo2s[1]) + ' (GHz)', fontsize=16)

        # Plot both LO signals for each receiver
        axs[0].plot(x, cts1lo1, 'r-', label='LO1=' + str(lo1s[0]) + ' GHz')
        axs[0].plot(x, cts1lo2, 'b:', label='LO2=' + str(lo1s[1]) + ' GHz')
        axs[1].plot(x, cts2lo1, 'r-', label='LO1=' + str(lo2s[0]) + ' GHz')
        axs[1].plot(x, cts2lo2, 'b:', label='LO2=' + str(lo2s[1]) + ' GHz')
        axs[0].grid(True)
        axs[1].grid(True)
        axs[0].set_xlabel('Frequency channel')
        axs[1].set_xlabel('Frequency channel')
        axs[0].set_ylabel('Counts per cycle')
        axs[1].set_ylabel('Counts per cycle')
        axs[0].legend()
        axs[1].legend()
        plt.tight_layout()
        plt.savefig(self.fname + '_quicklook_bothlos.jpg', dpi=300, bbox_inches='tight')
        plt.show()




    def set_best_sky_list(self, best_sky_list):
        self.best_sky_list = np.zeros(len(self.onoff))
        self.best_sky_list[getindexes(best_sky_list)] = 1

    def ave_sky_cts_counts(self, best_sky_list = None):
        if best_sky_list:
            self.best_sky_list = best_sky_list
        
        # calculate the average sky counts
        self.sky1 = np.mean(self.cts1[self.best_sky_list==1], axis=0)
        self.sky2 = np.mean(self.cts2[self.best_sky_list==1], axis=0)
        self.sky = [self.sky1, self.sky2]
    
    def plot_L01B_ave_sky_cts_counts(self, best_sky_list = None):
        if best_sky_list:
            self.set_best_sky_list(best_sky_list)
        
        if not hasattr(self, 'sky'):
            self.ave_sky_cts_counts()

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(self.fname + ' Average Sky Counts ' + str(self.data.GMT.data[0]))
        # local oscilator frequency is stored in loader.obstable['LO1'] and LO2 but we need to check if all values are the same
        self.LO1 = "Varies"
        self.LO2 = "Varies"
        if np.all(self.obstable['LO1']==self.obstable['LO1'][0]):
            self.LO1 = self.obstable['LO1'][0]
        if np.all(self.obstable['LO2']==self.obstable['LO2'][0]):
            self.LO2 = self.obstable['LO2'][0]

        axs[0].set_title('Receiver 1 - LO='+str(self.LO1)+' GHz')
        axs[1].set_title('Receiver 2 - LO='+str(self.LO2)+' GHz')  

        # x axis - frequency channels
        x = np.arange(len(self.sky1))
        axs[0].plot(x, self.sky1, 'r-')
        axs[1].plot(x, self.sky2, 'r-')
        axs[0].grid(True)
        axs[1].grid(True)
        axs[0].set_xlabel('Frequency channel')
        axs[1].set_xlabel('Frequency channel')
        axs[0].set_ylabel('Counts per cycle')
        axs[1].set_ylabel('Counts per cycle')
        plt.show()

    def ave_onoff_cts_counts(self, best_sky_list = None):
        # the purpose of this method is to calculate the average on and off counts (for each receiver)
        # this is useful for the nadir stare mode for example
        if best_sky_list:
            self.best_sky_list = best_sky_list
        
        if not hasattr(self, 'sky'):
            self.ave_sky_cts_counts()

        # calculate the average on and off counts
        self.on1 = np.mean(self.cts1[self.onoff==1], axis=0)
        # for the off, we use the sky counts
        self.on2 = np.mean(self.cts2[self.onoff==1], axis=0)
        # for the off, we use the sky counts
        self.on = [self.on1, self.on2]
        
    def plot_L01B_ave_onoff_cts_counts(self, best_sky_list = None):
        if best_sky_list:
            self.set_best_sky_list(best_sky_list)
        
        if not hasattr(self, 'on'):
            self.ave_onoff_cts_counts()

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('Average On/Off Counts ' + self.fname + ' ' + str(self.data.GMT.data[0]))
        # local oscilator frequency is stored in loader.obstable['LO1'] and LO2 but we need to check if all values are the same
        self.LO1 = "Varies"
        self.LO2 = "Varies"
        if np.all(self.obstable['LO1']==self.obstable['LO1'][0]):
            self.LO1 = self.obstable['LO1'][0]
        if np.all(self.obstable['LO2']==self.obstable['LO2'][0]):
            self.LO2 = self.obstable['LO2'][0]

        axs[0].set_title('Receiver 1 - LO='+str(self.LO1)+' GHz')
        axs[1].set_title('Receiver 2 - LO='+str(self.LO2)+' GHz')  

        # x axis - frequency channels
        # y axis - plot the average on and off counts
        x = np.arange(len(self.on1))
        axs[0].plot(x, self.on1, 'g-', label='On')
        axs[1].plot(x, self.on2, 'g-', label='On')
        axs[0].plot(x, self.sky1, 'r-', label='Off')
        axs[1].plot(x, self.sky2, 'r-', label='Off')
        axs[0].grid(True)
        axs[1].grid(True)
        axs[0].set_xlabel('Frequency channel')
        axs[1].set_xlabel('Frequency channel')
        axs[0].set_ylabel('Counts per cycle')
        axs[1].set_ylabel('Counts per cycle')
        axs[0].legend()
        axs[1].legend()

        plt.show()


    # OBSID : 227 to 232 : internal hot (i.e. we can directly compute the TSYS from that)
    # OBSID :  232+ ...      : external hot (i.e. we need another obsid+HOT to compute the TSYS)

    def tsys_cts_internal(self, hotobsid = None, best_sky_list = None, averaging = (20,40), biases = (541,220)):
        # Input parameters:
        # hotobsid - ObsID of the hot observation. If None, the hot observation is taken from the same observation.
        # best_sky_list - list of indexes to be used for TSYS calibration. If None, the best sky indexes are determined from the data.
        # averaging - tuple of two integers specifying the number of frequency channels to average Tsys over. Default is (20,40).

        if best_sky_list: # manually set best sky indexes if given
            self.set_best_sky_list(best_sky_list)
        
        if not hasattr(self, 'sky'): # get average sky counts, best sky indexes are always computed
            self.ave_sky_cts_counts()
        
        bias1 = self.obstable['BIAS1']
        bias2 = self.obstable['BIAS2']
        if np.all(bias1==bias1[0]):
            self.bias1 = bias1[0]
            self.bias2 = bias2[0] # this should be okay
            skiploadinghot = False
        else:
            # Using a TSYS file directly
            self.bias1 = int(biases[0])
            self.bias2 = int(biases[1])
            # raise Exception("Observation", self.obsid, " not suitable for TSYS calibration")
            hotobsid = self.obsid # to avoid meeting the condition below which looks for internal hot observations
            skiploadinghot = True

        Tc = 2.73 # CMB temperature

        # we decide what to do about the hot observation
        if not hotobsid:
            # see if there is a hot observation within
            self.hotobs = self
            hotobs = self.hotobs
            hotmask = np.logical_and(hotobs.obstable['FLM']=='HOT',
                                        hotobs.obstable['COMB']!='1')
            
            Thot = np.mean([hotobs.data.HK_CHL_1_T.data,
                            hotobs.data.HK_CHL_2_T.data,
                            hotobs.data.HK_CHL_3_T.data,
                            hotobs.data.HK_CHL_4_T.data],
                            axis=0) + 273.15
            

            Thot = np.nanmean(Thot) # TODO: check if this is okay or if we need to use the Thot for the specific record
            # print("Thot =", Thot)
            
            hot1 = np.mean(hotobs.cts1[hotmask], axis=0)
            hot2 = np.mean(hotobs.cts2[hotmask], axis=0)   
            
            # calculate the system temperature
            
            
        else:
            if self.obsid in range(227, 233):
                print("Consider using the internal hot observation available for ObsID", self.obsid)
            # We use the hot from a separate observation

            #if skiploadinghot:
            #    hotobs = self
            #else:    
            hotobs = SWIDataLoader(hotobsid)
            self.hotobs = hotobs
            hotobs.obstable['BIAS1'] = hotobs.obstable['BIAS1'].astype(int)
            hotobs.obstable['BIAS2'] = hotobs.obstable['BIAS2'].astype(int)
            hotmask1 = np.logical_and(np.logical_and(hotobs.obstable['FLM']=='HOT',
                                        hotobs.obstable['COMB']=='0'),
                                        hotobs.obstable['BIAS1']==self.bias1)
            hotmask2 = np.logical_and(np.logical_and(hotobs.obstable['FLM']=='HOT',
                                        hotobs.obstable['COMB']=='0'),
                                        hotobs.obstable['BIAS2']==self.bias2)
            skymask1 = np.logical_and(np.logical_and(hotobs.obstable['FLM']=='SKY',
                                        hotobs.obstable['COMB']=='0'),
                                        hotobs.obstable['BIAS1']==self.bias1)
            skymask2 = np.logical_and(np.logical_and(hotobs.obstable['FLM']=='SKY',
                                        hotobs.obstable['COMB']=='0'),
                                        hotobs.obstable['BIAS2']==self.bias2)
            hotmask = np.logical_and(hotobs.obstable['FLM']=='HOT',
                                        hotobs.obstable['COMB']!='1')
            Thot = np.mean([hotobs.data.HK_CHL_1_T.data,
                            hotobs.data.HK_CHL_2_T.data,
                            hotobs.data.HK_CHL_3_T.data,
                            hotobs.data.HK_CHL_4_T.data],
                            axis=0) + 273.15
            Thot = np.nanmean(Thot) # TODO: check if this is okay or if we need to use the Thot for the specific record
            # print("Thot =", Thot)
            print(hotmask)
            hotobs.cts_counts_to_continuum()
            hot1 = np.mean(hotobs.cts1[hotmask], axis=0)
            hot2 = np.mean(hotobs.cts2[hotmask], axis=0)
            tsyssky1 = np.mean(hotobs.cts1[hotmask], axis=0)
            tsyssky2 = np.mean(hotobs.cts2[hotmask], axis=0)         

            # Debugging (print all inputs for _calibrate)
            # print("hot1 =", hot1)
            # print("tsyssky1 =", tsyssky1)
            # print("Thot =", Thot)
            # print("Tc =", Tc)
            
            # calculate the system temperature for the tsys file
            TsysTsys1 = self._calibrate(hot1, tsyssky1, Thot, Tc)
            TsysTsys2 = self._calibrate(hot2, tsyssky2, Thot, Tc)

            # remove artefacts from the edges
            TsysTsys1 = self.removeartefacts(TsysTsys1)
            TsysTsys2 = self.removeartefacts(TsysTsys2)
            # average over the frequency channels
            TsysTsys1 = np.pad(TsysTsys1, (averaging[0]//2, averaging[0]-1-averaging[0]//2), mode='median') # pad with median
            TsysTsys2 = np.pad(TsysTsys2, (averaging[1]//2, averaging[1]-1-averaging[1]//2), mode='median')
            TsysTsys1 = np.convolve(TsysTsys1, np.ones(averaging[0])/averaging[0], mode='valid')
            TsysTsys2 = np.convolve(TsysTsys2, np.ones(averaging[1])/averaging[1], mode='valid')

            self.TsysTsys = [TsysTsys1, TsysTsys2]

        Tsys1 = self._calibrate(hot1, self.sky1, Thot, Tc)
        Tsys2 = self._calibrate(hot2, self.sky2, Thot, Tc)
        # remove artefacts from the edges

        Tsys1 = self.removeartefacts(Tsys1)
        Tsys2 = self.removeartefacts(Tsys2)
        
        

        # average over the frequency channels # N-1-N//2
        Tsys1 = np.pad(Tsys1, (averaging[0]//2, averaging[0]-1-averaging[0]//2), mode='median') # pad with median
        Tsys2 = np.pad(Tsys2, (averaging[1]//2, averaging[1]-1-averaging[1]//2), mode='median')
        Tsys1 = np.convolve(Tsys1, np.ones(averaging[0])/averaging[0], mode='valid')
        Tsys2 = np.convolve(Tsys2, np.ones(averaging[1])/averaging[1], mode='valid')
        
        self.Tsys = [Tsys1, Tsys2]

        # store the time difference between used hot and sky observations
        self.hotobs.time = datetime.strptime(self.hotobs.data.GMT.data[0], '%Y-%m-%dT%H:%M:%S.%f')
        
        self.timediff = (self.time - self.hotobs.time).total_seconds()/86400 # in days

    def removeartefacts(self, Tsys, edge=30, criterion=3):
        validmean = np.mean(Tsys[edge:-edge])
        validstd = np.std(Tsys[edge:-edge])
        Tsysleftedge = Tsys[:edge]
        Tsysrightedge = Tsys[-edge:]
        Tsysleftedge[np.logical_or(Tsysleftedge>validmean+criterion*validstd,Tsysleftedge<validmean-criterion*validstd)] = validmean
        Tsysrightedge[np.logical_or(Tsysrightedge>validmean+criterion*validstd,Tsysrightedge<validmean-criterion*validstd)] = validmean
        Tsys[:edge] = Tsysleftedge
        Tsys[-edge:] = Tsysrightedge
        return Tsys

    def get_thot_tsky(self, hotobsid = None, best_sky_list = None):
        self.tsys_cts_internal(hotobsid, best_sky_list)
        return (self.bias1, self.timediff, self.Tsys[0]), (self.bias2 ,self.timediff, self.Tsys[1])

    def get_thot_tsys(self, hotobsid = None, best_sky_list = None):
        self.tsys_cts_internal(hotobsid, best_sky_list)
        return (self.bias1, self.timediff, self.TsysTsys[0]), (self.bias2 ,self.timediff, self.TsysTsys[1])

    def plot_L01B_tsys_cts_internal_nomean(self, hotobsid = None, best_sky_list = None, plotTsysTsys = False):
        self.plot_L01B_tsys_cts_internal(hotobsid, best_sky_list, plotTsysTsys, (1,1))

    def plot_L01B_tsys_cts_internal(self, hotobsid = None, best_sky_list = None, plotTsysTsys = False, averaging = (20,40)):
        self.tsys_cts_internal(hotobsid, best_sky_list, averaging)
        
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        # make the two lines of suptitle aligned
        #fig.suptitle('TSYS Calibration ' + self.fname + ' ' + str(self.data.GMT.data[0] + '\n' + 'Using hot      ' + self.hotobs.fname + ' ' + str(self.hotobs.data.GMT.data[0])))
        
        fig.suptitle('%18s %40s %27s\n%18s %40s %27s' % ('Using sky', self.fname, str(self.data.GMT.data[0]), 'Using hot', self.hotobs.fname, str(self.hotobs.data.GMT.data[0])), fontfamily='monospace')
        if not hasattr(self, 'LO1'):
            self.LO1 = "Varies"
            self.LO2 = "Varies"
            if np.all(self.obstable['LO1']==self.obstable['LO1'][0]):
                self.LO1 = self.obstable['LO1'][0]
            if np.all(self.obstable['LO2']==self.obstable['LO2'][0]):
                self.LO2 = self.obstable['LO2'][0]
        # plot T vs frequency channel
        x = np.arange(len(self.Tsys[0]))
        axs[0].plot(x, self.Tsys[0], 'b-')
        axs[1].plot(x, self.Tsys[1], 'b-')
        if plotTsysTsys:
            axs[0].plot(x, self.TsysTsys[0], 'g-')
            axs[1].plot(x, self.TsysTsys[1], 'g-')
        axs[0].legend(['Tsys', 'TsysTsys'])
        axs[1].legend(['Tsys', 'TsysTsys'])
        axs[0].grid(True)
        axs[1].grid(True)
        axs[0].set_xlabel('Frequency channel')
        axs[1].set_xlabel('Frequency channel')
        axs[0].set_ylabel('System temperature [K]')
        axs[1].set_ylabel('System temperature [K]')
        axs[0].set_title('Receiver 1 - LO='+str(self.LO1)+' GHz' + ', bias = '+str(self.bias1))
        axs[1].set_title('Receiver 2 - LO='+str(self.LO2)+' GHz' + ', bias = '+str(self.bias2))
        axs[0].set_ylim(500, 2000)
        axs[1].set_ylim(1000, 5000)
        plt.tight_layout()
        plt.savefig(f'obsid_{self.obsid}_hotobsid_{self.hotobs.obsid}_tsys_cts_internal.png', dpi=300, bbox_inches='tight')
        plt.show()

        # print out mean and std of tsys
        print("Receiver 1: Mean Tsys =", np.nanmean(self.Tsys[0]), "STD Tsys =", np.nanstd(self.Tsys[0]))
        print("Receiver 2: Mean Tsys =", np.nanmean(self.Tsys[1]), "STD Tsys =", np.nanstd(self.Tsys[1]))
        if plotTsysTsys:
            print("Receiver 1: Mean TsysTsys =", np.nanmean(self.TsysTsys[0]), "STD TsysTsys =", np.nanstd(self.TsysTsys[0]))
            print("Receiver 2: Mean TsysTsys =", np.nanmean(self.TsysTsys[1]), "STD TsysTsys =", np.nanstd(self.TsysTsys[1]))
    

    def _RJT(self,T, f):
        """ Convert temperatures to Rayleigh Jeans scale. """
        h = 6.62607015e-34
        k = 1.380649e-23
        x = h*f/k
        return x/(np.exp(x/T) - 1.0)
    
    def _calibrate(self, hot, sky, Th, Tc):
        """ Calculate noise specturm according to Y-factor method. """
        Y = hot/sky
        cal = (Th - Y*Tc)/(Y - 1.0)
        return cal

    def calibratebias(self, rx, bias):
        """ 
        Calibrate the system temperature for a selected bias 
        rx - receiver number (1 or 2)
        bias - integer specifying the bias
        """
        if not 'TSYS' in self.fname:
            raise Exception('Cannot perform TSYS calibration using', self.fname)
        if rx==1:
            curbias = self.bias1
            idoffset = 0
        elif rx==2:
            curbias = self.bias2
            idoffset = 1
        else:
            raise Exception("Invalid receiver argument rx", rx)
        # reduced indexes:
        rhi = np.argmax(np.logical_and(np.logical_and(self.comb==0, self.FLM=='HOT'),curbias==bias))
        rsi = np.argmax(np.logical_and(np.logical_and(self.comb==0, self.FLM=='SKY'),curbias==bias))
        # full indexes
        hotidx = idoffset+2*rhi
        skyidx = idoffset+2*rsi
        fLO = 1e9*[self.LO1,self.LO2][rx-1][rhi]
        Th = self._RJT(self.Thot[hotidx],fLO)
        Tc = self._RJT(2.73,fLO)

        hot = self.cts_mat[hotidx,:]
        sky = self.cts_mat[skyidx,:]

        Tsys = self._calibrate(hot,sky,Th,Tc)
        
        return Tsys
    
    def apply_tsys_cts(self, useTsysTsys = False):
        # Apply the system temperature to the counts
        # We use the formula T = (CTS_on/CTS_off - 1)*Tsys
        # First we need to check if the system temperature is available
        # if not, we need to calculate it
        if not hasattr(self, 'Tsys'):
            self.tsys_cts_internal() # we try to obtain the system temperature
        if useTsysTsys:
            TSYS = self.TsysTsys
        else:
            TSYS = self.Tsys
        
        # apply the system temperature to the counts - use best sky list for the off
        CTSoff1 = np.mean(self.cts1[self.best_sky_list==1], axis=0)
        CTSoff2 = np.mean(self.cts2[self.best_sky_list==1], axis=0)
        self.T1 = (self.cts1/CTSoff1 - 1)*TSYS[0]
        self.T2 = (self.cts2/CTSoff2 - 1)*TSYS[1]
        #print("T1", T1)
        #print("T2", T2)



    