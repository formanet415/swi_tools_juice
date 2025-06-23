import swincloadobsid 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt



def main():
    # We look at TSYS files and observation files with internal hot measurements

    obsids = []
    timestamps = []
    tsysch1 = []
    tsysch2 = []
    bias1 = 541
    bias2 = 220

    # TSYS file obsid 174
    nc = swincloadobsid.SWIDataLoader(174)
    nc.tsys_cts_internal(biases=(bias1,bias2)) 
    # store the tsys tsys data and the time
    tsysch1.append(nc.TsysTsys[0])
    tsysch2.append(nc.TsysTsys[1])
    timestamps.append(nc.time)
    obsids.append(174)

    # TSYS file obsid 191
    nc = swincloadobsid.SWIDataLoader(191)
    nc.tsys_cts_internal(biases=(bias1,bias2)) 
    # store the tsys tsys data and the time
    tsysch1.append(nc.TsysTsys[0])
    tsysch2.append(nc.TsysTsys[1])
    timestamps.append(nc.time)
    obsids.append(191)

    # Observation file obsid 227
    nc = swincloadobsid.SWIDataLoader(227)
    nc.tsys_cts_internal() 
    # store tsys data and the time
    tsysch1.append(nc.Tsys[0])
    tsysch2.append(nc.Tsys[1])
    timestamps.append(nc.time)
    obsids.append(227)

    # Observation file obsid 228
    nc = swincloadobsid.SWIDataLoader(228)
    nc.tsys_cts_internal()
    # store tsys data and the time
    tsysch1.append(nc.Tsys[0])
    tsysch2.append(nc.Tsys[1])
    timestamps.append(nc.time)
    obsids.append(228)

    # Observation file obsid 229
    nc = swincloadobsid.SWIDataLoader(229)
    nc.tsys_cts_internal()
    # store tsys data and the time
    tsysch1.append(nc.Tsys[0])
    tsysch2.append(nc.Tsys[1])
    timestamps.append(nc.time)
    obsids.append(229)


    # Plot the evolution of the TSYS - two subplots, chronologically offset by a set amount
    LO1 = nc.obstable['LO1'][0]
    LO2 = nc.obstable['LO2'][0]

    plt.rcParams.update({'font.size': 14})

    plt.figure(figsize=(18, 6))
    plt.subplot(121)
    ch1offset = 300
    for i in range(len(obsids)-1, -1, -1):
        plt.plot(tsysch1[i] + i*ch1offset, label=f'ObsID {obsids[i]}')
    plt.title('TSYS evolution for CTS1, bias %d, LO: %s GHz' % (bias1, LO1))
    plt.xlabel('Channel number')
    plt.ylabel('TSYS (K) + offset')
    plt.legend(loc=1)

    plt.subplot(122)
    ch2offset = 800
    for i in range(len(obsids)-1, -1, -1):
        plt.plot(tsysch2[i] + i*ch2offset, label=f'ObsID {obsids[i]}')
    plt.title('TSYS evolution for CTS2, bias %d, LO: %s GHz' % (bias2, LO2))
    plt.xlabel('Channel number')
    plt.ylabel('TSYS (K) + offset')
    plt.legend(loc=1)

    plt.tight_layout()
    plt.savefig(f"Figures/TSYS_evolution.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Plot the relative variation of the TSYS - two subplots, normalized by ObsID 228
    plt.figure(figsize=(18, 6))
    plt.subplot(121)
    ch1offset = 0.  # Offset for channel 1
    for i in range(len(obsids)-1, -1, -1):
        plt.plot(tsysch1[i] / tsysch1[obsids.index(228)] + i*ch1offset, label=f'ObsID {obsids[i]}')
    plt.title('Relative TSYS variation for CTS1, bias %d, LO: %s GHz' % (bias1, LO1))
    plt.xlabel('Channel number')
    plt.ylabel('Relative TSYS (normalized to ObsID 228)')
    plt.legend(loc=1)

    plt.subplot(122)
    ch2offset = 0.  # Offset for channel 2
    for i in range(len(obsids)-1, -1, -1):
        plt.plot(tsysch2[i] / tsysch2[obsids.index(228)] + i*ch2offset, label=f'ObsID {obsids[i]}')
    plt.title('Relative TSYS variation for CTS2, bias %d, LO: %s GHz' % (bias2, LO2))
    plt.xlabel('Channel number')
    plt.ylabel('Relative TSYS (normalized to ObsID 228)')
    plt.legend(loc=1)

    plt.tight_layout()
    plt.savefig(f"Figures/TSYS_relative_variation.png", dpi=300, bbox_inches='tight')
    plt.show()




if __name__ == "__main__":
    main()