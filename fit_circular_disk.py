import switools as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
from scipy.interpolate import griddata
from matplotlib.patches import Circle

# this script will be executed with one input argument of obsid
if __name__ == "__main__":
    obsid = int(sys.argv[1])
    ds = st.load(obsid)

    CT = ds.CT.data
    AT = ds.AT.data
    CCH1 = ds.CCH1.data
    CCH2 = ds.CCH2.data

    # Define the 2D circular Heaviside function
    # Smoothed circular Heaviside function with epsilon as a fit parameter
    def circular_heaviside(xy, AT0, CT0, R, A, epsilon):
        x, y = xy
        r = np.sqrt((x - AT0)**2 + (y - CT0)**2)
        return A / (1 + np.exp((r - R) / epsilon))
    
        # the same equation in Latex:
        # \begin{equation}
        # T(x,y) = \frac{A}{1 + e^{\frac{r - R}{\epsilon}}},
        # \end{equation}
        # where \( r = \sqrt{(x - AT0)^2 + (y - CT0)^2} \), R is the radius, A is the amplitude, and epsilon is the smoothing parameter. 

    # Fit the data
    # Initial guess for CT0, AT0, R, A, epsilon
    initial_guess = [0.001, 0.001, 0.5, 150, 0.01]
    popt1, pcov1 = curve_fit(circular_heaviside, (AT, CT), CCH1, p0=initial_guess)
    popt2, pcov2 = curve_fit(circular_heaviside, (AT, CT), CCH2, p0=initial_guess)
    # Extract fitted parameters, including uncertainties
    CT0_1, AT0_1, R1, A1, eps1 = popt1
    CT0_2, AT0_2, R2, A2, eps2 = popt2
    perr1 = np.sqrt(np.diag(pcov1))
    perr2 = np.sqrt(np.diag(pcov2))
    print("==========================")
    print("Fitting results for obsID:", obsid)
    print("CCH1 fit parameters:")
    print("AT0 = {:.5f} ± {:.5f}".format(CT0_1*60, perr1[0]*60)) # convert to arcmin
    print("CT0 = {:.5f} ± {:.5f}".format(AT0_1*60, perr1[1]*60)) # convert to arcmin
    print("R = {:.5f} ± {:.5f}".format(R1*60, perr1[2]*60)) # convert to arcmin
    print("A = {:.5f} ± {:.5f}".format(A1, perr1[3]))
    print("epsilon = {:.5f} ± {:.5f}".format(eps1, perr1[4]))
    print("Angular = {:.5f} ± {:.5f}".format(ds.ANGULAR.values[0], 0))
    print("CCH2 fit parameters:")
    print("AT0 = {:.5f} ± {:.5f}".format(CT0_2*60, perr2[0]*60)) # convert to arcmin
    print("CT0 = {:.5f} ± {:.5f}".format(AT0_2*60, perr2[1]*60)) # convert to arcmin
    print("R = {:.5f} ± {:.5f}".format(R2*60, perr2[2]*60)) # convert to arcmin
    print("A = {:.5f} ± {:.5f}".format(A2, perr2[3]))
    print("epsilon = {:.5f} ± {:.5f}".format(eps2, perr2[4]))
    print("==========================")

    # save the fitted offset centers to a file in offset_center directory
    # create the directory if it doesn't exist
    with open(f"pointing_offsets/deltas_{obsid}.txt", "w") as f:
        f.write("DeltaAT0CCH1,DeltaCT0CCH1,DeltaAT0CCH2,DeltaCT0CCH2\n")
        f.write("%f,%f,%f,%f\n" % (CT0_1*60, AT0_1*60, CT0_2*60, AT0_2*60)) # units in arcmin
        
    f.close()

    # Create grid
    grid_x, grid_y = np.mgrid[AT.min():AT.max():500j, CT.min():CT.max():500j]
    points = np.vstack((AT, CT)).T
    # figure out limits for plotting based on absolute values of AT and CT 
    xmin = min([-AT.max(), AT.min()])
    xmax = -xmin
    ymin = min([-CT.max(), CT.min()])
    ymax = -ymin

    # Evaluate fitted models on the grid
    model_z1 = circular_heaviside((grid_x, grid_y), *popt1)
    model_z2 = circular_heaviside((grid_x, grid_y), *popt2)

    # Interpolate observed data to the same grid
    obs_z1 = griddata(points, CCH1, (grid_x, grid_y), method='cubic')
    obs_z2 = griddata(points, CCH2, (grid_x, grid_y), method='cubic')

    # Compute residuals
    resid_z1 = obs_z1 - model_z1
    resid_z2 = obs_z2 - model_z2

    # Beam width in degrees (8' and 4')
    beam_radius1 = 8.730520/2
    beam_radius2 = 4.956557/2 

    # Plot function
    # change default matplotlib fontsize to 14
    plt.rcParams.update({'font.size': 14})

    def plot_set(obs, model, resid, CT0, AT0, R, A, beam_radius, title_prefix, LO):
        fig, axs = plt.subplots(1, 3, figsize=(18, 4.6), sharex=True, sharey=True)
        scale=0.8
        im0 = axs[0].pcolormesh(grid_x*60, grid_y*60, obs, shading='auto', cmap='jet')
        axs[0].set_title(f"{title_prefix} Observed", fontsize=16)
        axs[0].add_patch(Circle((AT0, CT0), beam_radius, edgecolor='white', facecolor='none', lw=1.5))
        cb = plt.colorbar(im0, ax=axs[0], label=r"$T_{RJ}$ (K)",shrink=scale)
        cb.ax.tick_params(labelsize=14)

        im1 = axs[1].pcolormesh(grid_x*60, grid_y*60, model, shading='auto', cmap='jet', vmin=(cb.ax.get_ylim()[0]), vmax=(cb.ax.get_ylim()[1]))
        axs[1].set_title(f"{title_prefix} Fitted model", fontsize=16)
        axs[1].add_patch(Circle((AT0, CT0), beam_radius, edgecolor='white', facecolor='none', lw=1.5))
        cb = plt.colorbar(im1, ax=axs[1], label=r"$T_{RJ}$ (K)",shrink=scale)
        cb.ax.tick_params(labelsize=14)

        im2 = axs[2].pcolormesh(grid_x*60, grid_y*60, resid, shading='auto', cmap='bwr')
        axs[2].set_title(f"{title_prefix} Residual (Obs - Model)", fontsize=16)
        axs[2].add_patch(Circle((AT0, CT0), beam_radius, edgecolor='black', facecolor='none', lw=1.5))
        cb = plt.colorbar(im2, ax=axs[2], label=r"$\Delta T_{RJ}$ (K)",shrink=scale)
        cb.ax.tick_params(labelsize=14)

        for ax in axs:
            ax.set_xlabel(r"AT ($\mathrm{'}$)", fontsize=14)
            ax.set_ylabel(r"CT ($\mathrm{'}$)", fontsize=14)
            ax.set_aspect('equal')
            ax.set_xlim(xmin*60, xmax*60)
            ax.set_ylim(ymin*60, ymax*60)

        #plt.suptitle(f"ObsID {obsid}, {title_prefix} — LO = {LO:.3f} GHz", fontsize=16)
        #plt.tight_layout()
        plt.tight_layout(rect=[0, 0, 1, 0.93])  # Reserve space at the top
        plt.suptitle(f"ObsID {obsid}, {title_prefix} — LO = {LO:.3f} GHz", fontsize=16)
        plt.savefig(f"Figures/fit_circular_disk_{title_prefix}_{obsid}.jpg", dpi=300, bbox_inches='tight')
        plt.close()

    # Plot for CCH1
    plot_set(obs_z1, model_z1, resid_z1, CT0_1, AT0_1, R1, A1, beam_radius1, 'CCH1', ds.LO1.values[0])

    # Plot for CCH2
    plot_set(obs_z2, model_z2, resid_z2, CT0_2, AT0_2, R2, A2, beam_radius2, 'CCH2', ds.LO2.values[0])
