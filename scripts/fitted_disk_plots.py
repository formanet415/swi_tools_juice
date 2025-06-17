import re
import matplotlib.pyplot as plt
import numpy as np

def load_fitted_disks(filepath):
    """
    Load and parse the fitted_disks.txt file.

    Parameters:
    filepath (str): Path to the fitted_disks.txt file.

    Returns:
    dict: A dictionary where keys are obsIDs and values are dictionaries of parameters and uncertainties.
    """
    results = {}
    with open(filepath, 'r') as file:
        lines = file.readlines()

    current_obsid = None
    for line in lines:
        # Match the obsID
        obsid_match = re.match(r"Fitting results for obsID: (\d+)", line)
        if obsid_match:
            current_obsid = int(obsid_match.group(1))
            results[current_obsid] = {"CCH1": {}, "CCH2": {}}
            continue

        # Match the parameters and uncertainties
        param_match = re.match(r"(\w+) = ([\-\d\.]+) ± ([\d\.]+)", line)
        if param_match and current_obsid is not None:
            param_name = param_match.group(1)
            param_value = float(param_match.group(2))
            param_uncertainty = float(param_match.group(3))

            # Determine if it's CCH1 or CCH2
            if not param_name in results[current_obsid]["CCH1"]:
                # If the parameter is already assigned in CCH1, assign it to CCH2
                results[current_obsid]["CCH1"][param_name] = (param_value, param_uncertainty)
            else:
                results[current_obsid]["CCH2"][param_name] = (param_value, param_uncertainty)

    return results



def plot_comparison_across_obsids(results):
    """
    Compare each variable across obsIDs for both CCH1 and CCH2.

    Parameters:
    results (dict): The parsed results from the fitted_disks.txt file.
    """
    # Extract all variables
    variables = set()
    for obsid_data in results.values():
        for component_data in obsid_data.values():
            variables.update(component_data.keys())
    variables = sorted(variables)  # Sort variables for consistent plotting

    # Plot each variable across obsIDs
    for variable in variables:
        plt.figure(figsize=(10, 6))
        obsids = []
        cch1_values = []
        cch1_uncertainties = []
        cch2_values = []
        cch2_uncertainties = []

        for obsid, data in results.items():
            obsids.append(obsid)
            # Extract CCH1 data
            if variable in data["CCH1"]:
                cch1_values.append(data["CCH1"][variable][0])
                cch1_uncertainties.append(data["CCH1"][variable][1])
            else:
                cch1_values.append(None)
                cch1_uncertainties.append(None)

            # Extract CCH2 data
            if variable in data["CCH2"]:
                cch2_values.append(data["CCH2"][variable][0])
                cch2_uncertainties.append(data["CCH2"][variable][1])
            else:
                cch2_values.append(None)
                cch2_uncertainties.append(None)

        # Plot CCH1
        plt.errorbar(
            obsids, cch1_values, yerr=cch1_uncertainties,
            fmt='o', markersize=5, color='blue', label='CCH1',
            ecolor='blue', elinewidth=1, capsize=5, capthick=1
        )

        # Plot CCH2
        plt.errorbar(
            obsids, cch2_values, yerr=cch2_uncertainties,
            fmt='s', markersize=5, color='red', label='CCH2',
            ecolor='red', elinewidth=1, capsize=5, capthick=1
        )

        plt.title(f"Comparison of {variable} Across obsIDs")
        plt.xlabel("obsID")
        plt.ylabel(variable)
        plt.legend()
        plt.grid(True)
        # set xticks to be formatted as integers
        plt.xticks(obsids, [str(obsid) for obsid in obsids])
        plt.savefig(f"Figures/Disk_Fitting_Comparison_{variable}.png")
        plt.show()

def plot_AT0_CT0_R(results):
    """
    Plot AT0, CT0, and R for each obsID.

    Parameters:
    results (dict): The parsed results from the fitted_disks.txt file.

    This function generates plots for the fitted center AT, CT, and radius R for two different configurations (CCH1 and CCH2).
    It also saves the results in a CSV file and generates a PNG file of the plots.

    Plotting Parameters:
    - Title Font Size: 18
    - Label Font Size: 16
    - Legend Font Size: 14 (applied to ticks as well)
    """
    obsids = []
    AT0_valuesCCH1 = []
    AT0_uncertaintiesCCH1 = []
    CT0_valuesCCH1 = []
    CT0_uncertaintiesCCH1 = []
    R_valuesCCH1 = []
    R_uncertaintiesCCH1 = []
    TRJ_valuesCCH1 = []
    TRJ_uncertaintiesCCH1 = []
    epsilon_valuesCCH1 = []
    epsilon_uncertaintiesCCH1 = []
    AT0_valuesCCH2 = []
    AT0_uncertaintiesCCH2 = []
    CT0_valuesCCH2 = []
    CT0_uncertaintiesCCH2 = []
    R_valuesCCH2 = []
    R_uncertaintiesCCH2 = []
    TRJ_valuesCCH2 = []
    TRJ_uncertaintiesCCH2 = []
    epsilon_valuesCCH2 = []
    epsilon_uncertaintiesCCH2 = []
    Angular = []

    for obsid, data in results.items():
        obsids.append(obsid)
        
        # Extract CCH1 data
        AT0_valuesCCH1.append(data["CCH1"]["AT0"][0])
        AT0_uncertaintiesCCH1.append(data["CCH1"]["AT0"][1])
        CT0_valuesCCH1.append(data["CCH1"]["CT0"][0])
        CT0_uncertaintiesCCH1.append(data["CCH1"]["CT0"][1])
        R_valuesCCH1.append(data["CCH1"]["R"][0])
        R_uncertaintiesCCH1.append(data["CCH1"]["R"][1])
        TRJ_valuesCCH1.append(data["CCH1"]["A"][0])
        TRJ_uncertaintiesCCH1.append(data["CCH1"]["A"][1])
        epsilon_valuesCCH1.append(data["CCH1"]["epsilon"][0])
        epsilon_uncertaintiesCCH1.append(data["CCH1"]["epsilon"][1])
        
        # Extract CCH2 data
        AT0_valuesCCH2.append(data["CCH2"]["AT0"][0])
        AT0_uncertaintiesCCH2.append(data["CCH2"]["AT0"][1])
        CT0_valuesCCH2.append(data["CCH2"]["CT0"][0])
        CT0_uncertaintiesCCH2.append(data["CCH2"]["CT0"][1])
        R_valuesCCH2.append(data["CCH2"]["R"][0])
        R_uncertaintiesCCH2.append(data["CCH2"]["R"][1])
        TRJ_valuesCCH2.append(data["CCH2"]["A"][0])
        TRJ_uncertaintiesCCH2.append(data["CCH2"]["A"][1])
        epsilon_valuesCCH2.append(data["CCH2"]["epsilon"][0])
        epsilon_uncertaintiesCCH2.append(data["CCH2"]["epsilon"][1])
        
        # Extract Angular data
        Angular.append(data["CCH1"]["Angular"][0])
    
    # save a table with the resulting offsets. the table will contain columns obsid, DeltaAT1, sDeltaAT1, DeltaAT2, sDeltaAT2, DeltaCT1, sDeltaCT1, DeltaCT2, sDeltaCT2 (all in arcmin)
    import pandas as pd
    data_offsets = {
        "ObsID": obsids,
        "DeltaAT1": AT0_valuesCCH1,
        "sDeltaAT1": AT0_uncertaintiesCCH1,
        "DeltaAT2": AT0_valuesCCH2,
        "sDeltaAT2": AT0_uncertaintiesCCH2,
        "DeltaCT1": CT0_valuesCCH1,
        "sDeltaCT1": CT0_uncertaintiesCCH1,
        "DeltaCT2": CT0_valuesCCH2,
        "sDeltaCT2": CT0_uncertaintiesCCH2
    }
    savetable4 = pd.DataFrame(data_offsets)
    # also include the values in steps
    ATstep = 29.92/60  # step size in arcmin
    CTstep = 8.67/60  # step size in arcmin
    savetable4["DeltaAT1_steps"] = savetable4["DeltaAT1"] / ATstep
    savetable4["sDeltaAT1_steps"] = savetable4["sDeltaAT1"] / ATstep
    savetable4["DeltaAT2_steps"] = savetable4["DeltaAT2"] / ATstep
    savetable4["sDeltaAT2_steps"] = savetable4["sDeltaAT2"] / ATstep
    savetable4["DeltaCT1_steps"] = savetable4["DeltaCT1"] / CTstep
    savetable4["sDeltaCT1_steps"] = savetable4["sDeltaCT1"] / CTstep
    savetable4["DeltaCT2_steps"] = savetable4["DeltaCT2"] / CTstep
    savetable4["sDeltaCT2_steps"] = savetable4["sDeltaCT2"] / CTstep
    # save
    savetable4.to_csv("offset_results_2D.csv", index=False)


    # Font sizes
    title_fontsize = 18
    label_fontsize = 16
    legend_fontsize = 14

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].errorbar(obsids, AT0_valuesCCH1, yerr=AT0_uncertaintiesCCH1, fmt='o', markersize=5, color='blue', label='CCH1')
    axs[0].errorbar(obsids, AT0_valuesCCH2, yerr=AT0_uncertaintiesCCH2, fmt='s', markersize=5, color='red', label='CCH2')
    axs[0].set_title("Fitted center AT", fontsize=title_fontsize)
    axs[0].set_xlabel("ObsID", fontsize=label_fontsize)
    axs[0].set_ylabel(r"$\Delta_{AT}$ ($\mathrm{'}$)", fontsize=label_fontsize)
    axs[0].legend(fontsize=legend_fontsize)
    axs[0].grid(True)
    axs[0].set_xticks(obsids, [str(obsid) for obsid in obsids])
    axs[0].tick_params(axis='both', labelsize=legend_fontsize)

    axs[1].errorbar(obsids, CT0_valuesCCH1, yerr=CT0_uncertaintiesCCH1, fmt='o', markersize=5, color='blue', label='CCH1')
    axs[1].errorbar(obsids, CT0_valuesCCH2, yerr=CT0_uncertaintiesCCH2, fmt='s', markersize=5, color='red', label='CCH2')
    axs[1].set_title("Fitted center CT", fontsize=title_fontsize)
    axs[1].set_xlabel("ObsID", fontsize=label_fontsize)
    axs[1].set_ylabel(r"$\Delta_{CT}$ ($\mathrm{'}$)", fontsize=label_fontsize)
    axs[1].legend(fontsize=legend_fontsize)
    axs[1].grid(True)
    axs[1].set_xticks(obsids, [str(obsid) for obsid in obsids])
    axs[1].tick_params(axis='both', labelsize=legend_fontsize)

    axs[2].errorbar(obsids, R_valuesCCH1, yerr=R_uncertaintiesCCH1, fmt='o', markersize=5, color='blue', label='CCH1')
    axs[2].errorbar(obsids, R_valuesCCH2, yerr=R_uncertaintiesCCH2, fmt='s', markersize=5, color='red', label='CCH2')
    axs[2].plot(obsids, np.array(Angular)/2*60, '--', color='black', label='Angular/2')
    axs[2].set_title("Fitted radius R", fontsize=title_fontsize)
    axs[2].set_xlabel("ObsID", fontsize=label_fontsize)
    axs[2].set_ylabel(r"R ($^\circ$)", fontsize=label_fontsize)
    axs[2].legend(fontsize=legend_fontsize)
    axs[2].grid(True)
    axs[2].set_xticks(obsids, [str(obsid) for obsid in obsids])
    axs[2].tick_params(axis='both', labelsize=legend_fontsize)

    plt.tight_layout()
    plt.savefig("Figures/Disk_Fitting_Comparison_AT0_CT0_R.png")
    plt.show()

    

    # produce a table with the values
    # one column for each obsid
    # first column will contain variable names
    # then we add other columns - two for each obsid (value and uncertainty)
    import pandas as pd
    dataCCH1 = {
        "ObsID": ["$\Delta_{AT}$ ($\mathrm{'}$)", "$\Delta_{AT}$ (step)", "$\Delta_{CT}$ ($\mathrm{'}$)", "$\Delta_{CT}$ (step)","$R$ ($\mathrm{'}$)", "$T_{RJ}$ ($\mathrm{K}$)", "$\epsilon$"]
    }
    ATstep = 29.92/60 # step size in arcmin
    CTstep = 8.67/60 # step size in arcmin
    dataCCH2 = {
        "ObsID": ["$\Delta_{AT}$ ($\mathrm{'}$)", "$\Delta_{AT}$ (step)", "$\Delta_{CT}$ ($\mathrm{'}$)", "$\Delta_{CT}$ (step)","$R$ ($\mathrm{'}$)", "$T_{RJ}$ ($\mathrm{K}$)", "$\epsilon$"]
    }
    # difference between CCH1 and CCH2 - to characterise the offset
    dataoffset = {
        "ObsID": ["$\Delta_{AT}$ ($\mathrm{'}$)", "$\Delta_{AT}$ (step)", "$\Delta_{CT}$ ($\mathrm{'}$)", "$\Delta_{CT}$ (step)"]
                  }
    dfCCH1 = pd.DataFrame(dataCCH1)
    dfCCH2 = pd.DataFrame(dataCCH2)
    dfoffset = pd.DataFrame(dataoffset)
    for obsid in obsids:
        deltaATstepCCH1 = AT0_valuesCCH1[obsids.index(obsid)]/ATstep
        deltaCTstepCCH1 = CT0_valuesCCH1[obsids.index(obsid)]/CTstep
        deltaATstepCCH2 = AT0_valuesCCH2[obsids.index(obsid)]/ATstep
        deltaCTstepCCH2 = CT0_valuesCCH2[obsids.index(obsid)]/CTstep
        # difference between CCH1 and CCH2 - to characterise the offset
        dfoffset[f"{obsid}"] = [(deltaATstepCCH1 - deltaATstepCCH2)*ATstep,(deltaATstepCCH1 - deltaATstepCCH2), (deltaCTstepCCH1 - deltaCTstepCCH2)*CTstep,(deltaCTstepCCH1 - deltaCTstepCCH2)]
        # uncertainties
        deltaATstep_uncertaintyCCH1 = AT0_uncertaintiesCCH1[obsids.index(obsid)]/ATstep
        deltaCTstep_uncertaintyCCH1 = CT0_uncertaintiesCCH1[obsids.index(obsid)]/CTstep
        deltaATstep_uncertaintyCCH2 = AT0_uncertaintiesCCH2[obsids.index(obsid)]/ATstep
        deltaCTstep_uncertaintyCCH2 = CT0_uncertaintiesCCH2[obsids.index(obsid)]/CTstep
        # difference between CCH1 and CCH2 - to characterise the offset - add square of uncertainties
        dfoffset[f"{obsid}_uncertainty"] = [np.sqrt(deltaATstep_uncertaintyCCH1**2 + deltaATstep_uncertaintyCCH2**2)*ATstep,np.sqrt(deltaATstep_uncertaintyCCH1**2 + deltaATstep_uncertaintyCCH2**2),
                                            np.sqrt(deltaCTstep_uncertaintyCCH1**2 + deltaCTstep_uncertaintyCCH2**2)*CTstep,np.sqrt(deltaCTstep_uncertaintyCCH1**2 + deltaCTstep_uncertaintyCCH2**2)]
        
        dfCCH1[f"{obsid}"] = [AT0_valuesCCH1[obsids.index(obsid)], 
                              deltaATstepCCH1,                              
                              CT0_valuesCCH1[obsids.index(obsid)], 
                              deltaCTstepCCH1,
                              R_valuesCCH1[obsids.index(obsid)],
                              TRJ_valuesCCH1[obsids.index(obsid)], 
                              epsilon_valuesCCH1[obsids.index(obsid)]]
        dfCCH2[f"{obsid}"] = [AT0_valuesCCH2[obsids.index(obsid)], 
                              deltaATstepCCH2,
                              CT0_valuesCCH2[obsids.index(obsid)], 
                              deltaCTstepCCH2,
                              R_valuesCCH2[obsids.index(obsid)],
                              TRJ_valuesCCH2[obsids.index(obsid)], 
                              epsilon_valuesCCH2[obsids.index(obsid)]]
        # add uncertainties
        dfCCH1[f"{obsid}_uncertainty"] = [AT0_uncertaintiesCCH1[obsids.index(obsid)], 
                                          deltaATstep_uncertaintyCCH1,
                                          CT0_uncertaintiesCCH1[obsids.index(obsid)], 
                                          deltaCTstep_uncertaintyCCH1,
                                          R_uncertaintiesCCH1[obsids.index(obsid)],
                                          TRJ_uncertaintiesCCH1[obsids.index(obsid)], 
                                          epsilon_uncertaintiesCCH1[obsids.index(obsid)]]
        dfCCH2[f"{obsid}_uncertainty"] = [AT0_uncertaintiesCCH2[obsids.index(obsid)], 
                                          deltaATstep_uncertaintyCCH2,
                                          CT0_uncertaintiesCCH2[obsids.index(obsid)], 
                                          deltaCTstep_uncertaintyCCH2,
                                          R_uncertaintiesCCH2[obsids.index(obsid)],
                                          TRJ_uncertaintiesCCH2[obsids.index(obsid)], 
                                          epsilon_uncertaintiesCCH2[obsids.index(obsid)]]
    # Add a column for the mean value of each parameter and its uncertainty
    # select the variable columns
    CCH1vars = dfCCH1.columns[1::2]
    CCH2vars = dfCCH2.columns[1::2]
    offsetvars = dfoffset.columns[1::2]
    # add the mean value and uncertainty to the dataframe
    # take the mean of vars axis=1
    dfCCH1["Mean"] = dfCCH1[CCH1vars].mean(axis=1)
    dfCCH1["Stdev"] = dfCCH1[CCH1vars].std(axis=1)
    dfCCH2["Mean"] = dfCCH2[CCH2vars].mean(axis=1)
    dfCCH2["Stdev"] = dfCCH2[CCH2vars].std(axis=1)
    dfoffset["Mean"] = dfoffset[offsetvars].mean(axis=1)
    dfoffset["Stdev"] = dfoffset[offsetvars].std(axis=1)
    
    print(dfCCH1)
    print(dfCCH2)
    print(dfoffset)
    
    def round_uncertainties(df):
        """
        Round the uncertainties in the dataframe to one decimal place.
        """
        # rounding - keep digits up to one decimal of uncertainty - this needs to be done one by one
        for i, col in enumerate(df.columns):
            if i % 2 != 0: # values (zeroth column is variable names)
                for j in range(len(df)):
                    # figure out how to round the value - what is the order (power of 10) of the uncertainty
                    digits = int(np.floor(np.log10(df[df.columns[i+1]][j])))
                    if digits < 0:
                        digits = -digits
                        df.loc[j, col] = round(df.loc[j, col], digits)
                        df.loc[j, df.columns[i+1]] = round(df.loc[j, df.columns[i+1]], digits)
                    else:
                        print(df[df.columns[i+1]][j])
                        # no decimal places - represent as formatted string
                        df.loc[j, col] = "{:.0f}".format(df.loc[j, col])
                        df.loc[j, df.columns[i+1]] = "{:.0f}".format(df.loc[j, df.columns[i+1]])
                        
                        
                    
           
    # round the uncertainties
    round_uncertainties(dfCCH1)
    round_uncertainties(dfCCH2)
    round_uncertainties(dfoffset)
    # print the dataframes after rounding
    print("CCH1:")
    print(dfCCH1.to_string(index=False))
    print("CCH2:")
    print(dfCCH2.to_string(index=False))

    # produce a well formatted table for LaTeX
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\caption{Parameters obtained by fitting 2D OTF maps during \acrshort{LEGA}.}")
    print(r"\label{tab:2DMapParams}")
    print(r"\begin{tabular}{@{}l" + "c" * len(obsids) + "c@{}}")
    print(r"\toprule")
    print(r"ObsID & " + " & ".join([str(obsid) for obsid in obsids]) + r" & Mean \& Stdev \\")
    print(r"\midrule")
    print(r"\multicolumn{"+f"{len(obsids)+2}"+r"}{c}{\textbf{CCH1 Fit Parameters}} \\ \midrule")
    for j in range(len(dfCCH1)):
        for i, col in enumerate(dfCCH1.columns):
            if i == 0:
                print(f"{dfCCH1[col][j]} & ", end="")
            if i % 2 != 0:
                print(f"{dfCCH1[col][j]} "+r"$\pm$ ", end="")
            if i == len(dfCCH1.columns) - 1:
                print(f"{dfCCH1[col][j]} \\\\")
            elif i % 2 == 0 and i != 0:
                print(f"{dfCCH1[col][j]} &", end="")
            
    print(r"\midrule")
    print(r"\multicolumn{"+f"{len(obsids)+2}"+r"}{c}{\textbf{CCH2 Fit Parameters}} \\ \midrule")
    for j in range(len(dfCCH2)):
        for i, col in enumerate(dfCCH2.columns):
            if i == 0:
                print(f"{dfCCH2[col][j]} & ", end="")
            if i % 2 != 0:
                print(f"{dfCCH2[col][j]} "+r"$\pm$ ", end="")
            if i == len(dfCCH2.columns) - 1:
                print(f"{dfCCH2[col][j]}"+r" \\ ")
            elif i % 2 == 0 and i != 0:
                print(f"{dfCCH2[col][j]} &", end="")
    print(r"\bottomrule")
    print(r"\multicolumn{"+f"{len(obsids)+2}"+r"}{c}{\textbf{CCH1-CCH2 Offset}} \\ \midrule")
    for j in range(len(dfoffset)):
        for i, col in enumerate(dfoffset.columns):
            if i == 0:
                print(f"{dfoffset[col][j]} & ", end="")
            if i % 2 != 0:
                print(f"{dfoffset[col][j]} "+r"$\pm$ ", end="")
            if i == len(dfoffset.columns) - 1:
                print(f"{dfoffset[col][j]}"+r" \\ ")
            elif i % 2 == 0 and i != 0:
                print(f"{dfoffset[col][j]} &", end="")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    

    


if __name__ == "__main__":
    # Example usage
    filepath = "fitted_disks.txt"
    results = load_fitted_disks(filepath)
    plot_AT0_CT0_R(results)