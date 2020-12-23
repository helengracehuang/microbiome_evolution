# import sample_utils
# import config
import parse_midas_data
import os.path
import os
import sys
import numpy
import scipy.stats as stats
import random
import gzip
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import calculate_linkage_disequilibria_Helen_interspecies_clean as calculate_linkage_disequilibria
# import gene_diversity_utils
# import calculate_substitution_rates
# import clade_utils
# import stats_utils
import figure_utils
from math import log10,ceil,fabs, floor
from numpy.random import randint, choice

ld_directory = '%slinkage_disequilibria/' % (parse_midas_data.data_directory)
intermediate_filename_template = '%s%s.txt.gz'

# species_list = parse_midas_data.parse_good_species_list()
species_list = ['Bacteroides_vulgatus_57955', 'Bacteroides_uniformis_57318', 'Bacteroides_ovatus_58035', 'Alistipes_putredinis_61533', 'Prevotella_copri_61740', 'Bacteroides_stercoris_56735', 'Eubacterium_rectale_56927', 'Bacteroides_xylanisolvens_57185', 'Bacteroides_caccae_53434', 'Ruminococcus_bicirculans_59300', 'Parabacteroides_merdae_56972', 'Alistipes_onderdonkii_55464', 'Bacteroides_massiliensis_44749', 'Parabacteroides_distasonis_56985', 'Bacteroides_thetaiotaomicron_56941', 'Akkermansia_muciniphila_55290', 'Alistipes_shahii_62199', 'Ruminococcus_bromii_62047', 'Bacteroides_cellulosilyticus_58046', 'Barnesiella_intestinihominis_62208', 'Bacteroides_fragilis_54507', 'Dialister_invisus_61905', 'Escherichia_coli_58110', 'Oscillibacter_sp_60799', 'Alistipes_finegoldii_56071', 'Faecalibacterium_prausnitzii_57453', 'Bacteroides_plebeius_61623', 'Faecalibacterium_cf_62236', 'Odoribacter_splanchnicus_62174', 'Phascolarctobacterium_sp_59817', 'Faecalibacterium_prausnitzii_62201', 'Bacteroides_coprocola_61586', 'Bacteroides_finegoldii_57739', 'Eubacterium_eligens_61678', 'Bacteroides_eggerthii_54457', 'Bacteroidales_bacterium_58650', 'Butyrivibrio_crossotus_61674', 'Bifidobacterium_adolescentis_56815', 'Faecalibacterium_prausnitzii_61481', 'Alistipes_sp_60764', 'Bacteroides_faecis_58503', 'Blautia_wexlerae_56130', 'Eubacterium_siraeum_57634', 'Lachnospiraceae_bacterium_51870', 'Roseburia_inulinivorans_61943', 'Prevotella_stercorea_58308', 'Bifidobacterium_longum_57796', 'Anaerostipes_hadrus_55206', 'Bacteroides_intestinalis_61596', 'Oscillospiraceae_bacterium_54867', 'Roseburia_intestinalis_56239', 'Subdoligranulum_sp_62068', 'Sutterella_wadsworthensis_56828', 'Burkholderiales_bacterium_56577', 'Paraprevotella_clara_33712', 'Coprococcus_sp_62244', 'Alistipes_sp_59510', 'Guyana_massiliensis_60772', 'Roseburia_hominis_61877', 'Ruminococcus_torques_62045', 'Dorea_longicatena_61473', 'Bacteroides_sartorii_54642', 'Eubacterium_hallii_61477', 'Sutterella_wadsworthensis_62218', 'Coprococcus_comes_61587', 'Collinsella_sp_62205', 'Butyricimonas_virosa_58742', 'Megamonas_hypermegale_57114', 'Clostridium_sp_61482', 'Clostridiales_bacterium_56470', 'Alistipes_indistinctus_62207', 'Ruminococcus_gnavus_57638', 'Ruminococcus_sp_55468', 'Parabacteroides_johnsonii_55217', 'Eubacterium_ventriosum_61474', 'Bacteroides_salyersiae_54873', 'Bacteroides_coprophilus_61767', 'Bacteroides_clarus_62282', 'Ruminococcus_obeum_61472', 'Acidaminococcus_intestini_54097', 'Coprobacter_fastidiosus_56550', 'Alistipes_senegalensis_58364', 'Bacteroides_fragilis_56548', 'Bifidobacterium_bifidum_55065', 'Klebsiella_pneumoniae_54788', 'Clostridium_nexile_61654', 'Paraprevotella_xylaniphila_62280', 'Odoribacter_laneus_62216', 'Coprococcus_eutactus_61480', 'Phascolarctobacterium_succinatutens_61948', 'Streptococcus_salivarius_58037']

def plot_heatmap(species_list, suffix = "", condition_string = "0", zoom = False):
    data = {} # dictionary
    xlabels = []
    good_species_list = []
    sample_sizes = []

    if zoom:
        species_list = species_list[:20]
    pretty_species_list = [figure_utils.get_pretty_species_name(species, include_number=False) for species in species_list]


    for i in range(len(species_list)):
        current_species = figure_utils.get_pretty_species_name(species_list[i], include_number=False)
        data[current_species] = numpy.empty(len(species_list))
        for j in range(len(species_list)):
            species_name = species_list[i] + '_' + species_list[j] + suffix
            ld_map = calculate_linkage_disequilibria.load_ld_map(species_name)
            if len(ld_map.keys()) == 0: # file does not exist (might not have had enough hosts)
                data[current_species][j] = numpy.nan
                # sample_sizes.append(0)
            else:
                rN = ld_map[('all', '1D')][3] / ld_map[('all', '1D')][4] # genome-wide num / genome-wide denom
                rS = ld_map[('all', '4D')][3] / ld_map[('all', '4D')][4]
                data[current_species][j] = rN / rS

    df = pd.DataFrame.from_dict(data=data, orient='index', columns=pretty_species_list)
    mask = df.isnull()
    # print(mask)

    # Create a figure instance
    if zoom:
        fig = plt.figure(1, figsize=(7, 6))
    else:
        fig = plt.figure(1, figsize=(28, 24))

    # Create an axes instance
    ax = fig.add_subplot(111)

    # Create the plot
    plt.title("Heatmap of Interspecies LD (rN/rS)", fontsize=14)

    cmap = sns.color_palette("Spectral", as_cmap=True)
    cmap.set_bad(".7")
    # sns.heatmap(df, vmin=-0.3, vmax=0.3, cmap=cmap)
    sns.heatmap(df, vmin=0.3, vmax=1.7, cmap=cmap)

    # Save the figure
    if zoom:
        output_path = '/u/home/h/helenhua/project-ngarud/analysis/Heatmap/heatmap' + suffix + '_zoomed.png'
    else:
        output_path = '/u/home/h/helenhua/project-ngarud/analysis/Heatmap/heatmap' + suffix + '.png'
    fig.savefig(output_path, bbox_inches='tight')
    return 1

if __name__=='__main__':


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--zoom", help="Loads only frist 21 species", action="store_true")
    parser.add_argument("--suffix", help="Suffix of species name to separate conditions", default="")
    parser.add_argument("--condition", help="Allele frequency condition", default="0")

    args = parser.parse_args()

    zoom = args.zoom
    suffix_string = args.suffix
    condition_string = args.condition

    plot_heatmap(species_list, suffix_string, condition_string, zoom)
    # all_pretty_plot(species_list, "Prevotella_copri_61740", "_condition_0.05")