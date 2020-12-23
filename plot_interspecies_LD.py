import sample_utils
import config
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

import calculate_linkage_disequilibria_Helen_interspecies_clean as calculate_linkage_disequilibria
import gene_diversity_utils
import calculate_substitution_rates
import clade_utils
import stats_utils
import figure_utils
from math import log10,ceil,fabs, floor
from numpy.random import randint, choice

ld_directory = '%slinkage_disequilibria/' % (parse_midas_data.data_directory)
intermediate_filename_template = '%s%s.txt.gz'       

low_divergence_threshold = config.between_low_divergence_threshold
#low_divergence_threshold = 5e-04 # this was picked by looking at inflection point of dN/dS vs dS plot 
# min_sample_size = config.between_host_min_sample_size
min_sample_size = 10 # for testing purposes!!!! minimum is 4, but 10 is safer
min_ld_sample_size = config.between_host_ld_min_sample_size
allowed_variant_types = set(['1D','4D'])
bootstrapping_replicates = 1000
bootstrapping_bin_size = 1000
# max_species_display = 15

# species_list = ["Alistipes_putredinis_61533", "Bacteroides_uniformis_57318", "Bacteroides_vulgatus_57955", "Prevotella_copri_61740", "Eubacterium_rectale_56927", "Ruminococcus_bromii_62047"]
species_list = parse_midas_data.parse_good_species_list()

# rcParams['font.family'] = ['sans-serif']
# rcParams['font.sans-serif'] = ['Helvetica']
rcParams['font.size'] = 9
rcParams['lines.linewidth'] = 0.5
rcParams['legend.frameon']  = False
rcParams['legend.fontsize']  = 'large'

def reduce_significant_digit(a_number, significant_digits):
    if a_number == 0:
        return a_number
    rounded_number =  round(a_number, significant_digits - int(floor(log10(abs(a_number)))) - 1)
    return rounded_number

def all_pretty_plot(species_list, focal_species, suffix = "", condition_string = "0"):
    data = []
    xlabels = []
    good_species_list = []
    sample_sizes = []

    for i in range(len(species_list)):
        species_name = focal_species + '_' + species_list[i] + suffix
        ld_map = calculate_linkage_disequilibria.load_ld_map(species_name)
        if len(ld_map.keys()) == 0: # file does not exist (might not have had enough data)
            continue
        for variant_type in allowed_variant_types:
            data.append(ld_map[('all', variant_type)][2])
            sample_sizes.append(ld_map[('all', variant_type)][5])
        xlabels.append(figure_utils.get_pretty_species_name(species_list[i], include_number=False))
        xlabels.append("")
        good_species_list.append(species_list[i])
        # if len(good_species_list) > max_species_display:
        #     break

    if len(good_species_list) == 0:
        return 0
    # all_mwu_pval = {}
    # for i in range(len(good_species_list)):
    #     mwu_stat, mwu_pval = stats.mannwhitneyu(data[2*i], data[2*i+1])
    #     all_mwu_pval[good_species_list[i]] = reduce_significant_digit(mwu_pval, 4)
    #     xlabels[2*i+1] = reduce_significant_digit(mwu_pval, 4)
    all_t_pval = {}
    for i in range(len(good_species_list)):
        t_stat, t_pval = stats.ttest_ind(data[2*i], data[2*i+1])
        all_t_pval[good_species_list[i]] = reduce_significant_digit(t_pval, 4)
        xlabels[2*i+1] = reduce_significant_digit(t_pval, 4) + '\tn=' + sample_sizes[i]

    # Create a figure instance
    if len(good_species_list) > 4:
        fig_width = len(good_species_list)
    else:
        fig_width = 4
    fig = plt.figure(1, figsize=(fig_width, 6))

    # Create an axes instance
    ax = fig.add_subplot(111)

    # Format properties
    medianprops = dict(linestyle='-', linewidth=2.5, color='black')
    meanpointprops = dict(marker='D', markeredgecolor='#FFDE8C',
                      markerfacecolor='#FFDE8C', markersize=2)
    flierprops = dict(marker='o', markerfacecolor='#A3A3A3', markeredgecolor='#A3A3A3', markersize=2,
                  linestyle='none')

    # Create the boxplot
    bp = ax.boxplot(data, flierprops=flierprops, medianprops=medianprops, meanprops=meanpointprops, showmeans=True, patch_artist = True)
    ax.set_xticklabels(xlabels)
    ax.set_yscale("log")
    ax.set_ylabel('Linkage disequilibrium, $\sigma^2_d$')
    # ax.set_ylim([5e-3,1])
    plt.title("Interspecies LD between " + figure_utils.get_pretty_species_name(focal_species, include_number=False) + " and another species\n" + "Allele frequency < " + condition_string, fontsize=14)
    plt.xticks(rotation=90)

    # Fill with colors
    colors = ['#04DEDA', '#FF6B43'] * len(species_list) # alternating colors for synonymous vs. non-synonymous
    for patch, color in zip(bp['boxes'], colors): 
        patch.set_facecolor(color)

    # Add legend
    plt.legend((bp['boxes'][0], bp['boxes'][1]), ('Synonymous', 'Non-synonymous'))

    # Save the figure
    output_path = '/u/home/h/helenhua/project-ngarud/analysis/interspecies_LD' + suffix + '/all_' + focal_species + suffix + '.png'
    fig.savefig(output_path, bbox_inches='tight')
    return 1

if __name__=='__main__':


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", help="Loads only a subset of SNPs for speed", action="store_true")
    parser.add_argument("--focal", help="Name of focal species to run code on", default="Prevotella_copri_61740")
    parser.add_argument("--suffix", help="Suffix of species name to separate conditions", default="")
    parser.add_argument("--condition", help="Allele frequency condition", default="0")

    args = parser.parse_args()

    debug = args.debug
    focal_species = args.focal
    suffix_string = args.suffix
    condition_string = args.condition

    all_pretty_plot(species_list, focal_species, suffix_string, condition_string)
    # all_pretty_plot(species_list, "Prevotella_copri_61740", "_condition_0.05")