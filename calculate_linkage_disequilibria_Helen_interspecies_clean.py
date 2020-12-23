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
import matplotlib.pyplot as plt

import diversity_utils_Helen as diversity_utils
from math import log10,ceil,fabs
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

def load_ld_map(species_name):

    ld_map = {}

    intermediate_filename = intermediate_filename_template % (ld_directory, species_name)

    if not os.path.isfile(intermediate_filename):
        return ld_map

    file = gzip.open(intermediate_filename,"rt")
    header_line = file.readline() # header
    header_items = header_line.split(",")
    
    for line in file:
        items = line.split(",")
        if items[0].strip()!=species_name:
            continue
        
        clade_type = items[1].strip()
        variant_type = items[2].strip()

        genome_wide_rsquared_numerator = float(items[3].strip())
        genome_wide_rsquared_denominator = float(items[4].strip())
        sample_size = int(items[5].strip())
        
        rsquared_numerators = []
        rsquared_denominators = []
        lds = []
        counts = []
        for item in items[6:]: # added genome wide num and deno so start from 5
            subitems = item.split(":")
            rsquared_numerators.append(float(subitems[0]))
            rsquared_denominators.append(float(subitems[1]))

        rsquared_numerators = numpy.array(rsquared_numerators)
        rsquared_denominators = numpy.array(rsquared_denominators)        
        
        lds = rsquared_numerators/rsquared_denominators
        
        intragene_rsquared_numerators = rsquared_numerators
        intragene_rsquared_denominators = rsquared_denominators
        
        # ld_map[(clade_type, variant_type)] = (intragene_rsquared_numerators, intragene_rsquared_denominators, lds, genome_wide_rsquared_numerator, genome_wide_rsquared_denominator, sample_size)
        ld_map[(clade_type, variant_type)] = (intragene_rsquared_numerators, intragene_rsquared_denominators, lds, genome_wide_rsquared_numerator, genome_wide_rsquared_denominator)

    return ld_map


if __name__=='__main__':


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", help="Loads only a subset of SNPs for speed", action="store_true")
    parser.add_argument("--chunk-size", type=int, help="max number of records to load", default=1000000000)
    parser.add_argument("--species1", help="Name of specific species to run code on", default="all")
    parser.add_argument("--species2", help="Name of specific species to run code on", default="all")
    parser.add_argument("--low", help="Allele frequency condition lower bound", default=0.0)
    parser.add_argument("--high", help="Allele frequency condition upper bound", default=1.0)
    parser.add_argument("--suffix", help="Suffix of species name to separate conditions", default="")

    args = parser.parse_args()

    debug = args.debug
    chunk_size = args.chunk_size
    species1 = args.species1
    species2 = args.species2

    low = float(args.low)
    high = float(args.high)
    suffix = args.suffix

    # Load subject and sample metadata
    sys.stderr.write("Loading sample metadata...\n")
    subject_sample_map = sample_utils.parse_subject_sample_map()
    sys.stderr.write("Done!\n")
    
    pre_good_species_list = parse_midas_data.parse_good_species_list()
    if species1!='all' and species2!='all':
        good_species_list = [[species1, species2]]
    elif species1=='all' and species2!='all':
        good_species_list = []
        for sp1 in pre_good_species_list:
            good_species_list.append([sp1, species2])
    elif species1!='all' and species2=='all':
        good_species_list = []
        for sp2 in pre_good_species_list:
            good_species_list.append([species1, sp2])
    else: # both 'all'
        good_species_list = []
        for sp1 in pre_good_species_list:
            for sp2 in pre_good_species_list:
                good_species_list.append([sp1, sp2])
    
    if debug and len(good_species_list)>3.5:
        good_species_list = good_species_list[:3]

    if debug:
        bootstrapping_bin_size = 10
        bootstrapping_replicates =10
        
    # header of the output file.
    record_strs = [", ".join(['Species', 'CladeType', 'VariantType', 'GenomeWideNumerator', 'GenomeWideDenominator'])]
    
    os.system('mkdir -p %s' % ld_directory)

    # good_species_list is now a list of pairs of species
    for species_pair in good_species_list:
        species_name_1 = species_pair[0]
        species_name_2 = species_pair[1]

        sys.stderr.write("Loading haploid samples...\n")
        # Only plot samples above a certain depth threshold that are "haploids"
        snp_samples_1 = diversity_utils.calculate_haploid_samples(species_name_1, debug=debug)
        snp_samples_2 = diversity_utils.calculate_haploid_samples(species_name_2, debug=debug)

        # intersect samples from the two species
        snp_samples_12 = numpy.intersect1d(snp_samples_1,snp_samples_2)
        # print(len(snp_samples_12))
    
        if len(snp_samples_12) < min_sample_size:
            sys.stderr.write("Not enough haploid samples!\n")
            continue
        else:
            sys.stderr.write("Found %d haploid samples!\n" % len(snp_samples_12))
        
        sys.stderr.write("Calculating unique hosts...\n")
        # Only consider one sample per person
        snp_samples_12 = snp_samples_12[sample_utils.calculate_unique_samples(subject_sample_map, sample_list=snp_samples_12)]

        if len(snp_samples_12) < min_sample_size:
            sys.stderr.write("Not enough hosts!\n")
            continue
        else:
            sys.stderr.write("Found %d unique hosts!\n" % len(snp_samples_12)) 

        # Load divergence matrices (deprecated)
        
        # Analyze SNPs, looping over chunk sizes. 
        # Clunky, but necessary to limit memory usage on cluster

        sys.stderr.write("Loading core genes...\n")
        core_genes_1 = parse_midas_data.load_core_genes(species_name_1)
        sys.stderr.write("Done! Core genome of species 1 consists of %d genes\n" % len(core_genes_1))
        core_genes_2 = parse_midas_data.load_core_genes(species_name_2)
        sys.stderr.write("Done! Core genome of species 2 consists of %d genes\n" % len(core_genes_2))

        # Load SNP information for species_name
        # sys.stderr.write("Loading SNPs for %s...\n" % species_name)
        # sys.stderr.write("(core genes only...)\n")
        
        # clade_types = ['all','largest_clade']
        clade_types = ['all'] # ignore largest clade for now !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        variant_types = ['4D','1D'] 
        
        binned_rsquared_numerators = {}
        binned_rsquared_denominators = {}
        binned_counts = {}

        indiv_rsquared_numerators = {}
        indiv_rsquared_denominators = {}
        indiv_counts = {}
            
        final_line_number = 0
        while final_line_number >= 0:
    
            sys.stderr.write("Loading chunk starting @ %d...\n" % final_line_number)
            snp_samples_12, allele_counts_map_1, passed_sites_map_1, final_line_number = parse_midas_data.parse_snps(species_name_1, debug=debug,         allowed_variant_types=allowed_variant_types, allowed_samples=snp_samples_12,allowed_genes=core_genes_1, chunk_size=chunk_size,initial_line_number=final_line_number)
            sys.stderr.write("Done! Loaded %d genes for species 1\n" % len(allele_counts_map_1.keys()))
            snp_samples_12, allele_counts_map_2, passed_sites_map_2, final_line_number = parse_midas_data.parse_snps(species_name_2, debug=debug,         allowed_variant_types=allowed_variant_types, allowed_samples=snp_samples_12,allowed_genes=core_genes_2, chunk_size=chunk_size,initial_line_number=final_line_number)
            sys.stderr.write("Done! Loaded %d genes for species 2\n" % len(allele_counts_map_2.keys()))

            # neighboring and control genes (deprecated)
            
            for clade_type in clade_types:
                for variant_type in variant_types:
                    binned_rsquared_numerators[(clade_type,variant_type)] = {}
                    binned_rsquared_denominators[(clade_type,variant_type)] = {}
                    binned_counts[(clade_type,variant_type)] = {}

                    for gene_name_1 in allele_counts_map_1.keys(): # loop over genes of first species
                        if gene_name_1 not in core_genes_1:
                            continue
                        for gene_name_2 in allele_counts_map_2.keys(): # loop over genes of second species
                            if gene_name_2 not in core_genes_2:
                                continue
                            binned_rsquared_numerators[(clade_type,variant_type)][(gene_name_1, gene_name_2)] = 0
                            binned_rsquared_denominators[(clade_type,variant_type)][(gene_name_1, gene_name_2)] = 0
                            binned_counts[(clade_type,variant_type)][(gene_name_1, gene_name_2)] = 0
    
            sys.stderr.write("Calculating LD...\n")
            for clade_type in clade_types:
    
                for variant_type in variant_types:
            
                    for gene_name_1 in allele_counts_map_1.keys(): # loop over genes of first species
            
                        if gene_name_1 not in core_genes_1:
                            continue
        
                        locations = numpy.array([location for chromosome, location in allele_counts_map_1[gene_name_1][variant_type]['locations']])*1.0
                        allele_counts_1 = allele_counts_map_1[gene_name_1][variant_type]['alleles']

                        for gene_name_2 in allele_counts_map_2.keys(): # loop over genes of second species
                            if gene_name_2 not in core_genes_2:
                                continue
                            
                            allele_counts_2 = allele_counts_map_2[gene_name_2][variant_type]['alleles']
        
        
                            if len(allele_counts_1)==0 or len(allele_counts_2)==0:
                                # no diversity to look at!
                                continue
            
                            target_chromosome = allele_counts_map_1[gene_name_1][variant_type]['locations'][0][0]

                            # rsquared_numerators, rsquared_denominators = diversity_utils.calculate_unbiased_sigmasquared(allele_counts_1, allele_counts_2)
                            rsquared_numerators, rsquared_denominators = diversity_utils.calculate_sigmasquared(allele_counts_1, allele_counts_2, low, high)
                            # ^^^changed unbiased_sigmasquared to signmasquared because of negative value problem
                            
                            # get the indices of the upper diagonal of the distance matrix
                            # numpy triu_indices returns upper diagnonal including diagonal
                            # the 1 inside the function excludes diagonal. Diagnonal has distance of zero.
                            desired_idxs = numpy.triu_indices(rsquared_numerators.shape[0],1,rsquared_numerators.shape[1])
                        
                            # fetch the rsquared vals corresponding to the upper diagonal. 
                            rsquared_numerators = rsquared_numerators[desired_idxs]
                            rsquared_denominators = rsquared_denominators[desired_idxs]
            
                            # fetch entries where denominator != 0 (remember, denominator=pa*(1-pa)*pb*(1-pb). If zero, then at least one site is invariant)
                            rsquared_numerators = rsquared_numerators[rsquared_denominators>1e-09] 
                            rsquared_denominators = rsquared_denominators[rsquared_denominators>1e-09]
            
                            if len(rsquared_numerators) == 0:
                                continue
                
                            for i in xrange(0,len(rsquared_numerators)-1):
                        
                                binned_counts[(clade_type,variant_type)][(gene_name_1, gene_name_2)] += 1
                                binned_rsquared_numerators[(clade_type,variant_type)][(gene_name_1, gene_name_2)] += rsquared_numerators[i]
                                binned_rsquared_denominators[(clade_type,variant_type)][(gene_name_1, gene_name_2)] += rsquared_denominators[i]

                                # if (clade_type,variant_type) not in indiv_counts:
                                #     indiv_counts[(clade_type,variant_type)] = 1
                                # else:
                                #     indiv_counts[(clade_type,variant_type)] += 1

                                # if (clade_type,variant_type) not in indiv_rsquared_numerators:
                                #     indiv_rsquared_numerators[(clade_type,variant_type)] = [rsquared_numerators[i]]
                                # else:
                                #     indiv_rsquared_numerators[(clade_type,variant_type)].append(rsquared_numerators[i])

                                # if (clade_type,variant_type) not in indiv_rsquared_denominators:
                                #     indiv_rsquared_denominators[(clade_type,variant_type)] = [rsquared_denominators[i]]
                                # else:
                                #     indiv_rsquared_denominators[(clade_type,variant_type)].append(rsquared_denominators[i])

        species_name = species_name_1 + '_' + species_name_2 + suffix

        bootstrapped_rsquared_numerators = {}
        bootstrapped_rsquared_denominators = {}
        bootstrapped_counts = {}
        genome_wide_rsquared_numerators = {}
        genome_wide_rsquared_denominators = {}

        for clade_type in clade_types:
            for variant_type in variant_types:
                binned_rsquared_numerators[(clade_type,variant_type)] = numpy.array(binned_rsquared_numerators[(clade_type,variant_type)].values())
                binned_rsquared_denominators[(clade_type,variant_type)] = numpy.array(binned_rsquared_denominators[(clade_type,variant_type)].values())
                binned_counts[(clade_type,variant_type)] = numpy.array(binned_counts[(clade_type,variant_type)].values())
                binned_rsquared_numerators[(clade_type,variant_type)] = binned_rsquared_numerators[(clade_type,variant_type)][binned_rsquared_denominators[(clade_type,variant_type)]>1e-09]
                binned_counts[(clade_type,variant_type)] = binned_counts[(clade_type,variant_type)][binned_rsquared_denominators[(clade_type,variant_type)]>1e-09]
                binned_rsquared_denominators[(clade_type,variant_type)] = binned_rsquared_denominators[(clade_type,variant_type)][binned_rsquared_denominators[(clade_type,variant_type)]>1e-09]
                # print(len(binned_rsquared_numerators[(clade_type,variant_type)]))

                # boostrapping 1000 gene pairs for 1000 times
                bootstrapped_rsquared_numerators[(clade_type,variant_type)] = []
                bootstrapped_rsquared_denominators[(clade_type,variant_type)] = []
                bootstrapped_counts[(clade_type,variant_type)] = []

                gene_pair_idx = range(len(binned_rsquared_numerators[(clade_type,variant_type)]))
                for i in range(bootstrapping_replicates):
                    bootstrapped_idx = random.sample(gene_pair_idx, bootstrapping_bin_size) # randomly subsample

                    bootstrapped_rsquared_numerators[(clade_type,variant_type)].append(sum(binned_rsquared_numerators[(clade_type,variant_type)][bootstrapped_idx]))
                    bootstrapped_rsquared_denominators[(clade_type,variant_type)].append(sum(binned_rsquared_denominators[(clade_type,variant_type)][bootstrapped_idx]))
                    bootstrapped_counts[(clade_type,variant_type)].append(sum(binned_counts[(clade_type,variant_type)][bootstrapped_idx]))

                genome_wide_rsquared_numerators[(clade_type,variant_type)] = sum(bootstrapped_rsquared_numerators[(clade_type,variant_type)])
                genome_wide_rsquared_denominators[(clade_type,variant_type)] = sum(bootstrapped_rsquared_denominators[(clade_type,variant_type)])

        for clade_type in clade_types:
            for variant_type in variant_types:
                binned_rsquared_strs = ["%g:%g:%d" % (rsquared_numerator, rsquared_denominator, count) for rsquared_numerator, rsquared_denominator, count in zip(bootstrapped_rsquared_numerators[(clade_type,variant_type)], bootstrapped_rsquared_denominators[(clade_type,variant_type)], bootstrapped_counts[(clade_type,variant_type)])]

                # indiv_rsquared_strs = ["%g:%g" % (rsquared_numerator, rsquared_denominator) for rsquared_numerator, rsquared_denominator in zip(indiv_rsquared_numerators[(clade_type,variant_type)], indiv_rsquared_denominators[(clade_type,variant_type)])[1:-1]]
            
                record_str = ", ".join([species_name, clade_type, variant_type, str(genome_wide_rsquared_numerators[(clade_type,variant_type)]), str(genome_wide_rsquared_denominators[(clade_type,variant_type)]), str(len(snp_samples_12))]+binned_rsquared_strs)
            
                record_strs.append(record_str)
            
        sys.stderr.write("Done with %s!\n" % species_name) 
    
         
        sys.stderr.write("Writing intermediate file...\n")
        intermediate_filename = intermediate_filename_template % (ld_directory, species_name)
        file = gzip.open(intermediate_filename,"w")
        record_str = "\n".join(record_strs)
        file.write(record_str)
        file.close()
        sys.stderr.write("Done!\n")

    sys.stderr.write("Done looping over species!\n")
    
    sys.stderr.write("Testing loading...\n")
    ld_map = load_ld_map(species_name)
    sys.stderr.write("Done!\n")

 
