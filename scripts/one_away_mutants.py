import os, sys
sys.path.append('/n/groups/marks/projects/binding_affinity')
import affinity_maturation_utilities as util
from modelling_functions import *
from argparse import ArgumentParser
import joblib

def main():
    
    parser = ArgumentParser()
    parser.add_argument('--wt_name', type=str,
                       help = 'the name of the sequence')
    parser.add_argument('--wa_fnm', type=str,
                       help='the filename for the wt sequence alignment')
    parser.add_argument('--with_WT', action='store_true',
                       help = 'whether to include the WT sequence or not')
    parser.add_argument('--fasta_outfnm', type=str,
                        help='the filename for the fasta output')
    args = parser.parse_args()

    mut_seqs, wt_seq, imgt_col = util.make_singles_muts(args.wa_fnm, args.wt_name,
                                                       args.with_WT)

    #save the seqs as a fasta
    if not os.path.isdir(os.path.dirname(args.fasta_outfnm)):
        os.makedirs(os.path.dirname(args.fasta_outfnm))
    with open(args.fasta_outfnm, 'w') as f:
        for l,s in mut_seqs.iteritems():
            f.write(f'>{l}\n')
            f.write(f'{s}\n')
            
            
if __name__ == '__main__':
    main()