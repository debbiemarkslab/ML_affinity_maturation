#!/home/as974/miniconda3/envs/torch_env/bin/python
from subprocess import check_output
import pandas as pd
import os
import sys
import argparse
import contextlib
from Bio.SeqIO.FastaIO import SimpleFastaParser
from Bio.Seq import translate, Seq
# from Bio.Alphabet import generic_dna, Gapped
# from Bio.Alphabet.IUPAC import ExtendedIUPACProtein
from Bio.Data.CodonTable import unambiguous_dna_by_name
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
from tqdm import tqdm

TMP_FOLDER = "amplicon_tmp"
GAP_CHARACTER = "-"
STOP_CHARACTER = "*"
UNKNOWN_AA = "X"
UNKNOWN_NT = "N"

gapped_codon_table = unambiguous_dna_by_name['Standard']
# gapped_codon_table.nucleotide_alphabet = Gapped(generic_dna)
# gapped_codon_table.protein_alphabet = Gapped(ExtendedIUPACProtein())
for key, value in gapped_codon_table.forward_table.copy().items():
    for i in range(len(key)):
        new_key = key[:i] + GAP_CHARACTER + key[i+1:]
        gapped_codon_table.forward_table[new_key] = UNKNOWN_AA
    for i in range(len(key)-1):
        new_key = key[:i] + GAP_CHARACTER * 2 + key[i + 2:]
        gapped_codon_table.forward_table[new_key] = UNKNOWN_AA
gapped_codon_table.forward_table[GAP_CHARACTER * 3] = GAP_CHARACTER


@contextlib.contextmanager
def smart_open(filename=None, mode='r'):
    if filename and isinstance(filename, str):
        fh = open(filename, mode)
    elif filename == 0:
        fh = sys.stdin
    elif filename == 1:
        fh = sys.stdout
    elif filename is None:
        fh = None
    else:
        fh = sys.stdin

    try:
        yield fh
    finally:
        if fh not in [sys.stdin, sys.stdout, sys.stderr, None]:
            fh.close()


def fill_gaps(fasta_file, output_file):
    with smart_open(fasta_file, 'r') as in_fa, open(output_file, 'w') as out_fa:
        for values in SimpleFastaParser(in_fa):
            out_fa.write(">{}\n{}\n".format(
                values[0],
                values[1].replace(GAP_CHARACTER, UNKNOWN_NT)
            ))


def find_start_end_with_motifs(fasta_file, cdr_motifs_file):
    """
    Requires MAST from the MEME suite.
    For MAST, the fasta file may not contain gap characters.
    """
    output = StringIO(check_output(["mast", cdr_motifs_file, fasta_file, "-norc", "-hit_list"]).decode())

    full_df = pd.read_table(output, sep='\s+', header=None, skiprows=2, index_col=None, dtype=str)
    full_df = full_df[:-1]

    motif_counts = full_df[[0, 1]].rename(columns={0: 'seq', 1: 'motif'})
    motif_counts = motif_counts.groupby(['seq', 'motif']).size().reset_index(name="count")
    motif_counts = motif_counts.pivot(index='seq', columns='motif').fillna(0).astype(int)
    motif_counts['keep'] = motif_counts.eq(1).all(1)
    motif_counts = motif_counts.reset_index()
    keep_seqs = motif_counts.loc[motif_counts['keep'], 'seq'].astype(str)

    full_df = full_df[full_df[0].isin(keep_seqs)]
    full_df = full_df.rename(columns={0: 'seq', 1: 'motif', 2: 'motif_name', 4: 'first', 5: 'last'})
    full_df = full_df[['seq', 'motif_name', 'first', 'last']]

    full_df = full_df.pivot(index='seq', columns='motif_name')
    full_df.columns = full_df.columns.swaplevel(0, 1)
    full_df = full_df.sort_index(axis=1, level=0)

    full_df.columns = ['_'.join(col).strip() for col in full_df.columns.values]
    full_df = full_df[['nb_before_last', 'nb_after_first']].astype(int)
    full_df['nb_after_first'] -= 1
    full_df = full_df.rename(columns={'nb_before_last': 'nb_start', 'nb_after_first': 'nb_end'})

    return full_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translates amplicon sequences to their protein sequence")
    parser.add_argument('-i', default=0, type=str, help="input fasta file")
    parser.add_argument('-o', default=1, type=str, help="output file")
    parser.add_argument('--untranslated', default=None, type=str, help="output untranslated sequences")
    parser.add_argument('--rf', default=None, type=int, help="Reading frame")
    parser.add_argument('--nb-start-dna', default=None, type=int,
                        help="first nanobody position in DNA sequence")
    parser.add_argument('--nb-end-dna', default=None, type=int,
                        help="last nanobody position in DNA sequence (negative if from end)")
    parser.add_argument('--nb-start-protein', default=None, type=int,
                        help="first nanobody position in protein sequence")
    parser.add_argument('--nb-end-protein', default=None, type=int,
                        help="last nanobody position in protein sequence (negative if from end)")
    parser.add_argument('--ba-motifs', default=None, type=str,
                        help="Motifs files for sequences before and after the protein sequence"
                             "(optional, overrides nb-start-dna and nb-end-dna)")
    parser.add_argument('--tmpdir', default="/tmp/", type=str,
                        help="Folder to store temporary files (if motif file provided)")
    pargs = parser.parse_args()

    tmp_folder = os.path.join(pargs.tmpdir, TMP_FOLDER)
    os.makedirs(tmp_folder, exist_ok=True)
    filled_fasta = os.path.join(tmp_folder, "filled.fasta")
    fill_gaps(pargs.i, filled_fasta)

    if pargs.ba_motifs:
        se_df = find_start_end_with_motifs(filled_fasta, pargs.ba_motifs)
        se_df = se_df.to_dict(orient='index')

    with smart_open(pargs.i, 'r') as fastas, smart_open(pargs.o, 'w') as out_file, \
            smart_open(pargs.untranslated, 'w') as out_untranslated:
        for fasta in tqdm(SimpleFastaParser(fastas), desc="Translating sequences"):
            if pargs.ba_motifs:
                start_end = se_df.get(fasta[0])
                if start_end is None:
                    continue
                nb_start = start_end['nb_start']
                nb_end = start_end['nb_end']
            else:
                nb_start = pargs.nb_start_dna
                nb_end = pargs.nb_end_dna

            seq = fasta[1][nb_start: nb_end]

            if out_untranslated is not None:
                out_untranslated.write(">{}\n{}\n".format(
                    fasta[0],
                    seq
                ))

            # move to correct reading frame
            if pargs.rf is not None:
                seq = seq[pargs.rf-1:]
            # trim the length if there is an an incomplete terminal codon
            if len(seq) % 3:
                seq = seq[:-(len(seq) % 3)]

#             seq = translate(seq, table=gapped_codon_table, to_stop=False, gap=GAP_CHARACTER)
            try: seq = str(Seq(seq).translate(to_stop = False, table=gapped_codon_table, gap=GAP_CHARACTER))
            except: continue
            seq = seq[:pargs.nb_end_protein]
            seq = seq.split(STOP_CHARACTER, maxsplit=1)[0]  # get only the sequence before the first stop codon
            seq = seq[pargs.nb_start_protein:]

            out_file.write(">{}\n{}\n".format(
                fasta[0],
                seq
            ))
