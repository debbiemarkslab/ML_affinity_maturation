import os, sys, re
sys.path.append('/n/groups/marks/projects/binding_affinity')
import affinity_maturation_utilities as util
from modelling_functions import *
from argparse import ArgumentParser
import joblib
import seaborn as sns
import torch
from matplotlib.patches import Rectangle

sys.path.append('/n/groups/marks/projects/binding_affinity/model_zoo')
from dataset_classes import OneHotArrayDataset
from supervised_model_classes import CNN, MLP, LSTM

from sklearn.metrics import roc_curve, auc, precision_recall_curve, r2_score
from scipy.stats import spearmanr
from scipy.stats import spearmanr, rankdata

def get_latest_ckpt(directory):
    ckpt_nms = os.listdir(directory)
    ckpt_nms = [c for c in ckpt_nms if re.search('.tar', c)]
    ckpt_epochs = pd.Series(ckpt_nms).str.extract('\w+_epoch(\d+)').iloc[:,0].values.astype(int)
    best_epoch_idx = np.argmax(ckpt_epochs)
    return (ckpt_nms[best_epoch_idx])

def get_sets(data, Xcol, Ycol, split):
    R = '_'.join(Ycol.split('_')[:-1])
    if split == 'any':
        ids = data.Y_df.index[~data.Y_df[f'{R}_split'].isna()]
    else:
        ids = data.Y_df.index[data.Y_df[f'{R}_split'] == split]
    Y = data.Y_df.loc[ids, Ycol]
    X = data.X_df.loc[ids, Xcol]
    df = pd.concat([X,Y], axis=1)
    return (df)

all_models = [
        ('median_MACS_dist', ''),
        ('WT_dist', ''),
       ('logistic_regression', 'FACS1_MACS'),
       ('linear_regression', 'FACS1_MACS'),
       ('logistic_regression', 'FACS2_MACS'),
       ('linear_regression', 'FACS2_MACS'),
        ('ESM2', ''),
       ('ESM2_MLP_classification', 'FACS1_MACS'),
       ('CNN_classification', 'FACS1_MACS'),
       ('CNN_regression', 'FACS1_MACS'),
       ('MLP_classification', 'FACS1_MACS'),
       ('MLP_regression', 'FACS1_MACS')
]

c_keys = {'logistic_regression':'#74c4f4', #blue
          'linear_regression':'#74c4f4',#blue
          'MLP_classification': '#521c87', #dark purple
          'MLP_regression': '#521c87', #dark purple
        'CNN_classification':'#9760ce', #purple
          'CNN_regression':'#9760ce', #purple
          'ESM2': '#bfa852', #yellow
          'ESM2_MLP_classification': '#ce606c', #red
          'ESM2_MLP_regression': '#ce606c', #red
          'WT_dist': '#989ca5', #light grey
        'median_MACS_dist':'#51555a' #dark grey
       }

model_keys = {
        'median_MACS_dist': 'Median Hamming to MACS',
        'WT_dist': 'Hamming to WT',
       'logistic_regression': 'Logistic Regression',
       'linear_regression': 'Linear Regression',
        'ESM2': 'ESM2',
       'ESM2_MLP_classification': 'ESM2_MLP',
        'ESM2_MLP_regression': 'ESM2_MLP',
       'CNN_classification': 'CNN',
       'CNN_regression': 'CNN',
       'MLP_classification': 'MLP',
       'MLP_regression': 'MLP'
}

def get_specs(fil=5, fil_abund=1):
    specs = {
    'NbB7': {'proj': 'NbB7',
             'fil':fil,
             'fil_abund': fil_abund,
             'data_dir': f'/n/groups/marks/projects/nanobodies/kruse_nanobody_sequencing/B2AR_affinity/campaign_A_v3/fil{fil}_filabund{fil_abund}',
             'data_sub': 'all',
             'round_labs': ['naive', 'MACS', 'FACS1', 'FACS2', 'FACS3']
            },
    'AT110': {'proj': 'AT110',
             'fil':fil,
             'fil_abund': fil_abund,
             'data_dir': f'/n/groups/marks/projects/nanobodies/kruse_nanobody_sequencing/AT110/AT110/fil{fil}_filabund{fil_abund}',
             'data_sub':  'all',
             'round_labs': ['MACS', 'FACS1', 'FACS2']
            },
    'LysoNbV1': {'proj': 'LysoNbV1',
             'fil':fil,
             'fil_abund': fil_abund,
             'data_dir': f'/n/groups/marks/projects/nanobodies/kruse_nanobody_sequencing/TMEM192_affinity/V1/fil{fil}_filabund{fil_abund}',
             'data_sub': 'all',
             'round_labs':  ['MACS', 'FACS1']
            },
    'LysoNbV1_V2': {'proj': 'LysoNbV1_V2',
             'fil':fil,
             'fil_abund': fil_abund,
             'data_dir': f'/n/groups/marks/projects/nanobodies/kruse_nanobody_sequencing/TMEM192_affinity/V2/fil{fil}_filabund{fil_abund}',
             'data_sub': 'all',
             'round_labs':  ['MACS', 'FACS1', 'FACS2-8nM', 'FACS2-20nM', 'FACS2-poly']
            },
    'LysoNbV1_V3': {'proj': 'LysoNbV1_V3',
             'fil':fil,
             'fil_abund': fil_abund,
             'data_dir': f'/n/groups/marks/projects/nanobodies/kruse_nanobody_sequencing/TMEM192_affinity/V3/fil{fil}_filabund{fil_abund}',
             'data_sub': 'all',
             'round_labs':  ['MACS', 'FACS1', 'FACS2-8nM', 'FACS2-20nM', 'FACS2-poly']
            },
    'RX002': {'proj': 'RX002',
             'fil':fil,
             'fil_abund': fil_abund,
             'data_dir': f'/n/groups/marks/projects/nanobodies/kruse_nanobody_sequencing/RXFP1_affinity/V2/fil{fil}_filabund{fil_abund}',
             'data_sub': 'all',
             'round_labs':  ['MACS', 'FACS1', 'FACS2']
        }
    }
    return (specs)
specs = get_specs()


#get arguments
parser = get_training_parser()
args = parser.parse_args('')

def update_args(proj, args):
    if proj == 'NbB7':

        args.sub_exp = 'campaign_A/compmodels'
        #specify to B2AR
        args.exp_dir = '/n/groups/marks/projects/binding_affinity/B2AR_v2'
        args.data_dir = '/n/groups/marks/projects/nanobodies/kruse_nanobody_sequencing/B2AR_affinity/campaign_A_v3/fil5_filabund1/'
        args.no_wt_cols = True
        args.train_Y_col = 'FACS1_MACS'
        args.eval_Y_col = 'FACS2_MACS'
        args.test_mut_fnm = '/n/groups/marks/projects/binding_affinity/B2AR_v2/accessory_files/NbB7_tested_mutations.csv'
        args.sin_mut_fnm = '/n/groups/marks/projects/binding_affinity/B2AR_v2/accessory_files/NbB7_single_mutants_withWT.fasta'

    elif proj == 'AT110':

        #specify to AT110
        args.sub_exp = 'AT110/compmodels'
        args.exp_dir = '/n/groups/marks/projects/binding_affinity/AT110'
        args.data_dir = '/n/groups/marks/projects/nanobodies/kruse_nanobody_sequencing/AT110/AT110/fil5_filabund1/'
        args.no_wt_cols = True
        args.train_Y_col = 'FACS1_MACS'
        args.eval_Y_col = 'FACS2_MACS'
        args.test_mut_fnm = '/n/groups/marks/projects/binding_affinity/AT110/accessory_files/AT110i1_mutations.csv'
        args.sin_mut_fnm = '/n/groups/marks/projects/binding_affinity/AT110/accessory_files/AT110_singles_withWT.fasta'
        args.contact_det_fnm = '/n/groups/marks/databases/sabdab/sabdab_dataset/6do1/H(C)_ant(A)_contact_details.csv'


    elif proj == 'LysoNbV1':

        #specify to AT110
        args.sub_exp = 'V1/compmodels'
        args.exp_dir = '/n/groups/marks/projects/binding_affinity/TMEM192'
        args.data_dir = specs[proj]['data_dir']
        args.no_wt_cols = True
        args.train_Y_col = 'FACS1_MACS'
        args.eval_Y_col = 'FACS1_MACS'
        args.test_mut_fnm = '/n/groups/marks/projects/binding_affinity/TMEM192/accessory_files/LysoNbV1_tested_mutations.csv'
        args.sin_mut_fnm = '/n/groups/marks/projects/binding_affinity/TMEM192/accessory_files/LysoNbV1_single_mutants_withWT.fasta'
        
    elif proj == 'LysoNbV1_V2':

        #specify to AT110
        args.sub_exp = 'V2/compmodels'
        args.exp_dir = '/n/groups/marks/projects/binding_affinity/TMEM192'
        args.data_dir = specs[proj]['data_dir']
        args.no_wt_cols = True
        args.train_Y_col = 'FACS1_MACS'
        args.eval_Y_col = 'FACS2-8nM_MACS'
        args.test_mut_fnm = '/n/groups/marks/projects/binding_affinity/TMEM192/accessory_files/LysoNbV1_tested_mutations.csv'
        args.sin_mut_fnm = '/n/groups/marks/projects/binding_affinity/TMEM192/accessory_files/LysoNbV1_single_mutants_withWT.fasta'
    
    elif proj == 'LysoNbV1_V3':

        #specify to AT110
        args.sub_exp = 'V3/compmodels'
        args.exp_dir = '/n/groups/marks/projects/binding_affinity/TMEM192'
        args.data_dir = specs[proj]['data_dir']
        args.no_wt_cols = True
        args.train_Y_col = 'FACS1_MACS'
        args.eval_Y_col = 'FACS2-8nM_MACS'
        args.test_mut_fnm = '/n/groups/marks/projects/binding_affinity/TMEM192/accessory_files/LysoNbV1_tested_mutations.csv'
        args.sin_mut_fnm = '/n/groups/marks/projects/binding_affinity/TMEM192/accessory_files/LysoNbV1_single_mutants_withWT.fasta'

    elif proj == 'RX002':

        args.sub_exp = 'V2/compmodels'
        args.exp_dir = '/n/groups/marks/projects/binding_affinity/RXFP1'
        args.data_dir = specs[proj]['data_dir']
        args.no_wt_cols = True
        args.train_Y_col = 'FACS1_MACS'
        args.eval_Y_col = 'FACS1_MACS'
        args.test_mut_fnm = '/n/groups/marks/projects/binding_affinity/RXFP1/accessory_files/RX002_tested_mutations.csv'
        args.sin_mut_fnm = '/n/groups/marks/projects/binding_affinity/RXFP1/accessory_files/RX002_singles_withWT.fasta'

    return (args)


def import_singles_data_and_scores(proj, args):
    
    ##### IMPORT DATA
    #import the single mutants
    test_res = util.import_fasta(args.sin_mut_fnm).reset_index().rename({'index':'mutname'}, axis=1)

    #extract any metadata
    test_res = util.unpack_mutname(test_res, 'mutname', fmt='NAME_WIM_(I)')
    test_res['IMGT_AA'] = test_res['IMGT'] + '_' + test_res['mut_AA']

    #annotate the tested mutants
    if os.path.isfile(args.test_mut_fnm):
        test_mut_df = pd.read_csv(args.test_mut_fnm)
        test_res['ii_AA'] = test_res['seqidx'] + '_' + test_res['mut_AA']
        test_mut_df['ii_AA'] = test_mut_df['idx'].astype(str) + '_' + test_mut_df['Mut']
        test_res['tested'] = test_res['ii_AA'].isin(test_mut_df['ii_AA'])
        mut_annot_df = test_res.loc[test_res['tested'] == True, ['mut_AA', 'seqidx', 'IMGT']]

    #other
    seqlen = len(test_res['seq'].iloc[0])
    exp_dir = args.exp_dir

    #import the WT alignment
    wt_afnm = os.path.join(args.exp_dir, 'aligned_WT_sequence.csv')
    wt_adf = pd.read_csv(wt_afnm)
    imgt_cols = wt_adf.columns[wt_adf.columns.str.contains('\d+')].tolist()
    imgt_cols = pd.Series(imgt_cols)[(wt_adf.loc[:,imgt_cols]).values[0] != '-']


    test_res['full_mut_name'] = pd.Series(test_res['mutname']).str.replace('\/', '_').values

    ### Sklearn models

    #set model_fnms
    for Y_col in ['FACS1_MACS', 'FACS2_MACS', 'FACS3_MACS', 'FACS2-8nM_MACS', 'FACS2-20nM_MACS', 'FACS2-poly_MACS']: 
        model_fnms = {
            'logistic_regression' : os.path.join(args.exp_dir, args.sub_exp, 'logistic',
                                                 f'Train_{Y_col}_enriched',
                                                 f'Train_{Y_col}_enriched.sav') ,
            'linear_regression' : os.path.join(args.exp_dir, args.sub_exp, 'linear_W5e-4',
                                                 f'Train_{Y_col}_logratio',
                                                 f'Train_{Y_col}_logratio.sav')
        }

        #get scores
        for nm,fnm in model_fnms.items():
            if os.path.isfile(fnm):
                #load model
                load_model = joblib.load(fnm)
                #get coefs
                c_df = load_model.imgt_aa_col.set_index('IMGT_AA')
                c_df['beta'] = np.squeeze(load_model.model.coef_)
                #add a WT score here - the first WT AA is Q
                #the WT score is just 0 as here the score is
                #without the intercept considered
                c_df.loc['1_Q', :] = ['Q', '1', 0]
                #add to holder
                test_res.loc[:, f'{nm}_{Y_col}_scores'] = c_df.reindex(test_res['IMGT_AA'], axis='index')['beta'].values


    ### Zero shot LLMs

    model_fnms = {
        'ESM2' : f'/n/groups/marks/projects/binding_affinity/model_zoo/unsupervised_ESM2/{proj}/{proj}_singles_ESM2_scores.csv'
    }

    #get scores
    for nm, fnm in model_fnms.items():
        if os.path.isfile(fnm):
            #get scores
            sdf = pd.read_csv(fnm).set_index('full_mut_name')
            #add to df
            test_res.loc[:, f'{nm}_scores'] = sdf.reindex(test_res['full_mut_name'])['esm2_t33_650M_UR50D'].values
            #center around WT score
            test_res

    ### Semi-supervised models

    model_fnms = {
        'ESM2_MLP_classification' : f'/n/groups/marks/projects/binding_affinity/model_zoo/semi_supervised_ESM2/{proj}/F1_M_seqs_ckpts/MLP_topmodel_enriched/singles_scores.csv',
        'ESM2_MLP_regression' : f'/n/groups/marks/projects/binding_affinity/model_zoo/semi_supervised_ESM2/{proj}/F1_M_seqs_ckpts/MLP_topmodel_logratio/singles_scores.csv'
    }

    #get scores
    for nm, fnm in model_fnms.items():
        if os.path.isfile(fnm):
            #get scores
            sdf = pd.read_csv(fnm).set_index('seq_ID')
            #add to df
            test_res.loc[:, f'{nm}_FACS1_MACS_scores'] = sdf.reindex(test_res['full_mut_name'])['score'].values

    ### Neural Networks

    #set filenames
    for Y_col in ['FACS1_MACS']:
        model_fnms_pre = {
            'CNN_classification' : os.path.join(args.exp_dir, args.sub_exp, 'CNN',
                                        f'Train_{Y_col}_enriched', 'ckpt'),
            'CNN_regression' : os.path.join(args.exp_dir, args.sub_exp, 'CNN',
                                        f'Train_{Y_col}_logratio', 'ckpt'),
            'MLP_classification' : os.path.join(args.exp_dir, args.sub_exp, 'MLP',
                                        f'Train_{Y_col}_enriched', 'ckpt'),
            'MLP_regression' : os.path.join(args.exp_dir, args.sub_exp, 'MLP',
                                        f'Train_{Y_col}_logratio', 'ckpt')
        }
        
        model_fnms = {}
        for nm,d in model_fnms_pre.items():
            if os.path.isdir(d):
                model_fnms[nm] = os.path.join(d, get_latest_ckpt(d))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #get scores from models
        for nm, fnm in model_fnms.items():
            if os.path.isfile(fnm):
                #load model architecture
                model_type = nm.split('_')[0]
                if model_type == 'CNN':
                    load_model = CNN(seqlen)
                elif model_type == 'MLP':
                    load_model = MLP(seqlen*20)

                #load parameters
                load_model.load_state_dict(torch.load(fnm))

                #score sequences
                dataset = OneHotArrayDataset(test_res,'seq','seqidx')
                loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = 128, shuffle = False)
                scores = []
                for seq_array, labels in loader:
                    inputs = torch.reshape(seq_array,(seq_array.shape[0],seq_array.shape[2],seq_array.shape[1])).float().to(device)
                    outputs = load_model(inputs).to(device)
                    outputs = outputs.squeeze().tolist()
                    scores.extend(outputs)
                test_res.loc[:, f'{nm}_{Y_col}_scores'] = scores

    
    return (test_res)



def import_rounds_data_and_scores(proj, args):
    ##### IMPORT DATA
    data = data_importer(args)

    #extract the data split we want for testing
    idx = (~data.Y_df[f'{args.eval_Y_col}_split'].isna()) #use all the seqs
    #idx = data.Y_df[f'{args.eval_Y_col}_split'] == 'test' #use just the test seqs from that split
    X, _, _ = data.sub_data(args, idx)

    test_res = data.Y_df.loc[idx, [f'{args.eval_Y_col}_enriched', f'{args.eval_Y_col}_logratio']]
    test_res['seq'] = data.X_df.loc[idx, 'FullSeqs_withgaps']
    test_res['full_mut_name'] = pd.Series(test_res.index).str.replace('\/', '_').values

    #other
    seqlen = len(data.X_df['FullSeqs_withgaps'].iloc[0])

    ### Distance to MACS

    #set model_fnms

    nm = 'median_MACS_dist'
    fnm = os.path.join(args.exp_dir, args.sub_exp, 'MACSdist', 'median_dist_to_MACS.csv')
    if os.path.isfile(fnm):
        dist_df = pd.read_csv(fnm).set_index('seq_ID')
        #add
        test_res.loc[:, f'{nm}_scores'] = dist_df.reindex(test_res.index).loc[:, 'median_dist_to_MACS']

    ### Distance to WT

    #import wt sequence
    wt_fnm = os.path.join(args.exp_dir, 'aligned_WT_sequence.csv')
    wt_adf = util.extract_cdrs(wt_fnm)
    wt_seq = wt_adf.iloc[0].loc['FullSeq_nogaps']

    #get distances
    def _get_dist(mut):
        return (len(util.id_mut_from_seq(wt_seq, mut, justnames=True)))

    test_res.loc[:, 'WT_dist_scores'] = test_res['seq'].apply(_get_dist)    

    ### Sklearn models

    #set model_fnms
    for Y_col in ['FACS1_MACS', 'FACS2_MACS', 'FACS3_MACS']:
        model_fnms = {
            'logistic_regression' : os.path.join(args.exp_dir, args.sub_exp, 'logistic',
                                                 f'Train_{Y_col}_enriched',
                                                 f'Train_{Y_col}_enriched.sav') ,
            'linear_regression' : os.path.join(args.exp_dir, args.sub_exp, 'linear_W5e-4',
                                                 f'Train_{Y_col}_logratio',
                                                 f'Train_{Y_col}_logratio.sav')
        }

        #get scores
        for nm,fnm in model_fnms.items():
            if os.path.isfile(fnm):
                #load model
                load_model = joblib.load(fnm)
                #get scores
                if nm == 'logistic_regression':
                    ypred = load_model.model.predict_proba(X)[:,1]
                if nm == 'linear_regression':
                    ypred = load_model.model.predict(X)
                #add to df
                test_res.loc[:, f'{nm}_{Y_col}_scores'] = ypred

    ### Zero shot LLMs

    model_fnms = {
        'ESM2' : f'/n/groups/marks/projects/binding_affinity/model_zoo/unsupervised_ESM2/{proj}/{args.eval_Y_col}_seqs_ESM2_scores.csv'
    }

    #get scores
    for nm, fnm in model_fnms.items():
        if os.path.isfile(fnm):
            #get scores
            sdf = pd.read_csv(fnm).set_index('full_mut_name')
            #add to df
            test_res.loc[:, f'{nm}_scores'] = sdf.reindex(test_res['full_mut_name'])['esm2_t33_650M_UR50D'].values

    ### Semi-supervised models

    model_fnms = {
        'ESM2_MLP_classification' : f'/n/groups/marks/projects/binding_affinity/model_zoo/semi_supervised_ESM2/{proj}/F1_M_seqs_ckpts/MLP_topmodel_enriched/F2_M_scores.csv',
        'ESM2_MLP_regression' : f'/n/groups/marks/projects/binding_affinity/model_zoo/semi_supervised_ESM2/{proj}/F1_M_seqs_ckpts/MLP_topmodel_logratio/F2_M_scores.csv'
    }

    #get scores
    for nm, fnm in model_fnms.items():
        if os.path.isfile(fnm):
            #get scores
            sdf = pd.read_csv(fnm).set_index('seq_ID')
            #add to df
            test_res.loc[:, f'{nm}_FACS1_MACS_scores'] = sdf.reindex(test_res['full_mut_name'])['score'].values

    ### Neural Networks

    #set filenames
    for Y_col in ['FACS1_MACS']:
        model_fnms = {
            'CNN_classification' : os.path.join(args.exp_dir, args.sub_exp, 'CNN',
                                        f'Train_{args.train_Y_col}_enriched', 'ckpt'),
            'CNN_regression' : os.path.join(args.exp_dir, args.sub_exp, 'CNN',
                                        f'Train_{args.train_Y_col}_logratio', 'ckpt'),
            'MLP_classification' : os.path.join(args.exp_dir, args.sub_exp, 'MLP',
                                        f'Train_{args.train_Y_col}_enriched', 'ckpt'),
            'MLP_regression' : os.path.join(args.exp_dir, args.sub_exp, 'MLP',
                                        f'Train_{args.train_Y_col}_logratio', 'ckpt')
        }

        model_fnms = {nm:os.path.join(d, get_latest_ckpt(d)) for nm,d in model_fnms.items() if os.path.isdir(d)}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #get scores from models
        for nm, fnm in model_fnms.items():
            if os.path.isfile(fnm):
                #load model architecture
                model_type = nm.split('_')[0]
                if model_type == 'CNN':
                    load_model = CNN(seqlen)
                elif model_type == 'MLP':
                    load_model = MLP(seqlen*20)

                #load parameters
                load_model.load_state_dict(torch.load(fnm))

                #score sequences
                dataset = OneHotArrayDataset(test_res,'seq',f'{args.eval_Y_col}_enriched')
                loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = 128, shuffle = False)
                scores = []
                for seq_array, labels in loader:
                    inputs = torch.reshape(seq_array,(seq_array.shape[0],seq_array.shape[2],seq_array.shape[1])).float().to(device)
                    labels = labels.reshape(-1,1).to(device)
                    outputs = load_model(inputs).to(device)
                    outputs = outputs.squeeze().tolist()
                    scores.extend(outputs)
                test_res.loc[:, f'{nm}_{Y_col}_scores'] = scores
                
    return (test_res)