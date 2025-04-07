from Bio.Data import IUPACData
import time as time
import numpy as np
import pandas as pd
import os, sys
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, Lasso, LinearRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import joblib
from argparse import ArgumentParser


### FUNCTIONS THAT HELP WITH PROCESSING DATA FOR MODELLING


def import_AA_alphabet(withgaps = True):
    '''
    Function for easily importing the amino acid alphabet
    and adding gaps
    
    Output:
        Alphabet: the alphabet of choice
        len: the length of the alphabet for downstream processing
    '''
    alphabet_ungapped = IUPACData.protein_letters
    alphabet = alphabet_ungapped + "-"
    if withgaps:
        return (alphabet, len(alphabet))
    else:
        return (alphabet_ungapped, len(alphabet_ungapped))
    
    
def onehot_encode_seqs(seqs, alph, gap_cutoff=0.2, shape='vector', verbose=True):
    '''
    Function to one hot encode sequences
    Note that we are shaping from (_, N, D) -> (_, L*D) to
    get a 1D vector for each seq
    So to go back to a matrix one hot reshape (_,L*D) -> (_,L,D)
    
    Inputs:
        seqs: the series/list containg the sequences
        alph: the alphabet being used
        gap_cutoff: the threshold for which columns to remove for too many gaps
        shape: whether to reshape matrix to a vector
    
    '''
    N = len(seqs)
    L = len(seqs[0])
    D = len(alph)
    unk_let_cnt = 0 #counter for how many unkown letters are found
    #init
    onehot = np.zeros((N, L, D), dtype=np.uint8)
    #populate
    if verbose:
        print ('One hot encoding sequences...')
    ST = time.time()
    for s, seq in enumerate(seqs):
        for l, let in enumerate(seq):
            try:
                a = alph.index(let)
            except:
                a = alph.index('-')
                unk_let_cnt += 1
            onehot[s,l,a] = 1
    if verbose:
        print (f'{100*unk_let_cnt/(N*L):}% of residues were unknown')
        print ('Encoding done. Took {:.2f}s'.format(time.time() - ST))

    #remove cols with too many gaps
    if gap_cutoff:
        #get the gap distribution
        gap_freq = onehot[:,:,-1].mean(0)
        #remove cols
        onehot = onehot[:, (gap_freq <= gap_cutoff), :]
        L = onehot.shape[1]
        
    #reshape
    if shape == 'vector':
        onehot = onehot.reshape(N,L*D)
    
    if verbose:        
        print ('Num seqs: {}\nLength: {}\n'.format(N,L))
            
    return (onehot)



def get_training_parser():
    '''
    Function to initalize the training arguments used in the main data
    importer
    '''
    parser = ArgumentParser()
    parser.add_argument("--model", type=str,
                        default='logistic_regression',
                        help="Which model to use")
    
    parser.add_argument("--data_dir", type=str,
                        default='/n/groups/marks/projects/nanobodies/kruse_nanobody_sequencing/B2AR_affinity/campaign_A_v3/fil5_filabund1/',
                        help="Where your training and test data lives")
    
    parser.add_argument("--data_sub", type=str,
                        default='manual',
                        help="What subsplit of the data youre using")
    
    parser.add_argument("--exp_dir", type=str,
                        default='B2AR_v2',
                        help="The folder in the code directory for this project")
    
    parser.add_argument("--experiment_name", type=str,
                        default = 'campaign_A',
                        help="the experiment within this project")
    
    parser.add_argument("--sub_experiment", type=str,
                        default = 'testmodel',
                        help='The sub experiment within the project')
    
    parser.add_argument("--CDR_choice", type=str,
                        default='FullSeqs_withgaps')
    
    parser.add_argument("--gap_cutoff", type=float,
                        default=1.0,
                       help = 'Only columns with less than this gap frequency will be kept')
    
    parser.add_argument("--no_wt_cols", action='store_true',
                        default=False,
                        help="Whether to train with the WT columns or not")
    
    parser.add_argument("--Y_col", type=str,
                        default = 'FACS1_MACS_enriched',
                        help='Which metric to use')
    
    parser.add_argument("--split", type=str,
                        default='train', 
                        help='Training or Test split')
    
    parser.add_argument("--ovlp", action='store_true',
                        default=False,
                        help="Whether to only run on sequences in both rounds")
    
    parser.add_argument("--l1_lambda", type=float,
                        default=1.0,
                        help="The regularization strength for the l1 penalty")
    return (parser)


class data_importer():
    '''
    class for importing the data. 
    '''
    
    def __init__(self, args):
        
        #set some defaults
        if 'split' not in dir(args):
            args.split = 'train'
        if 'ovlp' not in dir(args):
            args.ovlp = False
        if 'CDR_choice' not in dir(args):
            args.CDR_choice = 'FullSeqs_trimgaps'
            args.gap_cutoff = 1.0
        if 'no_wt_cols' not in dir(args):
            args.no_wt_cols = True
        
        #Import alphabet
        if ('withgaps' in args.CDR_choice) | (('trimgaps' in args.CDR_choice)):
            WG = True
        else:
            WG = False
            args.gap_cutoff = None
        alphabet, D = import_AA_alphabet(withgaps=WG)

        #import X data
        X_df = pd.read_csv(os.path.join(args.data_dir, f'X_{args.data_sub}.csv'))
        #Import Y data
        Y_df = pd.read_csv(os.path.join(args.data_dir, f'Y_{args.data_sub}.csv'))
        #make sure X_df is in the same order as Y_df using the seq_ID
        X_df = X_df.set_index('seq_ID').loc[Y_df['seq_ID']]
        Y_df = Y_df.set_index('seq_ID')
        #get the IMGT and IMGT_AA columns used in this dataset
        imgt_col, imgt_aa_col = get_which_IMGT_cols(args, alphabet)
        
        self.X_df = X_df
        self.Y_df = Y_df
        self.alphabet = alphabet
        self.imgt_col = imgt_col
        self.orig_imgt_aa_col = imgt_aa_col
        
        #by default sub out the data according to which split is being used
        col = '_'.join(args.Y_col.split('_')[:-1])
        idx = Y_df[f'{col}_split'] == args.split
        
        X,Y,imgt_aa_col = self.sub_data(args,idx)
        self.X = X
        self.Y = Y
        self.imgt_aa_col = imgt_aa_col
        
        if args.ovlp:
            self.sub_ovlp(args)
        
    def sub_data(self,args,idx,Y_col=None):
        '''
        idx: a boolean index vector or list of seq_ID to extract the
                data being imported
        Y_col: the column in the Y_df dataframe to use as Y
        '''
        if not Y_col:
            Y_col = args.Y_col
        
        Y_m = self.Y_df.loc[idx, Y_col].values.astype(np.float64)
        N_m = len(idx)
        print ('{} set size: {}'.format(args.split, N_m))
        if 'enriched' in Y_col:
            print ('Num enriched: {}\nNum not enriched: {}'.format(
                    sum(Y_m==1), sum(Y_m == 0)))
        #select seqs        
        X_seq = self.X_df.loc[idx.values, args.CDR_choice]
        #onehot encode
        X = onehot_encode_seqs(X_seq, self.alphabet, args.gap_cutoff)
        if args.no_wt_cols:
            X, wt_oh, imgt_aa_col = remove_wt_cols(args, X, self.alphabet, self.imgt_col, self.orig_imgt_aa_col)        
            return (X, Y_m, imgt_aa_col)
        else:
            return (X, Y_m, self.orig_imgt_aa_col)
        
        
    def seq_to_OH(self, args, X_seq):
        '''
        Convert any list/array of sequences to a OH encoded array
        Need to pass in a sequence that is only substititions of the
        WT sequence.
        '''
        assert len(X_seq[0]) == len(self.imgt_col), "Pass in sequences that are substitutions of the WT sequence"
        
        #onehot encode
        X = onehot_encode_seqs(X_seq, self.alphabet, args.gap_cutoff, verbose=False)
        if args.no_wt_cols:
            X, wt_oh, imgt_aa_col = remove_wt_cols(args, X, self.alphabet, self.imgt_col, self.orig_imgt_aa_col)        
        return (X)

        
    
    
    def sub_ovlp(self, args):
        #get the set of the test sequences that were in the overlap
        r2, r1 , _ = args.Y_col.split('_')
        ovlp_idx = ((self.Y_df[f'{r1}_hasread']) & (self.Y_df[f'{r2}_hasread'])) & (self.Y_df[f'{r2}_{r1}_split'] == args.split)
        
        X,Y,_ = self.sub_data(args, ovlp_idx)
        self.X_ovlp = X
        self.Y_ovlp = Y
        self.ovlp_idx = ovlp_idx

            
            
            
            
def get_which_IMGT_cols(args, alphabet):
    '''
    Helper function to data_importer for getting the
    IMGT columns used in this dataset
    '''
    
    #import the IMGT cols
    imgt_col_fnm = os.path.join(args.data_dir, f'imgt_labels.sav')
    imgt_col = joblib.load(imgt_col_fnm)[args.CDR_choice]
    
    #make a index list to identify which IMGT_AA cols
    #are in the trained model
    col_idx = np.ones((len(alphabet), len(imgt_col)))
    col_df = pd.DataFrame(col_idx, index=list(alphabet), columns=imgt_col)
    col_df = col_df.reset_index().rename({'index':'AA'}, axis=1)
    col_dfm = pd.melt(col_df, id_vars = ['AA'], var_name = 'IMGT')
    col_dfm['IMGT_AA'] = col_dfm['IMGT'] + '_' + col_dfm['AA']
    imgt_aa_col = col_dfm.drop('value', axis=1)
    
    return (imgt_col, imgt_aa_col)
            
            
def remove_wt_cols(args, X, alphabet, imgt_col, imgt_aa_col):
    '''
    Helper function to data_importer for removing 
    wt columns from a sequence onehot encoding
    Requires a csv output of the aligned ANARCI wt sequence 
    (easy to use the online version of ANARCI for this)
    '''
    
    #import the WT alignment
    wt_fnm = os.path.join('/n/groups/marks/projects/binding_affinity',
                          args.exp_dir, 'aligned_WT_sequence.csv')
    wt_adf = pd.read_csv(wt_fnm, index_col=0)
    #add in any train_cols that the wt doesn't have
    train_col_not_in_wt = imgt_col[~pd.Series(imgt_col).isin(wt_adf.columns)]
    if len(train_col_not_in_wt) > 0:
        print ('Train Columns not in WT sequence')
        print (train_col_not_in_wt)
        print ('Adding to wt sequence')
        for col in train_col_not_in_wt:
            wt_adf[col] = '-'
    wt_keep = wt_adf.loc[:, imgt_col]
    wt_seq = ''.join(list(np.squeeze(wt_keep.values)))
    #onehot encode the WT sequence
    wt_oh = onehot_encode_seqs([wt_seq], alphabet, args.gap_cutoff, verbose=False)[0]
    #remove the WT cols from the data
    X = X[:, ~wt_oh.astype(bool)]
    
    #edit which cols are in the data
    nw_cols = imgt_aa_col[~wt_oh.astype(bool)]

    return (X, wt_oh, nw_cols)


            

def arrange_idx(uq_imgt):
    '''
    Function for ordering imgt keys
    '''

    def switch_ordering(df, ii):
        ndf = pd.concat([df[df['num'] < ii],
                     df[df['num'] == ii].iloc[::-1, :],
                     df[df['num'] > ii]])
        return (ndf)

    #construct an indexing df
    uq_imgt_df = pd.DataFrame({'imgt':uq_imgt})
    uq_imgt_df['num'] = uq_imgt_df['imgt'].str.extract('(\d+)', expand=False)
    uq_imgt_df['num'] = pd.to_numeric(uq_imgt_df['num'] )
    uq_imgt_df['let'] = uq_imgt_df['imgt'].str.extract('([A-z]+)', expand=False)
    uq_imgt_df.loc[uq_imgt_df['let'].isna(), 'let'] = ''
    uq_imgt_df = uq_imgt_df.sort_values(['num', 'let'], ascending=(True, True))
    #flip the ordering of certain IMGT numbers
    uq_imgt_df = switch_ordering(uq_imgt_df, 33)
    uq_imgt_df = switch_ordering(uq_imgt_df, 61)
    uq_imgt_df = switch_ordering(uq_imgt_df, 112)
    uq_imgt_df = uq_imgt_df.reset_index(drop=True)
    #make some dicts for easy indexing
    imgt_ii_key = {imgt:ii for ii, imgt in enumerate(uq_imgt_df['imgt'])}
    ii_imgt_key = {ii:imgt for ii, imgt in enumerate(uq_imgt_df['imgt'])}
        
    return (uq_imgt_df, imgt_ii_key, ii_imgt_key)


def plot_roc_curve(fpr, tpr, roc_auc, title=None):
    
    '''
    Self defined function to plot a roc curve and return the object
    '''
    
    lw=2
    fig, ax = plt.subplots(1,1, figsize=(5,3.5))
    ax.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Test Performance')
    ax.legend(loc="lower right")
    return (fig)


def plot_pr_curve(recall, precision, pr_auc, title=None):
    
    '''
    Self defined function to plot a PR curve and return the object
    '''
    
    lw=2
    fig, ax = plt.subplots(1,1, figsize=(5,3.5))
    ax.plot(recall, precision, color='blue',
             lw=lw, label='PR curve (area = %0.2f)' % pr_auc)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Test Performance')
    ax.legend(loc="lower right")
    return (fig)


def plot_scatter_r2(y, y_hat, r2, title=None):
    
    '''
    Self defined function to plot a Scatter plot 
    comparing the predictions to truth with the
    coefficient of determination (R^2) plotted
    '''
    
    lw=2
    fig, ax = plt.subplots(1,1, figsize=(5,3.5))
    ax.scatter(y, y_hat, color='darkorange',
             label='R^2 = {:.3f}'.format(r2))
    ax.set_xlabel('True value')
    ax.set_ylabel('Predicted value')
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Test Performance')
    ax.legend(loc="lower right")
    return (fig)
    
    
### FUNCTIONS THAT HELP WITH MODELLING

def check_models_match(args):
    '''
    Function wrapping up any checks we would want to employ
    for matching models with data
    '''
    
    if (args.model == 'linear_regression') & ('logratio' not in args.Y_col):
        raise Exception("The Logratio metric needs to be used with a Linear Regression model")
        
    if (args.model == 'logistic_regression') & ('enriched' not in args.Y_col):
        raise Exception("The enriched (binary label) metric needs to be used with a Logistic Regression model")



class logistic_regression_trainer():
    '''
    Class for performing Logistic Regression
    '''
    def __init__(self, data, args):
        self.imgt_aa_col = data.imgt_aa_col.copy()
        self.args = args
        if args.l1_lambda == 0.:
            self.model = LogisticRegression(random_state=0, penalty='none', solver='saga', fit_intercept=True)
        else:
            self.model = LogisticRegression(random_state=0, penalty='l1', solver='saga', fit_intercept=True, C=1/args.l1_lambda)
        
        
    def get_rocauc(self,X,Y):
        yscore = self.model.predict_proba(X)[:,1]
        fpr,tpr,thr = roc_curve(Y,yscore)
        return (auc(fpr, tpr))
        
        
    def train_model(self, X, Y, plot=True, plot_title='Train ROC'):
        ST = time.time()
        self.model.fit(X,Y)
        print('Training model: {:.3f}s'.format(time.time()-ST))
        
        #report ROC-AUC
        self.roc_auc_train = self.get_rocauc(X, Y)
        if plot:
            yscore = self.model.predict_proba(X)[:,1]
            fpr,tpr,thr = roc_curve(Y,yscore)
            train_rocauc = auc(fpr, tpr)
            fig = plot_roc_curve(fpr, tpr, train_rocauc, plot_title)
        print ('Training ROC-AUC: {:.3f}'.format(self.roc_auc_train))
        
    
    def evaluate(self, X,Y, plot=True, plot_title='Test ROC'):
        #doesnt use get_rocauc to avoid running the model twice
        yscore = self.model.predict_proba(X)[:,1]
        fpr,tpr,thr = roc_curve(Y,yscore)
        roc_stats = {'fpr':fpr, 'tpr':tpr, 'thr':thr}
        eval_rocauc = auc(fpr, tpr)
        
        print ('Evaluation ROC-AUC: {:.3f}'.format(eval_rocauc))
        if plot:
            fig = plot_roc_curve(fpr, tpr, eval_rocauc, plot_title)
            return (yscore, roc_stats, fig)
        else:
            return (yscore, roc_stats)
        
    
    def PR_evaluate(self, X,Y, plot=True, plot_title='Precision-Recall curve'):
        yscore = self.model.predict_proba(X)[:,1]
        precision, recall, thresholds = precision_recall_curve(Y, yscore)
        pr_stats = {'precision':precision, 'recall':recall, 'thr':thresholds}
        eval_prauc = auc(recall, precision)
        
        print ('Evaluation PR-AUC: {:.3f}'.format(eval_prauc))
        if plot:
            fig = plot_pr_curve(recall, precision, eval_prauc, plot_title)
            return (yscore, pr_stats, fig)
        else:
            return (yscore, pr_stats)
        
        
        
class linear_regression_trainer():
    '''
    Class for performing linear regression
    (By default L1 regularization is used - Lasso)
    '''
    
    def __init__(self, data, args):
        self.imgt_aa_col = data.imgt_aa_col.copy()
        self.args = args
        self.model = Lasso(alpha=args.l1_lambda, random_state=0, fit_intercept=True)
        #self.model = LinearRegression(fit_intercept=False)
        
        
    def train_model(self, X, Y, plot=True, plot_title='Train scatter'):
        ST = time.time()
        self.model.fit(X,Y)
        print('Training model: {:.3f}s'.format(time.time()-ST))
        
        #report R2 metric
        self.r2_train = self.model.score(X, Y)
        if plot:
            yhat = self.model.predict(X)
            fig = plot_scatter_r2(Y, yhat, self.r2_train, plot_title)
        print ('Training R^2: {:.3f}'.format(self.r2_train))
        
    
    def evaluate(self, X,Y, plot=True, plot_title='Test scatter'):
        
        eval_r2 = self.model.score(X, Y)
        stats = {'eval_r2':eval_r2}
        print ('Evaluation R^2: {:.3f}'.format(eval_r2))
        if plot:            
            yhat = self.model.predict(X)
            fig = plot_scatter_r2(Y, yhat, eval_r2, plot_title)
            return (yhat, stats, fig)
        else:
            return (yhat, stats)
        

class logistic_regression_trainer_SI():
    #from selectinf.algorithms import lasso
    '''
    Class for performing Logistic Regression
    '''
    def __init__(self, data, args):
        self.imgt_aa_col = data.imgt_aa_col.copy()
        self.args = args
        
    def get_rocauc(self,X,Y):
        yscore = self.model.predict_proba(X)[:,1]
        fpr,tpr,thr = roc_curve(Y,yscore)
        return (auc(fpr, tpr))
        
        
    def train_model(self, X, Y, plot=True, plot_title='Train ROC'):
        ST = time.time()
        #add a constant term to the X
        Xc = np.hstack([X, np.ones(X.shape[0])[:, None]])
        self.coef_summary = self.imgt_aa_col.copy()
        self.coef_summary.loc[self.coef_summary.index[-1] + 1] = ['-', '-', 'Const']
        #train
        self.modelSI = lasso.lasso.logistic(Xc, Y, feature_weights=1/self.args.l1_lambda)
        self.modelSI.fit()
        print('Training model: {:.3f}s'.format(time.time()-ST))
        #extract coefficients
        summary = self.modelSI.summary(compute_intervals=False)
        self.coef_summary['beta'] = self.modelSI.soln
        self.coef_summary['pvalue'] = np.nan
        self.coef_summary.loc[self.modelSI.soln != 0, 'pvalue'] = summary['pval'].values
        
        #convert to a sklearn model
        self.model = LogisticRegression(random_state=0, penalty='l1', solver='saga', fit_intercept=True, C=self.args.l1_lambda)
        coefs, intercept = self.modelSI.soln[:-1], self.modelSI.soln[-1]
        self.model.coef_, self.model.intercept_, self.model.classes_ = np.expand_dims(coefs, 0), intercept, np.array([False,  True])
        
        #report ROC-AUC
        self.roc_auc_train = self.get_rocauc(X, Y)
        if plot:
            yscore = self.model.predict_proba(X)[:,1]
            fpr,tpr,thr = roc_curve(Y,yscore)
            train_rocauc = auc(fpr, tpr)
            fig = plot_roc_curve(fpr, tpr, train_rocauc, plot_title)
        print ('Training ROC-AUC: {:.3f}'.format(self.roc_auc_train))
        
    
    def evaluate(self, X,Y, plot=True, plot_title='Test ROC'):
        #doesnt use get_rocauc to avoid running the model twice
        yscore = self.model.predict_proba(X)[:,1]
        fpr,tpr,thr = roc_curve(Y,yscore)
        roc_stats = {'fpr':fpr, 'tpr':tpr, 'thr':thr}
        eval_rocauc = auc(fpr, tpr)
        
        print ('Evaluation ROC-AUC: {:.3f}'.format(eval_rocauc))
        if plot:
            yscore = self.model.predict_proba(X)[:,1]
            fpr,tpr,thr = roc_curve(Y,yscore)
            test_rocauc = auc(fpr, tpr)
            fig = plot_roc_curve(fpr, tpr, test_rocauc, plot_title)
            return (yscore, roc_stats, fig)
        else:
            return (yscore, roc_stats)