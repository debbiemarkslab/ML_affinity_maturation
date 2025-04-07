import os, sys
sys.path.append('/n/groups/marks/projects/binding_affinity')
import affinity_maturation_utilities as util
from modelling_functions import *
from argparse import ArgumentParser
import joblib

#Arguments
parser = get_training_parser()
#custom parser argument for test settings
parser.add_argument("--train_Y_col",  type=str, help='Which metric was the model trained on')
args = parser.parse_args()

#Do some checks
check_models_match(args)

#set load parameters
s_dir = os.path.join(args.exp_dir, args.experiment_name)
if not os.path.isdir(s_dir):
    os.mkdir(s_dir)
sub_dir = os.path.join(s_dir, args.sub_experiment)
if not os.path.isdir(sub_dir):
    os.mkdir(sub_dir)
sample_name = f'Train_{args.train_Y_col}'
sample_dir = os.path.join(sub_dir, sample_name)
if not os.path.isdir(sample_dir):
    os.mkdir(sample_dir)
    
save_nm = os.path.join(sample_dir, '{}.sav'.format(sample_name))


##### IMPORT DATA

data = data_importer(args)

#### LOAD MODEL

model = joblib.load(save_nm)

#### EVALUATE

eval_sample_dir = os.path.join(sample_dir, f'Test_{args.Y_col}')
if not os.path.isdir(eval_sample_dir):
    os.mkdir(eval_sample_dir)
    
#ROC curve
#save the performance fig
fig_svnm = os.path.join(eval_sample_dir, 'Test ROC: ' + args.Y_col + '.pdf')
y_test_score, roc_stats, fig = model.evaluate(data.X, data.Y, plot=True)
plt.savefig(fig_svnm, format='pdf')
plt.close(fig)

fig_svnm = os.path.join(eval_sample_dir, 'Test PR: ' + args.Y_col + '.pdf')
y_test_score, pr_stats, fig = model.PR_evaluate(data.X, data.Y, plot=True)
plt.savefig(fig_svnm, format='pdf')
plt.close(fig)

#save performance statistics and test results
eval_res = {'y_test_score':y_test_score, 'roc_stats':roc_stats, 'pr_stats':pr_stats}
eval_res_svnm = os.path.join(eval_sample_dir, 'eval_res.sav')
joblib.dump(eval_res, eval_res_svnm) 

if args.ovlp:
    #get the set of the test sequences that were in the overlap
    r2, r1 , _ = args.Y_col.split('_')

    #evaluate
    ovlp_fig_svnm = os.path.join(eval_sample_dir, 'Ovlp performance: ' + args.Y_col + '.pdf')
    y_ovlp_score, ovlp_stats, ovlp_fig = model.evaluate(data.X_ovlp, data.Y_ovlp, plot_title='Ovlp performance')
    plt.savefig(ovlp_fig_svnm, format='pdf')
    plt.close(ovlp_fig)
    #performance statistics and test results
    ovlp_eval_res = {'y_test_score':y_ovlp_score, 'stats':ovlp_stats}
    ovlp_eval_svnm = os.path.join(eval_sample_dir, 'ovlp_eval_res.sav')
    joblib.dump(ovlp_eval_res, ovlp_eval_svnm)