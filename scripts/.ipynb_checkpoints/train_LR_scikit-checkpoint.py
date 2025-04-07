import os, sys
sys.path.append('/n/groups/marks/projects/binding_affinity')
import affinity_maturation_utilities as util
from modelling_functions import *
from argparse import ArgumentParser
import joblib

#Arguments
parser = get_training_parser()
args = parser.parse_args()

#Do some checks
check_models_match(args)

#set save parameters
s_dir = os.path.join(args.exp_dir, args.experiment_name)
if not os.path.isdir(s_dir):
    os.makedirs(s_dir)
sub_dir = os.path.join(s_dir, args.sub_experiment)
if not os.path.isdir(sub_dir):
    os.makedirs(sub_dir)
sample_name = f'Train_{args.Y_col}'
sample_dir = os.path.join(sub_dir, sample_name)
if not os.path.isdir(sample_dir):
    os.makedirs(sample_dir)
    
save_nm = os.path.join(sample_dir, '{}.sav'.format(sample_name))



##### IMPORT DATA

data = data_importer(args)

#### FIT MODEL
if args.model == 'linear_regression':
    trainer = linear_regression_trainer(data, args)
elif args.model == 'logistic_regression':
    trainer = logistic_regression_trainer(data, args)

trainer.train_model(data.X, data.Y)

#### SAVE MODEL
joblib.dump(trainer, save_nm)

