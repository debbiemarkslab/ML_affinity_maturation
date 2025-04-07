# Machine learning enables efficient and effective affinity maturation of nanobodies

Steffanie Paul1,2,*, Edward P. Harvey3,*, Aaron W. Kollasch1,2, Adam J. Riesselman1,2, James Osei-Owusu3, Louis R. Hollingsworth4, Biswaranjan Pani5, J. Wade Harper4, Andrew C. Kruse3,†, Debora S. Marks1,2,†

This repo contains the code to run the methods developed in the paper: _Machine learning enables efficient and effective affinity maturation of nanobodies_. This includes:
- Code for processing sequencing data from MiSeq pair-end sequencing into training data
- Code for training models on sequence data

We have also included the model checkpoints for the models trained on the `AT110` dataset for comparisons.


---

### Data processing

We have provided here the ML-ready datasets for each campaign in the `data` folder. Each folder contains a `X.csv` and a `Y.csv` file which contain the sequences and enrichment labels needed to train ML models. The raw fastq sequencing files for the affinity maturation campaigns were not provided as the files are very large. However, we have provided  code for processing fastq files into aligned datasets, and then for packaging the aligned datasets into ML-ready formats. See contents of `fastq_processing`.

---

### Packages install

Two environments are required for running this analysis. 1) `bioinf` for processing the fastq files into ML ready data and 2) `baff` for final data formatting and model training. 


```
conda env create -f baff.yml -n baff
conda env create -f bioinf.yml -n bioinf
```

Furthermore, the ANARCI package is required to align the sequencing data. With the `bioinf` env activated, follow the instructions on (https://github.com/oxpig/ANARCI) to install ANARCI.

---

### Logistic and linear regression

To train a logistic or linear regression model change the parameters located in the `run_scripts/train_test_logistic_regression.sh` or `run_scripts/train_test_linear_regression.sh` respectively to specify the data set and name of experiment. Note: logistic and linear regression models can only be trained for classification or regression respectively. This requires pointing to the `_enriched` (binary) or `_logratio` (continuous valued) label columns in the `Y.csv` dataset.

```bash
sh run_scripts/train_test_logistic_regression.sh
```

To plot a heatmap of the un-normalized per-site weights for the linear or logistic regression models:

```bash
python scripts/LR_model_parameters_heatmap.py --model_pth trained_models/AT110/compmodels/logistic/Train_FACS1_MACS_enriched/Train_FACS1_MACS_enriched.sav \
                                              --wt_alignment data/AT110/aligned_wt_sequence.csv \
                                              --fig_svpth trained_models/AT110/compmodels/logistic/Train_FACS1_MACS_enriched/Train_FACS1_MACS_enriched_heatmap.pdf \
                                              --model_nm AT110
```

---

### CNN model

Similar to above modify the parameters in the following script to point to the correct data.

```bash
sh run_scripts/train_CNN.sh
```

---

### Semi-supervised model (predictor on top of ESM2 embeddings)

To recapitulate the EMS2-MLP results, first ESM2 embeddings are extracted and then an MLP top-model is trained on top of these. To extract the ESM2 embeddings, a conda env with the esm installation must be present. To prevent confounding with base env, please install esm in a new conda env (more info on [https://github.com/facebookresearch/esm]).

```bash
conda create -n esm python=3.9
pip install "fair-esm[esmfold]"
# OpenFold and its remaining dependency
pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'
```

To run the full pipeline on the example AT110 data:

```bash
bash run_scripts/train_ESM2_MLP_full_pipeline.sh AT110 AT1R_affinity V1
```



