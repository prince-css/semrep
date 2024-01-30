--------------
PROJECT TITLE
--------------
Factuality check of the SemRep Predications

------------
Description
------------

The project deals with a Transformer based Language Model to filter predications belonging to the following subset of
predicates from SemMedDB, informally called the "Substance Interactions" group:

 * INTERACTS\_WITH
 * STIMULATES
 * INHIBITS


-------------------
AUTHOR INFORMATION
-------------------
Md Rakibul Islam Prince
Graduate Research Assistant
Department of Electrical and Computer Engineering
Indiana University-Purdue University Indianapolis
Email: mdprin@iu.edu


---------------------
PACKAGE INSTALLATION
---------------------
To reproduse the results at first all the necessarry packages are needed to be installed.
the "semrepenv.yml" YAML file encapsulates the conda environment I used.

run 
>> conda env create -f semrepenv.yml
>> conda activate semrepenv
or,
>> pip install -r requirements.txt

to install the environment before running any scripts or notebook.
Or, you can manually install the packages from the "requirements.txt" file


----------------------------------
PROJECT STRUCTURE AND DESCRIPTIONS
----------------------------------
/semrep  
├── /data  
│   ├── substance_interactions.csv  
│   └── substance_interactions_cleaned.csv  
├── /logs  
│   ├── bert_logfile.log  
│   ├── biobert_logfile.log  
│   └── ...  
├── /models  
│   ├── semrep_simple_bert_model  
│   ├── semrep_simple_biobert_model  
│   └── ...  
├── /plots  
│   ├── bert_cat_arg_dis_impact_all.png  
│   ├── bert_cat_arg_dis_impact_verbal.png  
│   ├── bert_cum_arg_dis_impact_all.png  
│   ├── bert_cum_arg_dis_impact_verbal.png  
│   ├── bert_precision_recall_curve_all.png  
│   ├── bert_precision_recall_curve_verbal.png  
│   ├── bert_roc_curve.png  
│   ├── bert_sub_obj_heatmap_all.png   
│   ├── bert_sub_obj_heatmap_verbal.png  
│   └── ...  
├── /results  
│   ├── bert_test_set_0_results.csv  
│   ├── val_bert_results.csv  
│   ├── test_bert_results.csv  
│   └── ...  
├── /src  
│   ├── semrep_model.ipynb  
│   └── utils.py  
├── README.txt  
├── requirements.txt  
└── semrepenv.yml  
  

Below is an overview of the key files and folders in this project:

- `data/': Directory where the raw and processed data files are stored.
- `data/substance_interactions.csv': raw data file
- `data/substance_interactions_cleaned.csv': processed and clean data file

- `logs/`: Directory containing the logs for each model.
- `logs/<model_name>_logfile.log`: logfile for model <model_name>

- `models/`: Directory containing the finetuned checkpoints of the models.
- `plots/`: Directory containing all the generated plots during analysis.
- `results/`: Directory where the test and validation results are installed.

- `src/`: Directory containing the model notebooks and scripts.
- `src/semrep_model.ipynb`: Notebook detailing the full implimentation of the project
- `src/utils.py`: Scripts used for data analysis visualization tasks

- `README.txt': File detailing the description of the codebase.
- `requirements.txt': File detailing necessarry packages.
- `semrepenv.yml': File for recreating the environment.



