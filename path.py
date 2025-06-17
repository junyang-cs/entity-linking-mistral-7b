from pathlib import Path
import os 

WORKSPACE_DIR = Path(os.getcwd())

base_model_dir = "assets/models/mistralai_Mistral-7B-Instruct-v0.2"
sentence_transformers_dir = "assets/models/sentence-transformers_all-MiniLM-L12-v2"



DATA_DIR = WORKSPACE_DIR/"data"
ASSET_DIR = WORKSPACE_DIR/'assets'
DATASET_DIR = DATA_DIR/'dataset'
TRAIN_NOTE_PATH = DATASET_DIR/'train_notes.csv'
TRAIN_ANNOTATION_PATH = DATASET_DIR/'train_annotations.csv'
DEV_NOTE_PATH = DATASET_DIR/'dev_notes.csv'
DEV_ANNOTATION_PATH = DATASET_DIR/'dev_annotations.csv'