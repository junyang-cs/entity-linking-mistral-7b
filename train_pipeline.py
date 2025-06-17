from multiprocessing.context import assert_spawning
import os
import subprocess
import sys
from pathlib import Path 

def run_command(name, command):
    print(f"\n[Task: {name}]")
    print(f"Running: {' '.join(command)}")
    result = subprocess.run(" ".join(command))
    if result.returncode != 0:
        print(f"\n❌ Error in task: {name}")
        sys.exit(1)
    print(f"✅ Completed: {name}")

# Define base paths
base_model_dir = "assets/models/mistralai_Mistral-7B-Instruct-v0.2"
sentence_transformers_dir = "assets/models/sentence-transformers_all-MiniLM-L12-v2"

WORKSPACE_DIR = Path(os.getcwd())
DATA_DIR = WORKSPACE_DIR/"data"
ASSET_DIR = WORKSPACE_DIR/'ASSETS'
DATASET_DIR = DATA_DIR/'dataset'
TRAIN_NOTE_PATH = DATASET_DIR/'train_notes.csv'
TRAIN_ANNOTATION_PATH = DATASET_DIR/'train_annotations.csv'
DEV_NOTE_PATH = DATASET_DIR/'dev_notes.csv'
DEV_ANNOTATION_PATH = DATASET_DIR/'dev_annotations.csv'

# Define tasks
tasks = [
    {
        "name": "Prepare FAISS DB",
        "cmd": [
            "python", "faiss_db_preparation.py",
            "--notes-path", TRAIN_NOTE_PATH.as_posix(),
            "--annotations-path", TRAIN_ANNOTATION_PATH.as_posix(),
            "--terminologies-path", "assets/dataflattened_terminology.csv",
            "--terminologies-path-syn", "assets/newdict_snomed.txt",
            "--terminologies-path-syn-extended", "assets/newdict_snomed_extended.txt",
            "--model-id", base_model_dir,
            "--model-path-faiss", sentence_transformers_dir,
            "--faiss-index", "assets/faiss_index_constitution_all-MiniLM-L12-v2_finetuned"
        ]
    },
    {
        "name": "Prepare Data for Classification with FAISS",
        "cmd": [
            "python", f"{WORKSPACE_DIR.as_posix()}/faiss_classification_data_preparation.py",
            #  TRAIN_NOTE_PATH.as_posix(),
            # TRAIN_ANNOTATION_PATH.as_posix(),
            # "assets/newdict_snomed_extended.txt",
            # "assets/faiss_index_constitution_all-MiniLM-L12-v2_finetuned",
            # "backup/annotations_extended_for_classification.gzip",
            # "mistralai/Mistral-7B-Instruct-v0.2",
            # "sentence-transformers/all-MiniLM-L12-v2"
        ]
    },
    {
        "name": "Finetune Classification Model",
        "cmd": [
            "python", "Finetuning-Classification.py",
            "backup/annotations_extended_for_classification.gzip",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "models/Mistral-7B-Instruct-v0.2-Pescu-faiss-clasify-lora_2/"
        ]
    },
    {
        "name": "Run Inference on Dev Set",
        "cmd": [
            "python", "main.py",
            "--notes-path", "data/dataset/dev_notes.csv",
            "--model-path-peft", "models/Mistral-7B-Instruct-v0.2-AddAnnotations-lora-v0.4",
            "--model-path-2-peft", "models/Mistral-7B-Instruct-v0.2-AddAnnotations-lora-v0.6",
            "--model-classification-path-peft", "models/Mistral-7B-Instruct-v0.2-Pescu-faiss-clasify-lora_2",
            "--faiss-index", "assets/faiss_index_constitution_all-MiniLM-L12-v2_finetuned",
            "--terminologies", "assets/newdict_snomed_extended.txt",
            "--annotations-path", "data/dataset/dev_annotations.csv"
        ]
    }
]

# Execute tasks
for task in tasks[1:]:
    run_command(task["name"], task["cmd"])
