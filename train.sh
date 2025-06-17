#!/usr/bin/bash
set -e

base_model_dir="assets/models/mistralai_Mistral-7B-Instruct-v0.2"
sentence_transformers_dir="assets/models/sentence-transformers_all-MiniLM-L12-v2"


python faiss_db_preparation.py \
       --notes-path data/dataset/train_notes.csv \
       --annotations-path data/dataset/train_annotations.csv \
       --terminologies-path assets/dataflattened_terminology.csv \
       --terminologies-path-syn assets/newdict_snomed.txt \
       --terminologies-path-syn-extended assets/newdict_snomed_extended.txt \
       --model-id ${base_model_dir} \
       --model-path-faiss ${sentence_transformers_dir} \
       --faiss-index assets/faiss_index_constitution_all-MiniLM-L12-v2_finetuned

if [ $? -ne 0 ]; then
  echo "Error: faiss_db_preparation failed."
  exit 1
fi

# python faiss_db_preparation.py \
#        --notes-path data/mimic-iv_notes_training_set.csv \
#        --annotations-path assets/train_annotations.csv \
#        --terminologies-path assets/dataflattened_terminology.csv \
#        --terminologies-path-syn assets/newdict_snomed.txt \
#        --terminologies-path-syn-extended assets/newdict_snomed_extended-150.txt \
#        --model-id mistralai/Mistral-7B-Instruct-v0.2 \
#        --model-path-faiss sentence-transformers/all-MiniLM-L12-v2 \
#        --faiss-index assets/faiss_index_constitution_all-MiniLM-L12-v2_finetuned-150 \
#        --nr-of-notes 150

python faiss_classification_data_preparation.py \
       data/mimic-iv_notes_training_set.csv \
       assets/train_annotations.csv \
       assets/newdict_snomed_extended-150.txt \
       assets/faiss_index_constitution_all-MiniLM-L12-v2_finetuned-150 \
       backup/annotations_extended_for_classification.gzip \
       mistralai/Mistral-7B-Instruct-v0.2 \
       sentence-transformers/all-MiniLM-L12-v2

if [ $? -ne 0 ]; then
  echo "Error: faiss_classification_data_preparation failed."
  exit 1
fi
# python Finetuning-Entity-Recognition.py \
#        data/mimic-iv_notes_training_set.csv \
#        assets/train_annotations.csv \
#        mistralai/Mistral-7B-Instruct-v0.2 \
#        models/Mistral-7B-Instruct-v0.2-AddAnnotations-lora-v0.4/ \
#        100 \
#        100

# python Finetuning-Entity-Recognition.py \
#        data/mimic-iv_notes_training_set.csv \
#        assets/train_annotations.csv \
#        mistralai/Mistral-7B-Instruct-v0.2 \
#        models/Mistral-7B-Instruct-v0.2-AddAnnotations-lora-v0.6/ \
#        500 \
#        400

python Finetuning-Classification.py \
       backup/annotations_extended_for_classification.gzip \
       mistralai/Mistral-7B-Instruct-v0.2 \
       models/Mistral-7B-Instruct-v0.2-Pescu-faiss-clasify-lora_2/

if [ $? -ne 0 ]; then
  echo "Error: Finetuning-Classification failed."
  exit 1
fi

python main.py \
       --notes-path data/dataset/dev_notes.csv \
       --model-path-peft models/Mistral-7B-Instruct-v0.2-AddAnnotations-lora-v0.4 \
       --model-path-2-peft models/Mistral-7B-Instruct-v0.2-AddAnnotations-lora-v0.6 \
       --model-classification-path-peft models/Mistral-7B-Instruct-v0.2-Pescu-faiss-clasify-lora_2 \
       --faiss-index assets/faiss_index_constitution_all-MiniLM-L12-v2_finetuned \
       --terminologies assets/newdict_snomed_extended.txt \
       --annotations-path data/dataset/dev_annotations.csv \

if [ $? -ne 0 ]; then
  echo "Error: main failed."
  exit 1
fi

# python remove-add-lists.py \
#        backup/df_notes_v4_v6.gzip \
#        assets/train_annotations.csv \
#        submission.csv \
#        sentence-transformers/all-MiniLM-L12-v2 \
#        assets/faiss_index_constitution_all-MiniLM-L12-v2_finetuned \
#        assets/newdict_snomed_extended.txt
