# Steps for running MFA
ref: [Montreal forced aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/)

```bash 
$ cd /ocean/projects/cis220031p/hkook
$ conda activate ./envs/aligner
$ export MFA_ROOT_DIR=/ocean/projects/cis220031p/hkook/Documents/MFA
# Data is present in /ocean/projects/cis220031p/hkook/Koushik
# Need to prepare lab files for each file in case they are all present in single transcript file. 
# Use  python3 create_lab_files.py --root /abs/path/to/LibriSpeech/train-clean-100 for this


# For fetching dictionary used in alignment and training
$ mfa model download dictionary english_mfa

# For getting mfa model for alignments
$ mfa model download acoustic english_us_arpa

# For training
$ mfa train [OPTIONS] CORPUS_DIRECTORY DICTIONARY_PATH OUTPUT_MODEL_PATH

# For alignment generation
$ mfa align [OPTIONS] CORPUS_DIRECTORY DICTIONARY_PATH ACOUSTIC_MODEL_PATH OUTPUT_DIRECTORY 
```