# Spanish dataset
wget https://dl.fbaipublicfiles.com/mls/mls_spanish.tar.gz
tar -xvf mls_spanish.tar.gz
python split_transcripts.py --transcripts mls_spanish/train/transcripts.txt --dataset-dir mls_spanish/train/audio
cd mls_spanish
mkdir scripts
cp ../copy_handles_to_low_resource.py scripts
python scripts/copy_handles_to_low_resource.py
cd ..
rm -rf mls_spanish.tar.gz


# French dataset
wget https://dl.fbaipublicfiles.com/mls/mls_french.tar.gz
tar -xvf mls_french.tar.gz
python split_transcripts.py --transcripts mls_french/train/transcripts.txt --dataset-dir mls_french/train/audio
cd mls_french
mkdir scripts
cp ../copy_handles_to_low_resource.py scripts
python scripts/copy_handles_to_low_resource.py
cd ..
rm -rf mls_french.tar.gz

# Dutch dataset
wget https://dl.fbaipublicfiles.com/mls/mls_dutch.tar.gz
tar -xvf mls_dutch.tar.gz
python split_transcripts.py --transcripts mls_dutch/train/transcripts.txt --dataset-dir mls_dutch/train/audio
cd mls_dutch
mkdir scripts
cp ../copy_handles_to_low_resource.py scripts
python scripts/copy_handles_to_low_resource.py
cd ..
rm -rf mls_dutch.tar.gz

# English dataset
wget https://dl.fbaipublicfiles.com/mls/mls_english.tar.gz.partaa
tar -xvf mls_english.tar.gz.partaa
python split_transcripts.py --transcripts mls_english/train/transcripts.txt --dataset-dir mls_english/train/audio
cd mls_english
mkdir scripts
cp ../copy_handles_to_low_resource.py scripts
python scripts/copy_handles_to_low_resource.py
cd ..
rm -rf  mls_english.tar.gz.partaa


mv mls_spanish/spanish_low_resource mls_french/french_low_resource mls_dutch/dutch_low_resource mls_english/english_low_resource train_new
## All transcripts have been prepared for the train sets. We have to combined them in a new train 
