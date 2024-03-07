. "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda create --name DDCFR python=3.7.10 -y
conda activate DDCFR
pip install "setuptools==65.5.0"
pip install -e .
conda create --name PokerRL python=3.7.10 -y
conda activate PokerRL
pip install "setuptools==65.5.0"
pip install -e .
cd PokerRL
tar -xzvf texas_lookup.tar.gz
pip install -e .
