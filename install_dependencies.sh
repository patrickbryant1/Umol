#Install conda env
conda env create -f environment.yml

wait
## Get network parameters for Umol (a few minutes)
wget https://zenodo.org/records/10397462/files/params40000.npy
mkdir data/params
mv params40000.npy  data/params/

wait
## Get Uniclust30 (10-20 minutes depending on bandwidth)
# 25 Gb download, 87 Gb extracted
wget http://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/uniclust30_2018_08_hhsuite.tar.gz --no-check-certificate
mkdir data/uniclust30
mv uniclust30_2018_08_hhsuite.tar.gz data
cd data
tar -zxvf uniclust30_2018_08_hhsuite.tar.gz
cd ..

wait
## Install HHblits (a few minutes)
git clone https://github.com/soedinglab/hh-suite.git
mkdir -p hh-suite/build && cd hh-suite/build
cmake -DCMAKE_INSTALL_PREFIX=. ..
make -j 4 && make install
cd ../..

wait
