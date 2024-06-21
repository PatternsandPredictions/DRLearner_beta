#!/bin/bash

echo "This is post-installation script"

pip install pip==21.3
pip install jax==0.3.7 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install jaxlib==0.3.7+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install setuptools==65.5.0
pip install wheel==0.38.0
pip install git+https://github.com/horus95/lazydict.git
pip install --no-cache-dir -r requirements.txt
pip install git+https://github.com/ivannz/gymDiscoMaze.git@stable

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib
echo $LD_LIBRARY_PATH

wget http://www.atarimania.com/roms/Roms.rar
unar Roms.rar
mv Roms roms
ale-import-roms roms/
