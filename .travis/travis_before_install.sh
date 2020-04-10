#!/usr/bin/env bash

set -e

# Install miniconda to avoid compiling scipy
if test -e $HOME/miniconda/bin ; then
    echo "miniconda already installed."
else
    echo "Installing miniconda."
    rm -rf $HOME/miniconda
    mkdir -p $HOME/download
    if [[ -d $HOME/download/miniconda.sh ]] ; then rm -rf $HOME/download/miniconda.sh ; fi
    wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME/download/miniconda.sh

    mkdir $HOME/.conda
    bash $HOME/download/miniconda.sh -b -p $HOME/miniconda
fi

$HOME/miniconda/bin/conda init bash
source ~/.bash_profile
conda activate base

hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
