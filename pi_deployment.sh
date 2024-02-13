#!/bin/bash
mkdir -r venv/number_plate
python -m --system-site-packages venv venv/number_plate
source venv/number_plate/bin/activate
pip install --upgrade setuptools
sudo apt-get install mupdf mupdf-tools libcap-dev python3-pyqt5 python3-opengl fonts-wqy-zenhei
pip install -r pi_requirements.txt --upgrade
