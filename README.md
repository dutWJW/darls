# Density-adaptive-Registration-of-Large-Scale-Point-Clouds-in-Diverse-Outdoor-Environments

We release the codes and scripts used to obtain the metrics reported in the paper.

Please use the following command for installation.

Build conda environment.

- conda
```
conda create -n darls python==3.7
conda activate darls
pip install -r requirements.txt

```
Build extension package
```
python setup.py build develop
cd geotransformer/modules/e2pn/vgtk
python setup.py build_ext -i
```

