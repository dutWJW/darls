# Density-adaptive-Registration-of-Large-Scale-Point-Clouds-in-Diverse-Outdoor-Environments

We release the codes and scripts used to obtain the metrics reported in the paper.

Please use the following command for installation.


- conda
```bash
conda create -n darls python==3.7
conda activate darls
pip install -r requirements.txt
```

- Build extension package

Replace the PYTHON_EXECUTABLE, PYTHON_INCLUDE_DIRS and PYTHON_LIBRARIES in the following two files with the correct path.

[distributionSample/pybind11/CMakeLists.txt](/utils/distributionSample/pybind11/CMakeLists.txt)

[uniformSample/pybind11/CMakeLists.txt](utils/uniformSample/pybind11/CMakeLists.txt)

```bash
cd utils
sh compile.sh
```
### Prepare Dataset and Weights

downloading datasets and uncompress them to ./DATASET

downloading weights and uncompress them to ./models/checkpoints

[Link](https://pan.quark.cn/s/c4b44f7d9f5d)


### Wildplaces
```bash
python testWildplaces.py --dataset 'DATASET_PATH/wildplaces'
```
### ETH
```bash
python testETH.py --dataset 'DATASET_PATH/eth'
```
### KITTI
```bash
python testKITTI.py --dataset 'DATASET_PATH/kitti'
```
