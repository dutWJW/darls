# Density-adaptive-Registration-of-Large-Scale-Point-Clouds-in-Diverse-Outdoor-Environments

[RA-L 25] Density-adaptive-Registration-of-Large-Scale-Point-Clouds-in-Diverse-Outdoor-Environments

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

[Quark Link](https://pan.quark.cn/s/107feaf4b2a8)


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

### Cite
```
@ARTICLE{10919083,
  author={Wang, Jiawei and Zhuang, Yan and Yan, Fei and Zhang, Hong},
  journal={IEEE Robotics and Automation Letters}, 
  title={Density Adaptive Registration of Large-Scale Point Clouds in Diverse Outdoor Environments}, 
  year={2025},
  volume={10},
  number={5},
  pages={4636-4643},
  keywords={Point cloud compression;Feature extraction;Convolutional neural networks;Bandwidth;Matched filters;Collaboration;Training;Three-dimensional displays;Sensors;Density measurement;Multi-robot systems;Density adaptive registration;point density variations;diverse outdoor environments},
  doi={10.1109/LRA.2025.3549659}}
```
