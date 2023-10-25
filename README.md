# LOTUS: Learning to Optimize Task-based US representations
### [Project Page](https://danivelikova.github.io/lotus/) | [Paper](https://arxiv.org/pdf/2307.16021.pdf) | [Data](https://github.com/danivelikova/lotus/releases/tag/miccai2023)

## Getting Started

You can install the required Python dependencies like this:

```bash
conda env create -f environment.yml
conda activate lotus
```

## Reproducing results

The hyperparameters set in `config/` can be used to reproduce the results from the paper. The data from the paper can be found in the release on GitHub, and the files should be placed in the `datasets/` directory of this repository:

```
data
├── CT_labelmaps/
├── trainA_500/
├── GT_data_stopp_crit/
├── GT_data_testing/
```

For training run this command:

```bash
python train.py -c config/config_file.yml
```
For inferencing run this command:

```bash
python inference.py -c config/config_file.yml
```

### Citation

```Bibtex
@inproceedings{velikova2023lotus,
            title={LOTUS: Learning to Optimize Task-Based US Representations},
            author={Velikova, Yordanka and Azampour, Mohammad Farid and Simson, Walter and Gonzalez Duque, Vanessa and Navab, Nassir},
            booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
            pages={435--445},
            year={2023},
            organization={Springer}
          }
```