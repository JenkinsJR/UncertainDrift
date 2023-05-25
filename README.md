# A DNN Framework for Learning Lagrangian Drift With Uncertainty

<a href='https://arxiv.org/abs/2204.05891v2'><img src='https://img.shields.io/badge/paper-arXiv-red'></a>

## Installing requirements through conda

**Main environment**

```
conda env create -f envs/main.yml
```

**Trajectory simulation ([OceanParcels](https://github.com/OceanParcels/parcels)) environment**

```
conda env create -f envs/parcels.yml
```

## Download field data and pretrained models

**Data**: daily velocity and SSH fields for the years 2018 and 2016 can be downloaded [here](https://drive.google.com/file/d/1D4jow_VCndjcka1R7Da7Iy7_9Ro65zsi/view?usp=drive_link).

**Models**: pretrained models (24 epochs) for different input scenarios (Emulator, SSH, Forecast) can be downloaded [here](https://drive.google.com/file/d/1K8T9dRHFY84k7cACYDZ8_-mpF9zJUcKR/view?usp=drive_link).

Place the extracted contents into your desired root directory `<rootdir>`:

```
<rootdir>/
├── data/
├── models/
```

## Running scripts

Run the scripts using the top level directory of *this repository as the current working directory*.

All scripts require the above `<rootdir>` to be specified as the first positional argument. To change the behaviour of scripts from the default, you may use the `-h` flag to get a list of other arguments.

### Data generation

***Warning: for each dataset***, the default configuration (matching the paper) will generate ~27GB of trajectory data and ~112GB of density map data.

This corresponds to 20K 15-day probabilistic trajectories with daily outputs (300K snapshots). See [(1) Trajectories](#1-trajectories) for how to generate a smaller dataset.

#### (1) Trajectories

Using the **raw** velocity fields, run trajectory similations for the datasets **daily2018** and **daily2016** and output the data to a simulation directory named **15day**.

A smaller dataset may be generated by using fewer or shorter trajectories, by using the `–-samples` and `–-runtime` arguments, respectively.

```
conda activate parcels

python -m scripts.gendata.trajectories.main <rootdir> daily2018 raw 15day
python -m scripts.gendata.trajectories.main <rootdir> daily2016 raw 15day
```

#### (2) Probability density maps

Generate density maps for the datasets **daily2018** and **daily2016** given the **15day** trajectory simulations.

```
conda activate uncertaindrift

python -m scripts.gendata.densitymaps.main <rootdir> daily2016 daily2018 15day --save-offsets --save-subsets
```

#### (3) Training fields

For the variables **velocity** and **ssh**, process the **raw** field data for training and output the data to a directory **train_raw** for the datasets **daily2016** and **daily2018**.

```
python -m scripts.gendata.process_glazur <rootdir> velocity raw train_raw daily2018 daily2016 --save-train
python -m scripts.gendata.process_glazur <rootdir> ssh raw train_raw daily2018 daily2016 --save-train
```

### Model training

Using the dataset **daily2018** and the simulation **15day**, train models for the **Emulator**, **SSH**, and **Forecast** scenarios to predict the **residual map**. By default, models are trained for 12 epochs.

Compute validation metrics for the datasets **daily2018** and **daily2016**. By default, metrics are computed every 12 epochs.

```
python -m scripts.train.snapshot <rootdir> Emulator daily2018 daily2018 daily2016 15day --residual-map
python -m scripts.train.snapshot <rootdir> SSH daily2018 daily2018 daily2016 15day --residual-map --field-name ssh --channels 3
python -m scripts.train.snapshot <rootdir> Forecast daily2018 daily2018 daily2016 15day --residual-map --t-only --channels 3
```

### Model evaluation

You may use the evaluation script to compute and record metrics for each sample individually. The call signatures below are identical to those above, with the exception of the argument that specifies the training set.

```
python -m scripts.eval.snapshot <rootdir> Emulator daily2018 daily2016 15day --residual-map
python -m scripts.eval.snapshot <rootdir> SSH daily2018 daily2016 15day --residual-map --field-name ssh --channels 3
python -m scripts.eval.snapshot <rootdir> Forecast daily2018 daily2016 15day --residual-map --t-only --channels 3
```
