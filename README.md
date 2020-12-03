# Opendirection
#### For any questions, contact Adam Tyson ([adam.tyson@ucl.ac.uk](adam.tyson@ucl.ac.uk))

**Work in progress**


## Aim
Opendirection aims to correlate spike times and spatial behaviour. Spike times
are generated using [kilosort](https://github.com/cortex-lab/KiloSort),
 and animal body positions using 
[deeplabcut](https://github.com/AlexEMG/DeepLabCut).

Various parameters are calculated from the deeplabcut data (e.g. angular
head velocity) and cells are studied based on their responses
to these behavioural parameters.

## Installation
The easiest way currently is to set up a conda environment and then clone from
[this repository](https://github.com/adamltyson/opendirection)

#### Set up conda environment
Conda environments are used to isolate python versions and packages

* Download miniconda installation file 
[here](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)

* Run miniconda installation file, e.g.:


```bash
    bash ~/Downloads/Miniconda3-latest-Linux-x86_64.sh
```
* Create and activate new conda environment

```bash
    conda create --name opendirection python
```    

#### Download and install opendirection (e.g. into home)

```bash
    cd ~/
    git clone https://www.github.com/adamltyson/opendirection
    pip install opendirection/
    
```    

**N.B. if you want to modify the code, use the `-e` flag with pip:**

`pip install -e opendirection/`


## Usage
Opendirection runs from the command line, e.g.:
 
 ```bash
    opendirection -o output/ --plot --save --spiketimes spike_times.npy 
    --spikeclusters spike_clusters.npy --clustergroups cluster_groups.csv
    --syncparams sync_parameter.mat --dlcfiles dlc1.xlsx dlc2.xlsx
```    

Command line inputs:
* --spiketimes - **Spike timing file (.npy)**
* --spikeclusters - **Spike clusters file (.npy)**
* --clustergroups - **Cluster groups file (.csv)**
* --syncparams - **Synchronisation parameters (.mat)**
* --dlcfiles - **DeepLabCut .xlsx files**
* -o - **Output directory**
* --config - **Path to the directory containing the configuration files (conditions, config, options, etc)**
* --summary-config - **Paths to N summary configuration files (in the style of 
                        opendirection/options/summary_eg.ini**
* --experiment-name - **Name of the experiment**
* -V --verbose - **Increase verbosity for debug** 
(all debug info will save to log)
* -P --plot - **Show plots**
* -S --save - **Save results as .csv**
* -E --export - **Export results as a pickled object**

*Alternatively, a text file 
([such as this](https://github.com/adamltyson/opendirection/blob/master/options/experiment.ini))
 can be passed to run opendirection, e.g:*
 
 `opendirection /path/to/experiment.ini`
 
## Output
Plots will be generated, based on --plot flag and based on /options/options

* output/opendirection[*date*].log will be generated once per run
(useful for debugging)
* [CONDITION].csv will generated per condition, with columns representing
calculated statistics, for each cell (row)

## Input
There are four .ini files for configuration of how opendirection is run.
Defaults are supplied in /options, but external configuration files can 
be supplied via the command line.

### config.ini
**Likely to stay constant for each experimental setup**
N.B. All in S.I. units (e.g. seconds, Hz)
PROBE_SAMPLES_PER_SEC: Sampling frequency of electrophysiological probe 
(e.g. 25000)

CAMERA_FRAMES_PER_SEC: Sampling frequency of behavioural camera (e.g. 40)

METERS_PER_PIXEL: Calibration of camera (e.g. 0.002)

### conditions.ini
**May be different for experimental acqusition**

N.B. All in frames, and indexed from 0

Use this file to define multiple experimental conditions (e.g. trial vs 
control). The name of the condition is defined, along with stand and end times,
e.g.:

```ini
[CONDITION_NAME]
START = 0
END = 1500
; use comma separated ranges, e.g. "5500-5700, 12000-12500", or "None"
EXCLUDE =  200-250,320-330
```

### options.ini
**May be different for every run of opendirection**
N.B. Unless stated, all values are in S.I. units (metres, seconds, etc), but 
angles are in **degrees**

### summary.ini
**This file must be configured separately, and passed with `--summary-config`,
the file `options/summary_eg.ini` is just an example.

This file allows summary files to be generated which for each cell will:
* Put all parameters (across different conditions) together
* Allow cells to be chosen based on cross-condition criteria

Criteria sets must be of the form:

```ini
[CRTIERIA_NAME]
PARAMETER = parameter
VALUE = value
DIRECTION = direction (optional, default=higher, can be 'higher',
'lower' or 'equal'. All inclusive.)
```

e.g.:

```ini
[1]
CONDITION = LIGHT1
PARAMETER = mean_vec_length
VALUE = 0.1
DIRECTION = higher
```

As many of these as required may be specified to develop complex criteria.


In addition, iff a section entitled "SAVE_PARAMETERS" exists, then only the 
parameters listed therin will be saved to the summary csv file. e.g.:

``` ini
[SAVE_PARAMETERS] ; must be comma separated
BLOCK_1 = mean_vec_length,hd_snr
BLOCK_2 = pearson_neg_percentile, velocity_pearson_p
```

#### GENERAL
PARALLEL: Whether to run some calculations (e.g. AHV shuffling) in 
parallel ('yes'/'no')

N_FREE_CPUS: How many CPU cores to leave free for other programs (e.g 6)

#### CONDITIONS
**Conditions refer to (temporal) parts of the experiment that should be 
treated individually. Defined in options/conditions.ini**

CONDITIONS_LIST: Either 'all', or a comma separated list (no spaces) of 
conditions to analyse (e.g. 'condition1,condition2,condition4)

CELL_CONDITION_INCLUSION: If different cells would be included in the 
analysis for each condition (based on criteria in `CELL_SELECTION), 
should the cells analysed be:
* 'individual': use the cells that meet criteria in each condition. 
This potentially results in different number of cells per condition.
* 'any': use the cells that meet the criteria in any condition.
This is similar to an 'OR' Operation. The number of cells will be the same 
 across conditions.
* 'all': only use the cells that meet the criteria for all conditions.
This is similar to an 'AND' Operation. The number of cells will be the same 
 across conditions.
 
#### CELL_SELECTION

MIN_FIRE_FREQ: Minimum overall firing frequency (e.g. 1)

MIN_HZ_IN_ANY_DIRECTION_BIN: Minimum firing frequency in any head direction
bin (e.g. 2)

#### BEHAVIOUR_SELECTION

SPEED_CUT_OFF: Frames in which the animal is moving slower than this 
are not analysed (e.g. 0.02). N.B. This is a global parameter, and so the 
highest of this and any analysis-type specific options 
(e.g `AHV_SPEED_CUT_OFF`) will be used for each analysis type.


#### HEAD_DIRECTION
HD_SPEED_CUT_OFF: Frames in which the animal is moving slower than this 
are not analysed (e.g. 0.02). N.B. This an analysis-type specific option, and 
the highest of this and `SPEED_CUT_OFF` will be used for this analysis type.

FILTER_HEAD_DIRECTION: Should the raw head direction signal be smoothed prior
to other calculations ('yes'/'no'). This is the smoothing of the raw data
with a median filter which will affect both plotting and stats.

HEAD_DIRECTION_MED_FILT_WIDTH: Head direction medial filter width (e.g. 0.4),
in seconds.

DIRECTION_BIN: Head direction binning (e.g. 6), in degrees

BASELINE_BIN: Lowest (non-continuous) X deg to choose as baseline firing
(e.g. 60)

HD_HIST_SMOOTH: Should the (overall) head direction histogram be 
smoothed? ('yes'/'no). This affects anything that the histogram is used for
(both stats at plotting).  

HD_HIST_SMOOTH_WIDTH: Head direction histogram smoothing sigma (e.g. 5). this is 
in degrees but the smoothing is done in number of bins. So you should consider the 
head direction bin size when setting this parameter. The final number of bins to 
smooth will be rounded. To see the effect of smoothing a value greater than 
2*HD-bin-size should be used.

HD_KDE_KAPPA: [Von Mises distribution](https://en.wikipedia.org/wiki/Von_Mises_distribution)
kappa parameter used for head direction Kernel Density Estimatation (e.g. 20).

#### ANGULAR_HEAD_VELOCITY
AHV_SPEED_CUT_OFF: Frames in which the animal is moving slower than this 
are not analysed (e.g. 0.02). N.B. This an analysis-type specific option, and 
the highest of this and `SPEED_CUT_OFF` will be used for this analysis type.

LOCAL_DERIVATIVE_WINDOW: Temporal head direction window to calculate angular
head velocity (e.g. 0.2), in seconds.

ANG_VEL_BIN_SIZE: Angular head velocity binning in deg/s (e.g. 6)

CALCULATE_CENTRAL_AHV: Automatically calculate the allowed (low magnitude)
angular head velocity values allowed ('yes'/'no')

CENTRAL_AHV_FRACTION: If calculating the allowed angular head velocity values 
allowed, what fraction to keep (e.g. 0.9)

MAX_AHV: If not calculating the allowed angular head velocity values,
 specify the maximum magnitude (e.g. 300)


#### VELOCITY
VELOCITY_SPEED_CUT_OFF: Frames in which the animal is moving slower than this 
are not analysed (e.g. 0.02). N.B. This an analysis-type specific option, and 
the highest of this and `SPEED_CUT_OFF` will be used for this analysis type.

VELOCITY_POSITION_HEAD: Use the head position of the mouse for calculation of 
locomotionvelocity. Defaults to the body otherwise.

FILTER_VELOCITY: Should the raw velocity signal be smoothed prior
to other calculations ('yes'/'no')

VELOCITY_MED_FILT_WIDTH: Velocity median filter width (e.g. 0.4)

VELOCITY_BIN_SIZE: Velocity binning in m/s (e.g. 0.05)

MAX_VELOCITY: Maximum velocity value included in the calculations (e.g. 0.5)

#### PLACE
PLACE_SPEED_CUT_OFF: Frames in which the animal is moving slower than this 
are not analysed (e.g. 0.02). N.B. This an analysis-type specific option, and 
the highest of this and `SPEED_CUT_OFF` will be used for this analysis type.

SPATIAL_POSITION_HEAD: Use the head position of the mouse for anything to do with 
body position (including plotting). Defaults to the body otherwise.

SPATIAL_BIN_SIZE = Spatial binning in m (e.g. 0.05)

MIN_TIME_IN_SPATIAL_BIN = How long must each spatial bin be sampled 
for (e.g. 5)

PLOT_SPACE_FIRING_SMOOTH: Smooth the spatial firing rates in space

PLOT_SPACE_FIRING_SMOOTH_WIDTH: How much to smooth the firing rate bins.


#### PLOTTING

HIST_OR_KDE:
* 'hist': Plot histograms only
* 'kde': Plot Kernel Density Estimatation only
* 'both' Plot both

HEAD_DIRECTION: Plot overall head directions sampled ('yes'/'no')

CELL_DIRECTION_OVERLAY: Plot the head direction tuning of each cell, overlaid  
('yes'/'no')

CELL_DIRECTION_SUBPLOT: Plot the head direction tuning of each cell,
in a row ('yes'/'no')

ALL_BEHAVIOUR: Plot all raw behaviour (speed, head direction and angular head
velocity) ('yes'/'no')

PLOT_RAW_SPIKES: Plot raw spiking of each cell  ('yes'/'no'). N.B. as a bug
 workaround, spikes at t=0 will not be plotted

PLOT_VELOCITY: Plot all velocity as a histogram  ('yes'/'no')

PLOT_VELOCITY_LOG: Plot velocity histogram on a log scale  ('yes'/'no') 

PLOT_VELOCITY_FIRING_RATE: Plot animal velocity tuning of cells  ('yes'/'no')

PLOT_VELOCITY_REMOVE_ZEROS: Don't plot bins with zero spikes ('yes'/'no')

PLOT_ANGULAR_VELOCITY: Plot all angular velocity as a histogram  ('yes'/'no')

PLOT_ANG_VEL_LOG: Plot angular velocity histogram on a log scale  ('yes'/'no') 

PLOT_AHV_FIRING_RATE: Plot angular head velocity tuning of cells  ('yes'/'no')

PLOT_AHV_REMOVE_ZEROS: Don't plot bins with zero spikes ('yes'/'no')

PLOT_AHV_FIT: Plot the linear fits for positive and negative AHV bins('yes'/'no')

PLOT_TRAJECTORY: Plot trajectory of mouse  ('yes'/'no')

PLOT_SPATIAL_OCCUPANCY = Plot the overall occupancy of each spatial bin

PLOT_SPACE_FIRING = Plot the firing rate in each bin

MEDIAN_FILTER_BEHAVIOUR_PLOT: Median filter raw behaviour before plotting
 ('yes'/'no')

FILTER_WIDTH_BEHAVIOUR_PLOT: Median filter width for raw behaviour plotting
(e.g. 1)


#### DISPLAY
PALETTE: Any seaborn default palette (e.g. 'husl')

PLOT_TRANSPARENCY = For aesthetics, base transparency value for plots
 (e.g. 0.7)


#### STATISTICS

HD_SHUFFLE_TEST: Run head directionality testing on each cell ('yes'/'no').
Turn off if just plots are needed etc.

HD_SHUFFLE_MIN_MAGNITUDE: Minimum shuffling magnitude. (How many seconds to 
shift the spike train by (e.g. 20).) Note this is both + and - and in seconds.

HD_SHUFFLE_MAX_MAGNITUDE: Maximum shuffling magnitude. (How many seconds to 
shift the spike train by (e.g. 20).) Note this is both + and - and in seconds.
**N.B. Set to 0 to force the maximum to be the total duration of the condition 
minus HD_SHUFFLE_MIN_MAGNITUDE**

HD_SHUFFLE_ITERATIONS: How many spike train shuffling iterations to perform
 (e.g. 1000)


CALC_HD_PEAK_METHOD: Calculate peak of HD cell firing based on mean
firing direction or histogram ('mean'/'hist', 'hist'/'kde').

CALC_HD_HIST_SMOOTH: If using histogram, whether it should be smoothed
('yes'/'no')

CALC_HD_HIST_SMOOTH_WIDTH: Head direction histogram smoothing sigma for peak
calculation integer (e.g. 5)


AHV_SHUFFLE_TEST = Run angular head velocity testing on each cell ('yes'/'no').
Turn off if just plots are needed etc.

AHV_SHUFFLE_MIN_MAGNITUDE: Minimum shuffling magnitude. (How many seconds to 
shift the spike train by (e.g. 20).) Note this is both + and - and in seconds.

AHV_SHUFFLE_MAX_MAGNITUDE: Maximum shuffling magnitude. (How many seconds to 
shift the spike train by (e.g. 20).) Note this is both + and - and in seconds.
**N.B. Set to 0 to force the maximum to be the total duration of the condition 
minus AHV_SHUFFLE_MIN_MAGNITUDE**

AHV_SHUFFLE_ITERATIONS: How many spike train shuffling iterations to perform
 (e.g. 1000)
 
AHV_CORRELATION_MAGNITUDE: Should the percentile just use the magnitude
of correlation ('yes'/'no')?

VELOCITY_SHUFFLE_TEST = Run velocity testing on each cell ('yes'/'no').
Turn off if just plots are needed etc.

VELOCITY_SHUFFLE_MIN_MAGNITUDE: Minimum shuffling magnitude. (How many seconds to 
shift the spike train by (e.g. 20).) Note this is both + and - and in seconds.

VELOCITY_SHUFFLE_MAX_MAGNITUDE: Maximum shuffling magnitude. (How many seconds to 
shift the spike train by (e.g. 20).) Note this is both + and - and in seconds.
**N.B. Set to 0 to force the maximum to be the total duration of the condition 
minus VELOCITY_SHUFFLE_MIN_MAGNITUDE**

VELOCITY_SHUFFLE_ITERATIONS: How many spike train shuffling iterations to perform
 (e.g. 1000)

AHV_CORRELATION_MAGNITUDE: Should the percentile just use the magnitude
of correlation ('yes'/'no')?

PLACE_SHUFFLE_TEST = Run place testing on each cell ('yes'/'no').
Turn off if just plots are needed etc.

PLACE_SHUFFLE_MIN_MAGNITUDE: Minimum shuffling magnitude. (How many seconds to 
shift the spike train by (e.g. 20).) Note this is both + and - and in seconds.

PLACE_SHUFFLE_MAX_MAGNITUDE: Maximum shuffling magnitude. (How many seconds to 
shift the spike train by (e.g. 20).) Note this is both + and - and in seconds.
**N.B. Set to 0 to force the maximum to be the total duration of the condition 
minus PLACE_SHUFFLE_MIN_MAGNITUDE**

PLACE_SHUFFLE_ITERATIONS: How many spike train shuffling iterations to perform
 (e.g. 1000)

#### OUTPUT

