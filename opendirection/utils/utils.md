## Opendirection utils

### gen_velo_profile
**Generates sinusoidal velocity profiles with a fixed total displacement and 
specified maximum**

If the maximum value is not sufficient to reach the specified fixed
integral, then another sine wave will be appended.

If the maximum value is too high, and would overshoot the specified
integral, then the duration of the sine wave is reduced, and the
returned array will be zero in all other areas.
    
Usage:
`gen_velo_profile 60 -o /home/files/output.csv`

Command line inputs:

*Mandatory*
* Maximum value to be reached by the sine wave

*Optional*
* -o, --output  **Output file path (.csv)**
* --target-integral - **Specified area under the resulting function**
* --num-samples - **Number of timepoints to calculate over**
* --sample-duration - **How long (in seconds) is each sample**
* --plot - **Show plot**

Defaults etc can be found by running
`gen_velo_profile -h`