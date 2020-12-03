# Multi-experiment null distributions

To calculate the null distributions from all cells from multiple experiments,
use `opendirection_multiexp`.

`opendirection_multiexp` takes a directory of experiment files as input:
```bash
opendirection_multiexp /path/to/dir_of_exp_files /path/to/options.ini /path/to/output_dir --conditions LIGHT DARK
```

Where `dir_of_exp_files` contains e.g.:
```bash
experiment_0.txt
experiment_1.txt
experiment_2.txt
experiment_3.txt
experiment_4.txt
```

And the `--conditions` flag is followed by the name of the conditions included 
in the analysis (as defined in the `conditions.ini` file)

You can also use the following options:
* `--verbose` Increase the level of log messages for debugging
* `--n-free-cpus` How many CPU cores to leave free