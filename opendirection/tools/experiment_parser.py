import configparser


class GetArgs:
    def __init__(self, experiment_config_path):
        self.spike_time_file = []
        self.spike_clusters_file = []
        self.cluster_groups_file = []
        self.sync_param_file = []
        self.dlc_files = []

        self.output_dir = []
        self.config_path = []
        self.summary_paths = []
        self.experiment_name = []

        self.verbose = False
        self.plot = False
        self.save = False

        self.get_experiment_config(experiment_config_path)

    def get_experiment_config(self, experiment_config):
        config = configparser.ConfigParser()
        config.read(experiment_config)

        self.spike_time_file = config["INPUT"]["SPIKE_TIMES"]
        self.spike_clusters_file = config["INPUT"]["SPIKE_CLUSTERS"]
        self.cluster_groups_file = config["INPUT"]["CLUSTER_GROUPS"]
        self.sync_param_file = config["INPUT"]["SYNC_PARAMETERS"]
        self.dlc_files = config.get("INPUT", "DLC_FILES").split(",")

        self.config_path = config["INPUT"]["CONFIG_DIRECTORY"]
        self.summary_paths = config["INPUT"]["SUMMARY_CONFIG_FILES"].split(",")
        if self.summary_paths == ["None"]:
            self.summary_paths = None
        self.experiment_name = config["INPUT"]["EXPERIMENT_NAME"]

        self.output_dir = config["OUTPUT"]["OUTPUT_DIR"]

        self.verbose = config["OPTIONS"].getboolean("VERBOSE")
        self.plot = config["OPTIONS"].getboolean("PLOT")
        self.save = config["OPTIONS"].getboolean("SAVE")
