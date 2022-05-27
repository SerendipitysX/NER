"""
NNI hyperparameter optimization example.

Check the online tutorial for details:
https://nni.readthedocs.io/en/stable/tutorials/hpo_quickstart_pytorch/main.html
"""

from pathlib import Path
import signal

from nni.experiment import Experiment

# Define search space
search_space = {
    'alpha': {'_type': 'loguniform', '_value': [0.05, 1]},
    'gamma': {'_type': 'loguniform', '_value': [0.5, 3]},
    'w': {'_type': 'uniform', '_value': [0.1, 0.8]},
}

# Configure experiment
experiment = Experiment('local')
experiment.config.trial_command = 'python train.py'
experiment.config.trial_code_directory = Path(__file__).parent
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.max_trial_number = 20
experiment.config.trial_concurrency = 1

### earlystop
experiment.config.assessor.name = 'Medianstop'
experiment.config.tuner.class_args = {
    'optimize_mode': 'maximize',
    'start_step': 3
}


# Run it!
experiment.run(port=8080, wait_completion=False)

print('Experiment is running. Press Ctrl-C to quit.')
signal.pause()
