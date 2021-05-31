import os
from pathlib import Path
import padertorch as pt
import padercontrib as pc

from padertorch.io import get_new_storage_dir
from padertorch.train.optimizer import SGD
from padertorch.train.trainer import Trainer
from sacred import Experiment, commands
from sacred.observers import FileStorageObserver
from padertorch.train.hooks import LRSchedulerHook
import torch

#from .model import Sad
from .Resnet_model import Sad
from .data import get_data_preparation

ex = Experiment("Speaker_Activity_Detection")

@ex.config
def config():
    
    subset = 'stream'
    debug = False
    batch_size = 10
    
    """Interactive trainer configuration parameters"""
        
    trainer = {
        "model": {
            'factory':Sad        
        },
        "storage_dir":get_new_storage_dir(
            'sad', id_naming='time', mkdir=False
        ),
        "optimizer": {
            "factory": SGD
        },
        'summary_trigger': (10_000, 'iteration'),
        'checkpoint_trigger': (44_975, 'iteration'),
        'stop_trigger': (4, 'epoch'),
        }

    trainer = Trainer.get_config(trainer)
    validation_metric = 'map'
    maximize_metric = True
    resume = False
    ex.observers.append(FileStorageObserver.create(trainer['storage_dir']))
        

        
@ex.automain
def main(_run, _log, trainer, database_json, subset, batch_size, validation_metric, maximize_metric, resume):
    commands.print_config(_run)
    trainer = Trainer.from_config(trainer)
    storage_dir = Path(trainer.storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    commands.save_config(_run.config, _log, config_filename=str(storage_dir/'config.json'))
        
        
    """Train and validation stream"""
    data = pc.database.fearless.Fearless()
    train_stream = data.get_dataset_train(subset=subset)
    validation_stream = data.get_dataset_validation(subset=subset)
    
    """Data preparation"""    
    training_data = get_data_preparation(data, train_stream, batch_size, shuffle=True)
    
    validation_data = get_data_preparation(data, validation_stream, batch_size)
    
    trainer.test_run(training_data, validation_data)
    """Learning rate decay"""
    trainer.register_hook(LRSchedulerHook(torch.optim.lr_scheduler.StepLR(trainer.optimizer.optimizer, step_size=3, gamma=0.98)))
    trainer.register_validation_hook(validation_data, metric=validation_metric, maximize=maximize_metric, early_stopping_patience=10)
    trainer.train(training_data, resume=resume)

