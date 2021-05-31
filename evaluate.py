from padertorch.contrib.jensheit.eval_sad import get_tp_fp_tn_fn as TF
from pathlib import Path
import numpy as np
import torch
import padertorch as pt
import padercontrib as pc
from padertorch import Model
from sacred import Experiment, commands
from sklearn import metrics
from tqdm import tqdm
from paderbox.io.new_subdir import get_new_subdir
from paderbox.io import load_json, dump_json
from pprint import pprint
from sklearn.metrics import confusion_matrix
from .data import get_data_preparation
from paderbox.io.audiowrite import dump_audio

ex = Experiment('SAD_Evaluate')
@ex.config
def config():
    exp_dir = ''
    assert len(exp_dir) > 0, 'Set the model path on the command line.'
    storage_dir = str(get_new_subdir(
        Path(exp_dir) / 'eval', id_naming='time', consider_mpi=True
    ))
    database_json = load_json(Path(exp_dir) / 'config.json')["database_json"]
    subset = 'stream'
    batch_size = 1
    device = 0
    ckpt_name = 'ckpt_best_map.pth'
    
    
@ex.automain
def main(_run, exp_dir, storage_dir, database_json, ckpt_name, subset, batch_size, device):
    
    commands.print_config(_run)

    exp_dir = Path(exp_dir)
    storage_dir = Path(storage_dir)

    config = load_json(exp_dir / 'config.json')

    model = Model.from_storage_dir(
        exp_dir, consider_mpi=True, checkpoint_name=ckpt_name
    )
    model.to(device)
    model.eval()    
    data = pc.database.fearless.Fearless()
    validation_stream = data.get_dataset_validation(subset=subset)
    validation_data = get_data_preparation(data, validation_stream, batch_size)
    
    with torch.no_grad():
        metric = {'threshold':[], 'DCF': [], 'F1': []}
        F1 = []
        DCF = []
        Threshold = []
        num_threshold = 11
        estimate = []        
        """All prediction are appended into the estiamte list """
        for example in tqdm(validation_data):
            example = model.example_to_device(example, device)
            output = model(example)
            pred = np.squeeze(output['prediction']).cpu().detach().numpy()
            estimate.extend(4000*[pred])
        
        
            
        for i,threshold in enumerate(np.linspace(0,1,num_threshold)):
            start = 0
            TP = []
            FP = []
            TN = []
            FN = []
            for k in tqdm(range(len(validation_stream))):
                activity = data.get_activity(validation_stream[k])
                target_length = len(activity)            
                #direc = f'/net/vol/k10u/project_pad/models/FS02_dev_00{j+1}.wav'
                end = start+target_length
                """Padding zeroes --> The dataset which are not divisible by 4000"""
                if target_length == 14955294:
                    end = end-3429   #Padding zeros
                    output = np.array(estimate[start:end])
                    output.resize(target_length)
                    start = end
                else:
                    output = np.array(estimate[start:end])
                    start = end
                    
                tn,fp,fn,tp = confusion_matrix(np.array((activity[:]).astype(int)), np.array((output>threshold).astype(int))).ravel()
                TP.append(tp)
                FP.append(fp)
                TN.append(tn)
                FN.append(fn)
                                   
            Precision = sum(TP)/(sum(TP)+sum(FP))
            Recall = sum(TP)/(sum(TP)+sum(FN))    
            FNR = sum(FN)/(sum(FN)+sum(TP))
            FPR = sum(FP)/(sum(FP)+sum(TN))
            f1_score = 2*(Precision*Recall) / (Precision+Recall) 
            dcf_metric = (0.75*FNR)+(0.25*FPR)
            
            
            F1.append(f1_score)
            DCF.append(dcf_metric)
            Threshold.append(threshold)
        """Sortinf metrics""""
        sorting = np.argsort(DCF)
        metric["threshold"] = np.array(Threshold)[sorting]
        metric["F1"] = np.array(F1)[sorting]
        metric["DCF"] = np.array(DCF)[sorting]
        
        dump_json( metric, storage_dir/'overall.json', indent=4, sort_keys=False)

    pprint(metric)
        