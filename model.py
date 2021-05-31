import padertorch
from padertorch.base import Model
import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import padertorch
import numpy as np

class Sad(Model):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
                                nn.Conv2d(1,4,kernel_size=3),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(kernel_size=2),
            
                                nn.Conv2d(4,16, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(16,32, kernel_size=3, padding=1),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(32,64,kernel_size=3,padding=1),
                                nn.ReLU(inplace=True),
            
                                nn.Conv2d(64,128,kernel_size=3),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(kernel_size=2),
                                       )
        self.linear_layer = nn.Sequential(nn.Linear(128*4*1, 1))
        self.sigmoid = nn.Sigmoid()


    def forward(self, batch_data):
        
        audio = batch_data['features']
        out = self.net(audio)
        out = out.view(out.size(0), -1)
        out = self.linear_layer(out)
        out = self.sigmoid(out)
        
        return dict(prediction = out)
        
    def review(self, batch_data, outputs):
        loss = nn.BCELoss(reduction='none')(outputs["prediction"], batch_data['label'])
        review = dict(
                        loss = loss.mean(),
                        buffers = dict(
                                        prediction = outputs['prediction'].data.cpu().numpy(),
                                        target = batch_data['label'].data.cpu().numpy()
                                      ),
                        scalars = dict()
                        )
        return review
    
    
    def modify_summary(self, summary):
        if 'prediction' in summary['buffers']:
            predictions = np.concatenate(summary['buffers'].pop('prediction'))
            targets = np.concatenate(summary['buffers'].pop('target'))
            if (targets.sum(0) > 1).all():
                """Threshold equal to 0.5"""
                prediction_05 = (predictions>0.5).astype(int)
                """Threshold equal to 0.3"""
                prediction_03 = (predictions>0.3).astype(int)
                
                TN,FP,FN,TP = confusion_matrix(targets, prediction_05).ravel()
                """Calcualting Detection Cost Function for 0.5 threshold"""
                FNR = FN/(FN+TP)
                FPR = FP/(FP+TN)
                Precision = TP/(TP+FP)
                Recall = TP/(TP+FN)
                
                summary['scalars']['map'] = metrics.average_precision_score(targets, predictions)
                summary['scalars']['mauc'] = metrics.roc_auc_score(targets, predictions)
                summary['scalars']['f1_score_threshold_05'] = metrics.f1_score(targets, prediction_05)
                summary['scalars']['f1_score_threshold_03'] = metrics.f1_score(targets, prediction_03)
                summary['scalars']['mdcf'] = (0.75*FNR)+(0.25*FPR)
                
        for key, scalar in summary['scalars'].items():
            summary['scalars'][key] = np.mean(scalar)
        return summary