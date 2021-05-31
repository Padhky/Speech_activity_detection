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
        
        """ResNet18 Layers"""
        self.layer1 = nn.Sequential(
                                    nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU())
        self.layer2 = nn.Sequential(
                                    nn.Conv2d(16,16, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(16,16, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(16))
        self.layer3 = nn.Sequential(
                                    nn.Conv2d(16,32, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(32,32, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(32),
                                    nn.MaxPool2d(kernel_size=2))
        
        self.layer4 = nn.Sequential(
                                    nn.Conv2d(32,64, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(64),
                                    nn.MaxPool2d(kernel_size=2))
        
        self.layer5 = nn.Sequential(
                                    nn.Conv2d(64,128, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(128),
                                    nn.MaxPool2d(kernel_size=2))

        
        """Bi-LSTM"""
        self.lstm = nn.Sequential(
                                   nn.LSTM(input_size=128,hidden_size=64,num_layers=2, batch_first=True, bidirectional=True),
                                   )
        self.drop_out = nn.Dropout(0.5)  # without dropout
        self.fc = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, batch_data):
        
        audio = batch_data['features']
        """ResNet18"""
        out = self.layer1(audio)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        #print(f"CON layer: {out.size()}")
        
        """1-d Pooling"""
        out = torch.mean(out, 3)
        """Transpose"""
        out = torch.transpose(out,1,2)
        
        """Bi-LSTM"""     
        out,_ = self.lstm(out)
        out = self.drop_out(out)
        out = self.fc(out[:,-1,:])
    
        """Sigmoid"""
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
                """Threshold equal to 0.1"""
                prediction_01 = (predictions>0.1).astype(int)
                """Threshold equal to 0.3"""
                #prediction_03 = (predictions>0.3).astype(int)
                
                TN,FP,FN,TP = confusion_matrix(targets, prediction_01).ravel()
                """Calcualting Detection Cost Function for 0.5 threshold"""
                FNR = FN/(FN+TP)
                FPR = FP/(FP+TN)
                Precision = TP/(TP+FP)
                Recall = TP/(TP+FN)
                
                summary['scalars']['map'] = metrics.average_precision_score(targets, predictions)
                summary['scalars']['mauc'] = metrics.roc_auc_score(targets, predictions)
                summary['scalars']['f1_score_threshold_01'] = metrics.f1_score(targets, prediction_01)
                #summary['scalars']['f1_score_threshold_03'] = metrics.f1_score(targets, prediction_03)
                summary['scalars']['mdcf'] = (0.75*FNR)+(0.25*FPR)
                
        for key, scalar in summary['scalars'].items():
            summary['scalars'][key] = np.mean(scalar)
        return summary