from paderbox.notebook import *
import paderbox as pb
import padercontrib as pc
from padertorch.data.segment import get_segment_boundaries
from padertorch.data.utils import collate_fn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import random



def get_data_preparation(data, dataset, batch_size=10, shuffle=False):
    """Accessing audio stream from train stream"""
    data = data
    dataset = dataset
    mfcc = pb.transform.mfcc

    """Adding a activity dictionary to existing dataset"""
    def activity(dataset):
        get_activity = data.get_activity(dataset)
        dataset['activity'] = get_activity
        dataset['audio_path'] = dataset['audio_path']['observation']
        return dataset

    """Segment the audio in dataset"""
    def segmentation(dataset, chunk_size=4000):
        segment_audio = []
        if shuffle:
            boundaries = get_segment_boundaries(dataset['num_samples'], chunk_size, anchor='random')
        else:
            boundaries = get_segment_boundaries(dataset['num_samples'], chunk_size, anchor='left')
            
        for start,stop in boundaries:
            audio_chunk = dataset.copy()
            audio_chunk.update(audio_start = start)
            audio_chunk.update(audio_stop = stop)
            audio_chunk.update(label = int(any(dataset['activity'][start:stop])))
            segment_audio.append(audio_chunk)
        return segment_audio

    """Read the audio file"""

    def mfcc_feature(dataset):
        audio = dataset['audio_path']
        start = dataset['audio_start']
        stop = dataset['audio_stop']
        feature = mfcc(pb.io.load_audio(audio, start=start, stop=stop))
        dataset['features'] = feature.astype(np.float32)
        dataset['label'] = np.asarray(dataset['label']).astype(np.float32)
        return dataset

    """Keeping only needed dicitionary"""
    def new_dataset(dataset):
        dic = dict()
        dic['example_id'] = dataset['example_id']
        dic['features'] = np.expand_dims(dataset['features'], axis=0)
        dic['features_shape'] = dic['features'].shape
        dic['label'] = dataset['label']
        return dic

    """Stacking all the batch of features and class label to nparray"""
    def conv_list_nparray(dataset):
        dataset['features'] = np.stack(dataset['features'])
        dataset['label'] = np.vstack(dataset['label'])
        return dataset


    """ Mapping, shuffling, Prefetch, unbatch, batch_map, batch and collate_fn"""

    dataset = dataset.map(activity)
    dataset = dataset.map(segmentation)
    if shuffle:
        dataset = dataset.shuffle()
    dataset = dataset.prefetch(num_workers=8, buffer_size=8).unbatch()
    
    dataset = dataset.map(mfcc_feature)
    dataset = dataset.map(new_dataset)
    dataset = dataset.batch(batch_size).map(collate_fn)
    dataset = dataset.map(conv_list_nparray)
    
    return dataset
