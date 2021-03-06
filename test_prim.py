"""
https://docs.nvidia.com/deeplearning/dali/user-guide/docs/supported_ops.html?highlight=fn#module-nvidia.dali.fn
Test DALI pipeline functions

Crop
Rotate
Flip
Resize
Normalize
"""
import json
import nvidia
from nvidia.dali import pipeline_def, Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import math
import os
import statistics
from alexnet_pytorch import AlexNet
from random import shuffle
import types
import collections

dir = "/home/tristen/research/imagenet-mini/train"

def tensor_to_image(pp, path):

    input_batch = torch.tensor([3*[250*[250*[1]]],], dtype=torch.float32)
    nvidia.dali.plugin.pytorch.feed_ndarray(pp,input_batch)

    torchvision.utils.save_image(tensor=input_batch,fp=path)

def py_test(process, text, n_tests):
    
    times = []
    img_list = []
    c = 0
    for subdir, dirs, files in os.walk(dir):
        if c >= n_tests:
            break
        for filename in files:
            if c >= n_tests:
                break
            if(filename.endswith(".JPEG")):
                c += 1
                img_list.append(os.path.join(subdir, filename))

    
    for i in range(n_tests):
        
        t_start = timer()
        img = Image.open(img_list[i])
        # Preprocess image
        preprocess = transforms.Compose([  
        process,
        transforms.ToTensor()
        ])
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
        input_batch
        t = timer() - t_start
        times.append(t)
                
    t_min = min(times)
    t_max = max(times)
    t_avg = sum(times)/len(times)
    t_stddv = statistics.stdev(times)

    print(text + "\nMin: " + str(t_min) + "\nMax: " + str(t_max) + "\nAvg: " + str(t_avg) + "\nSTDDV: " + str(t_stddv))
                
    return t_min, t_max, t_avg, t_stddv

def py_test_norm(process, text, n_tests):

    times = []
    img_list = []
    c = 0
    for subdir, dirs, files in os.walk(dir):
        if c >= n_tests:
            break
        for filename in files:
            if c >= n_tests:
                break
            if(filename.endswith(".JPEG")):
                c += 1
                img_list.append(os.path.join(subdir, filename))

    for i in range(n_tests):
        
        t_start = timer()
        img = Image.open(img_list[i])
        # Preprocess image
        preprocess = transforms.Compose([  
        transforms.ToTensor(),
        process
        ])
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
        input_batch
        t = timer() - t_start
        times.append(t)

    t_min = min(times)
    t_max = max(times)
    t_avg = sum(times)/len(times)
    t_stddv = statistics.stdev(times)

    print(text + "\nMin: " + str(t_min) + "\nMax: " + str(t_max) + "\nAvg: " + str(t_avg) + "\nSTDDV: " + str(t_stddv))
                
    return t_min, t_max, t_avg, t_stddv

def py_test_cmn(process, text, n_tests):

    times = []
    img_list = []
    c = 0
    for subdir, dirs, files in os.walk(dir):
        if c >= n_tests:
            break
        for filename in files:
            if c >= n_tests:
                break
            if(filename.endswith(".JPEG")):
                c += 1
                img_list.append(os.path.join(subdir, filename))

    for i in range(n_tests):
        
        t_start = timer()
        img = Image.open(img_list[i])
        # Preprocess image
        preprocess = transforms.Compose([  
        transforms.CenterCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        process
        ])
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
        input_batch
        t = timer() - t_start
        times.append(t)

    t_min = min(times)
    t_max = max(times)
    t_avg = sum(times)/len(times)
    t_stddv = statistics.stdev(times)

    print(text + "\nMin: " + str(t_min) + "\nMax: " + str(t_max) + "\nAvg: " + str(t_avg) + "\nSTDDV: " + str(t_stddv))
                
    return t_min, t_max, t_avg, t_stddv

global pipe_img
def test(pipeline, text, batch, n_threads, n_tests):
    global pipe_img
    times = []
    if(text[0:3] == "CPU"):
        decode_pipe = decode_cpu(batch_size=batch, num_threads=6, device_id=0)

    else:
        decode_pipe = decode_gpu(batch_size=batch, num_threads=6, device_id=0)

    decode_pipe.build()
    
    decode_pipe.run()

    
    pipe = pipeline(batch_size=batch, num_threads=6, device_id=0)
        

    pipe.build()
    # warmup
    for _ in range(10):
        pipe.run()

    # test    
    for _ in range(n_tests):
        t_start = timer()
        
        pipe.run()
        
        t = timer() - t_start
        times.append(t)

    t_min = min(times)
    t_max = max(times)
    t_avg = sum(times)/len(times)
    t_stddv = statistics.stdev(times)

    print(text + "\n" + "Speed: {} imgs/s".format((n_tests * batch)/sum(times))
     + "\nMin: " + str(t_min) + "\nMax: " + str(t_max) + "\nAvg: " + str(t_avg) + "\nSTDDV: " + str(t_stddv))

    return t_min, t_max, t_avg, t_stddv

@pipeline_def
def decode_cpu():
    global pipe_img
    jpegs, labels = fn.readers.file(
        file_root=dir, random_shuffle=False)
    images = fn.decoders.image(jpegs, device='cpu')
    pipe_img = images
    return images, labels

@pipeline_def
def decode_gpu():
    global pipe_img
    jpegs, labels = fn.readers.file(
        file_root=dir, random_shuffle=False)
    images = fn.decoders.image(jpegs, device='mixed')
    pipe_img = images
    return images, labels

@pipeline_def
def crop_pipeline():
    global pipe_img
    
    images = fn.crop(
        pipe_img,
        crop=(0,0)
    )

    return images


@pipeline_def
def rotate_pipeline():
    global pipe_img
    
    images = fn.rotate(
        pipe_img,
        angle=10
    )
    
    return images

@pipeline_def
def flip_pipeline():
    global pipe_img
    
    images = fn.flip(
        pipe_img,
        horizontal=1
    )
    
    return images

@pipeline_def
def resize_pipeline():
    global pipe_img
    
    images = fn.resize(
        pipe_img,
        resize_x=256,
        resize_y=256)
    
    return images

@pipeline_def
def normalize_pipeline():
    global pipe_img
    
    images = fn.normalize(
        pipe_img,
        mean=0.5,
        stddev=0.5
    )
    
    return images

@pipeline_def
def crop_mirror_normalize_pipeline():
    global pipe_img
    
    images = fn.crop(
        pipe_img
    )
    images = fn.flip(
        images,
        horizontal=1
    )
    images = fn.normalize(
        images
    )
    
    return images

@pipeline_def
def cmn_combo_pipeline():
    global pipe_img
    
    images = fn.crop_mirror_normalize(
        pipe_img,
        crop=(0,0)
    )
    
    return images

class ExternalInputIterator(object):
    def __init__(self, batch_size):
        self.images_dir = dir
        self.batch_size = batch_size
        with open("file_list.txt", 'r') as f:
            self.files = [line.rstrip() for line in f if line != '']
        #shuffle(self.files)
       

    def __iter__(self):
        self.i = 0
        self.n = len(self.files)
        return self

    def __next__(self):
        batch = []
        labels = []
        for _ in range(self.batch_size):
            jpeg_filename = self.files[self.i]
            f = open(self.images_dir + "/" + jpeg_filename, 'rb')
            batch.append(np.frombuffer(f.read(), dtype = np.uint8))
            #labels.append(np.array([label], dtype = np.uint8))
            self.i = (self.i + 1) % self.n
        return (batch, labels)

def main():
    print("This is the beginning...")

    test_batch_size = 1
    n_tests = 32

    #PyTorch Transforms
    print("PyTorch Transforms")
    py_crop_min, py_crop_max, py_crop_avg, py_crop_stddv = py_test(transforms.RandomCrop((128,128)), "Crop: ",n_tests)
    print("-----")
    py_rotate_min, py_rotate_max, py_rotate_avg, py_rotate_stddv = py_test(transforms.RandomRotation(degrees=10), "Rotate: ",n_tests)
    print("-----")
    py_flip_min, py_flip_max, py_flip_avg, py_flip_stddv = py_test(transforms.RandomHorizontalFlip(), "Flip: ",n_tests)
    print("-----")
    py_resize_min, py_resize_max, py_resize_avg, py_resize_stddv = py_test(transforms.Resize(size=(256,256)), "Resize: ",n_tests)
    print("-----")
    py_norm_min, py_norm_max, py_norm_avg, py_norm_stddv = py_test_norm(transforms.Normalize((0.5), (0.5)), "Normalize: ",n_tests)
    print("-----")
    py_cmn_min, py_cmn_max, py_cmn_avg, py_cmn_stddv = py_test_cmn(transforms.Normalize(mean=(0.5), std=(0.5)),"CMn: ",n_tests)

    print("==============================\n")
    
    #DALI CPU and GPU individual primitives
    cpu_crop_min, cpu_crop_max, cpu_crop_avg, cpu_crop_stddv = test(crop_pipeline, "CPU Crop: ", test_batch_size, 4, n_tests)
    print("-----")
    gpu_crop_min, gpu_crop_max, gpu_crop_avg, gpu_crop_stddv = test(crop_pipeline, "GPU Crop: ", test_batch_size, 4, n_tests)

    print("==============================\n")
    
    cpu_rotate_min, cpu_rotate_max, cpu_rotate_avg, cpu_rotate_stddv = test(rotate_pipeline, "CPU Rotate: ", test_batch_size, 4, n_tests)
    print("-----")
    gpu_rotate_min, gpu_rotate_max, gpu_rotate_avg, gpu_rotate_stddv = test(rotate_pipeline, "GPU Rotate: ", test_batch_size, 4, n_tests)

    print("==============================\n")

    cpu_flip_min, cpu_flip_max, cpu_flip_avg, cpu_flip_stddv = test(flip_pipeline, "CPU Flip: ", test_batch_size, 4, n_tests)
    print("-----")
    gpu_flip_min, gpu_flip_max, gpu_flip_avg, gpu_flip_stddv = test(flip_pipeline, "GPU Flip: ", test_batch_size, 4, n_tests)

    print("==============================\n")

    cpu_resize_min, cpu_resize_max, cpu_resize_avg, cpu_resize_stddv = test(resize_pipeline, "CPU Resize: ", test_batch_size, 4, n_tests)
    print("-----")
    gpu_resize_min, gpu_resize_max, gpu_resize_avg, gpu_resize_stddv = test(resize_pipeline, "GPU Resize: ", test_batch_size, 4, n_tests)

    print("==============================\n")

    cpu_norm_min, cpu_norm_max, cpu_norm_avg, cpu_norm_stddv = test(normalize_pipeline, "CPU Normalize: ", test_batch_size, 4, n_tests)
    print("-----")
    gpu_norm_min, gpu_norm_max, gpu_norm_avg, gpu_norm_stddv = test(normalize_pipeline, "GPU Normalize: ", test_batch_size, 4, n_tests)

    print("==============================\n")

    cpu_crop_mirror_norm_min, cpu_crop_mirror_norm_max, cpu_crop_mirror_norm_avg, cpu_crop_mirror_norm_stddv = test(crop_mirror_normalize_pipeline, "CPU Crop,Flip,Normalize: ", test_batch_size, 4, n_tests)
    print("-----")
    gpu_crop_mirror_norm_min, gpu_crop_mirror_norm_max, gpu_crop_mirror_norm_avg, gpu_crop_mirror_norm_stddv = test(crop_mirror_normalize_pipeline, "GPU Crop,Flip,Normalize: ", test_batch_size, 4, n_tests)

    print("==============================\n")

    #DALI CPU and GPU combined primitives
    cpu_cmn_min, cpu_cmn_max, cpu_cmn_avg, cpu_cmn_stddv = test(cmn_combo_pipeline, "CPU CMn: ", test_batch_size, 4, n_tests)
    print("-----")
    gpu_cmn_min, gpu_cmn_max, gpu_cmn_avg, gpu_cmn_stddv = test(cmn_combo_pipeline, "GPU CMn: ", test_batch_size, 4, n_tests)
    
    #crop data
    data = [
        [py_crop_min, py_crop_max, py_crop_avg, py_crop_stddv],
        [cpu_crop_min, cpu_crop_max, cpu_crop_avg, cpu_crop_stddv],
        [gpu_crop_min, gpu_crop_max, gpu_crop_avg, gpu_crop_stddv],
    ]

    X = np.arange(4)
    fig, x = plt.subplots()
    x_labels = ["min","max","avg","stddv"]
    x.bar(X + 0.00, data[0], color = 'b', width = 0.25)
    x.bar(X + 0.25, data[1], color = 'r', width = 0.25, tick_label=x_labels)
    x.bar(X + 0.50, data[2], color = 'g', width = 0.25)

    x.set_ylabel("time (s)")
    x.set_title("Crop")
    
    colors = {"PyTorch":'blue', "CPU DALI":'red', "GPU DALI":'green'}
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    plt.legend(handles, labels)
    plt.plot()
    plt.savefig('charts/crop.png')
    
    #rotate data
    data = [
        [py_rotate_min, py_rotate_max, py_rotate_avg, py_rotate_stddv],
        [cpu_rotate_min, cpu_rotate_max, cpu_rotate_avg, cpu_rotate_stddv],
        [gpu_rotate_min, gpu_rotate_max, gpu_rotate_avg, gpu_rotate_stddv],
    ]

    X = np.arange(4)
    fig, x = plt.subplots()
    x_labels = ["min","max","avg","stddv"]
    x.bar(X + 0.00, data[0], color = 'b', width = 0.25)
    x.bar(X + 0.25, data[1], color = 'r', width = 0.25, tick_label=x_labels)
    x.bar(X + 0.50, data[2], color = 'g', width = 0.25)

    x.set_ylabel("time (s)")
    x.set_title("Rotate")
    
    colors = {"PyTorch":'blue', "CPU DALI":'red', "GPU DALI":'green'}
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    plt.legend(handles, labels)
    plt.plot()
    plt.savefig('charts/rotate.png')
    
    #flip data
    data = [
        [py_flip_min, py_flip_max, py_flip_avg, py_flip_stddv],
        [cpu_flip_min, cpu_flip_max, cpu_flip_avg, cpu_flip_stddv],
        [gpu_flip_min, gpu_flip_max, gpu_flip_avg, gpu_flip_stddv],
    ]

    X = np.arange(4)
    fig, x = plt.subplots()
    x_labels = ["min","max","avg","stddv"]
    x.bar(X + 0.00, data[0], color = 'b', width = 0.25)
    x.bar(X + 0.25, data[1], color = 'r', width = 0.25, tick_label=x_labels)
    x.bar(X + 0.50, data[2], color = 'g', width = 0.25)

    x.set_ylabel("time (s)")
    x.set_title("Flip")
    
    colors = {"PyTorch":'blue', "CPU DALI":'red', "GPU DALI":'green'}
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    plt.legend(handles, labels)
    plt.plot()
    plt.savefig('charts/flip.png')

    #Resize data
    data = [
        [py_resize_min, py_resize_max, py_resize_avg, py_resize_stddv],
        [cpu_resize_min, cpu_resize_max, cpu_resize_avg, cpu_resize_stddv],
        [gpu_resize_min, gpu_resize_max, gpu_resize_avg, gpu_resize_stddv],
    ]

    X = np.arange(4)
    fig, x = plt.subplots()
    x_labels = ["min","max","avg","stddv"]
    x.bar(X + 0.00, data[0], color = 'b', width = 0.25)
    x.bar(X + 0.25, data[1], color = 'r', width = 0.25, tick_label=x_labels)
    x.bar(X + 0.50, data[2], color = 'g', width = 0.25)

    x.set_ylabel("time (s)")
    x.set_title("Resize")
    
    colors = {"PyTorch":'blue', "CPU DALI":'red', "GPU DALI":'green'}
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    plt.legend(handles, labels)
    plt.plot()
    plt.savefig('charts/resize.png')

    #Normalize Data
    data = [
        [py_norm_min, py_norm_max, py_norm_avg, py_norm_stddv],
        [cpu_norm_min, cpu_norm_max, cpu_norm_avg, cpu_norm_stddv],
        [gpu_norm_min, gpu_norm_max, gpu_norm_avg, gpu_norm_stddv],
    ]

    X = np.arange(4)
    fig, x = plt.subplots()
    x_labels = ["min","max","avg","stddv"]
    x.bar(X + 0.00, data[0], color = 'b', width = 0.25)
    x.bar(X + 0.25, data[1], color = 'r', width = 0.25, tick_label=x_labels)
    x.bar(X + 0.50, data[2], color = 'g', width = 0.25)

    x.set_ylabel("time (s)")
    x.set_title("Normalize")
    
    colors = {"PyTorch":'blue', "CPU DALI":'red', "GPU DALI":'green'}
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    plt.legend(handles, labels)
    plt.plot()
    plt.savefig('charts/normalize.png')

    #Sequential Crop Mirror Normalize Data
    data = [
        [py_cmn_min, py_cmn_max, py_cmn_avg, py_cmn_stddv],
        [cpu_crop_mirror_norm_min, cpu_crop_mirror_norm_max, cpu_crop_mirror_norm_avg, cpu_crop_mirror_norm_stddv],
        [gpu_crop_mirror_norm_min, gpu_crop_mirror_norm_max, gpu_crop_mirror_norm_avg, gpu_crop_mirror_norm_stddv],
    ]

    X = np.arange(4)
    fig, x = plt.subplots()
    x_labels = ["min","max","avg","stddv"]
    x.bar(X + 0.00, data[0], color = 'b', width = 0.25)
    x.bar(X + 0.25, data[1], color = 'r', width = 0.25, tick_label=x_labels)
    x.bar(X + 0.50, data[2], color = 'g', width = 0.25)

    x.set_ylabel("time (s)")
    x.set_title("Sequential Crop Mirror Normalize")
    
    colors = {"PyTorch":'blue', "CPU DALI":'red', "GPU DALI":'green'}
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    plt.legend(handles, labels)
    plt.plot()
    plt.savefig('charts/seq_cmn.png')

    #Combined Crop Mirror Normalize Data
    data = [
        [py_cmn_min, py_cmn_max, py_cmn_avg, py_cmn_stddv],
        [cpu_cmn_min, cpu_cmn_max, cpu_cmn_avg, cpu_cmn_stddv],
        [gpu_cmn_min, gpu_cmn_max, gpu_cmn_avg, gpu_cmn_stddv],
    ]

    X = np.arange(4)
    fig, x = plt.subplots()
    x_labels = ["min","max","avg","stddv"]
    x.bar(X + 0.00, data[0], color = 'b', width = 0.25)
    x.bar(X + 0.25, data[1], color = 'r', width = 0.25, tick_label=x_labels)
    x.bar(X + 0.50, data[2], color = 'g', width = 0.25)

    x.set_ylabel("time (s)")
    x.set_title("Combined Crop Mirror Normalize")
    
    colors = {"PyTorch":'blue', "CPU DALI":'red', "GPU DALI":'green'}
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    plt.legend(handles, labels)
    plt.plot()
    plt.savefig('charts/comb_cmn.png')

    #plt.show()

if __name__ == "__main__":
    main()