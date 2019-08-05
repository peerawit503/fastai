#!/usr/bin/env python
# coding: utf-8

# # What Sneaker is That?
# 
# *Inspiration and foundation from Francisco Ingham and Jeremy Howard. Inspired by [Adrian Rosebrock](https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/)*

# Model deployed at https://what-sneaker.onrender.com/.




from fastai.vision import *
import matplotlib.pyplot as plt

path = Path('data/left_in')

classes = ['cd4487_100', 'cq4277_001', 'aq8296_100', 'aq0818_148', 'aa3834_101']
np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=224, num_workers=0).normalize(imagenet_stats)
data.classes
data.show_batch(rows=5, figsize=(12,12))
data.classes, data.c, len(data.train_ds), len(data.valid_ds)

bs = 64
np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=299, bs=bs//2, num_workers=0).normalize(imagenet_stats)
learn = create_cnn(data, models.resnet50, metrics=error_rate)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5)
learn.save('stage-1-50')
learn.unfreeze()
learn.fit_one_cycle(3, max_lr=slice(1e-6,1e-4))
learn.save('stage-2-50')
learn.load('stage-2-50')
learn.fit_one_cycle(5)
learn = create_cnn(data, models.resnet50, metrics=error_rate)
learn.fit_one_cycle(5)
learn.save('stage-3-50')
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()
learn.fit_one_cycle(6, max_lr=slice(1e-6,1e-5))
learn.save('stage-4-50')
learn.fit_one_cycle(6)
learn.save('stage-5-50')
# It looks like stage-4-50 had the lowest error rate at 0.089
learn.load('stage-4-50')
# Resnet50 results
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(12,12))
# Resnet50 results
interp.plot_top_losses(16, figsize=(15,11))


learn.export()
