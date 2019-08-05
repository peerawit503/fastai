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
# learn = create_cnn(data, models.resnet34, metrics=error_rate)
# learn.fit_one_cycle(4)
# learn.save('stage-1')
# learn.unfreeze()
# learn.lr_find()
# learn.recorder.plot()
# learn.fit_one_cycle(8, max_lr=slice(3e-5,3e-4))
# learn.recorder.plot_losses()
# learn.save('stage-2')
# learn.load('stage-2')
# interp = ClassificationInterpretation.from_learner(learn)
# interp.plot_confusion_matrix(figsize=(12,12))
# interp.plot_top_losses(16, figsize=(15,11))
# # Training Resnet50
# Resnet50 seemed to do terrible at first (0.8 error rate that wasn't decreasing). Then did unfreeze and trained some more. Still terrible. Recreated learn object, and did learn.fit_one_cycle(20), and now I'm up to around 93% accuracy! Not sure what changed from when I first ran it to when I reran and it started to work as expected.
bs = 64
np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=299, bs=bs//2, num_workers=0).normalize(imagenet_stats)
learn = create_cnn(data, models.resnet50, metrics=error_rate)
learn.lr_find()
# learn.recorder.plot()
# learn.fit_one_cycle(5)
# learn.save('stage-1-50')
# learn.unfreeze()
# learn.fit_one_cycle(3, max_lr=slice(1e-6,1e-4))
# learn.save('stage-2-50')
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
# ## Trying Out Resnet101
# learn2 = create_cnn(data, models.resnet101, metrics=error_rate)
# learn2.fit_one_cycle(5)
# learn2.save('stage-1-101')
# learn2.fit_one_cycle(5)
# learn2.save('stage-2-101')
# learn2.load('stage-2-101')
# # Resnet101 Results
# interp2 = ClassificationInterpretation.from_learner(learn2)
# interp2.plot_confusion_matrix(figsize=(12,12))
# # Resnet101 results
# interp2.plot_top_losses(16, figsize=(15,11))
# learn2.export()

# ## Putting your model in production

# We'll use the export for deploying on Render. The blog post I wrote covers the necessary steps.

# Resnet50 model
learn.export()
# path
# defaults.device = torch.device('cpu')
# img = open_image(path/'air_jordan_3'/'00000004.jpg')
# img
# # We create our `Learner` in production enviromnent like this, just make sure that `path` contains the file 'export.pkl' from before.
# learn = load_learner(path)

# pred_class,pred_idx,outputs = learn.predict(img)
# pred_class

# outputs

# pred = learn.predict(img)


# # The first two elements of the tuple are, respectively, the predicted class and label. 
# # Label here is essentially an internal representation of each class, since class name is a string and cannot be used in computation. To check what each label corresponds to, run:
# # 
# # learn.data.classes
# # ['3', '7']
# # So category 0 is 3 while category 1 is 7.
# # 
# # probs = pred[2]
# # The last element in the tuple is the predicted probabilities. For a categorization dataset, the number of probabilities returned is the same as the number of classes; probs[i] is the probability that the item belongs to learn.data.classes[i].



# pred

# pred[0]

# learn.data.classes

# # probabilities and then classes
# pred_result = pred[2].sort(descending=True)
# pred_result




# top_3_pred_probs = pred_result[0][:3]
# # convert probs to numpy array because I just want the numbers by themselves without 'tensor'
# top_3_pred_probs = top_3_pred_probs.numpy()

# top_3_pred_class_idxs = pred_result[1][:3]

# # Convert label from 'air_jordan_3' to 'Air Jordan 3' after looking up proper index
# top_3_pred_classes = [learn.data.classes[i].replace('_', ' ').title() for i in top_3_pred_class_idxs]

# print(top_3_pred_probs)
# print(top_3_pred_classes)

# pred_top_3_output = list(zip(top_3_pred_classes, top_3_pred_probs))
# print(pred_top_3_output)





# plt.figure(figsize=(8,8))
# df=pd.DataFrame({'allvarlist':top_3_pred_classes,'importances': top_3_pred_probs})
# df.sort_values('importances',inplace=True)
# df.plot(kind='barh',y='importances',x='allvarlist', legend=False, title='Top 3 Predicted Models');





# for i in pred_top_3_output:
#     print(i)

