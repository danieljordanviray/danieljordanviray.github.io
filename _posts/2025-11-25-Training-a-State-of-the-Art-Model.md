![png](/images/112525_output_36_0.png)

```python
! pip install -Uqq fastbook
import fastbook
fastbook.setup_book()
```


```python
from fastbook import *
```

# Training a State-of-the-Art Model

Just completed my next project in learning how to build and engineer AI systems! In this project, I learned about techniques to train state-of-the-art AI computer vision models. Check it out here!

What excites you most about AI and AI engineering?

In this project, I learned about techniques to train state-of-the-art AI computer vision models: data normalization, progressive resizing, test-time augmentation, mixup, and label smoothing. In plain english, this means:

- Scaling the input data appropriately (data normalization)

- Training on smaller images first, and then large images later (progressive resizing)

- Augmenting test data for better generalized predictions (test-time augmentation)

- Combining / Blending images for data input (mixup)

- Adding uncertaintly to ground-truth data labels  (label smoothing)

## Imagenette


```python
# download imagenette data set (smaller subset of ImageNet)
from fastai.vision.all import *
path = untar_data(URLs.IMAGENETTE)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>





<div>
  <progress value='1557168128' class='' max='1557161267' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [1557168128/1557161267 00:26&lt;00:00]
</div>




```python
# create datablock and dataloaders
dblock = DataBlock(blocks=(ImageBlock(), CategoryBlock()),
                   get_items=get_image_files,
                   get_y=parent_label,
                   item_tfms=Resize(460),
                   batch_tfms=aug_transforms(size=224, min_scale=0.75))
dls = dblock.dataloaders(path, bs=64)
```


```python
# obtain baseline accuracy for a model
model = xresnet50(n_out=dls.c) # xresnet is fastai's modified version of ResNet50 # n_out = # of output classes in dls
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy) # CrossEntropyLossFlat is fastai's wrapper around Cross Entropy (reshaped tensors)
learn.fit_one_cycle(5, 3e-3) # no transfer learning or freezing layers, fit_one_cycle trains from scratch
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.601769</td>
      <td>3.243712</td>
      <td>0.331591</td>
      <td>01:03</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.194669</td>
      <td>1.683060</td>
      <td>0.529873</td>
      <td>01:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.955189</td>
      <td>1.310476</td>
      <td>0.603062</td>
      <td>01:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.750074</td>
      <td>0.702119</td>
      <td>0.777819</td>
      <td>01:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.603335</td>
      <td>0.544177</td>
      <td>0.824496</td>
      <td>01:01</td>
    </tr>
  </tbody>
</table>


## Normalization


```python
x,y = dls.one_batch()
```


```python
x.shape # 64 images, 3 channels, 224px width x 224 px height
```




    torch.Size([64, 3, 224, 224])




```python
x[0]
```




    TensorImage([[[0.1719, 0.2471, 0.2250,  ..., 0.1860, 0.2026, 0.2253],
                  [0.0831, 0.1762, 0.2253,  ..., 0.2179, 0.2462, 0.2645],
                  [0.0488, 0.0623, 0.1283,  ..., 0.2727, 0.2603, 0.2350],
                  ...,
                  [0.1088, 0.1284, 0.1177,  ..., 0.9208, 0.9268, 0.9292],
                  [0.1095, 0.1131, 0.1177,  ..., 0.9294, 0.9370, 0.9218],
                  [0.0878, 0.0886, 0.0945,  ..., 0.9038, 0.9419, 0.9519]],
    
                 [[0.1504, 0.2194, 0.1932,  ..., 0.1773, 0.1897, 0.2087],
                  [0.0993, 0.1628, 0.1965,  ..., 0.1923, 0.2148, 0.2419],
                  [0.0737, 0.0727, 0.1241,  ..., 0.2289, 0.2210, 0.2130],
                  ...,
                  [0.0764, 0.0903, 0.0902,  ..., 0.8081, 0.8124, 0.8140],
                  [0.0748, 0.0802, 0.0892,  ..., 0.8184, 0.8166, 0.7988],
                  [0.0736, 0.0785, 0.0801,  ..., 0.8087, 0.8341, 0.8400]],
    
                 [[0.0994, 0.1519, 0.1210,  ..., 0.2052, 0.2265, 0.2176],
                  [0.0783, 0.1181, 0.1255,  ..., 0.2039, 0.2304, 0.2336],
                  [0.0552, 0.0674, 0.1106,  ..., 0.2209, 0.2046, 0.2009],
                  ...,
                  [0.0732, 0.0857, 0.0754,  ..., 0.6161, 0.6077, 0.6067],
                  [0.0788, 0.0790, 0.0714,  ..., 0.6106, 0.6062, 0.5811],
                  [0.0726, 0.0750, 0.0618,  ..., 0.5675, 0.6218, 0.6387]]], device='cuda:0')




```python
y
```




    TensorCategory([6, 6, 8, 1, 1, 0, 4, 4, 2, 5, 2, 5, 2, 9, 3, 6, 8, 8, 7, 1, 8, 0, 2, 7, 7, 1, 9, 8, 6, 0, 3, 5, 6, 2, 8, 0, 3, 7, 3, 0, 0, 3, 2, 0, 2, 0, 1, 0, 3, 9, 3, 3, 6, 0, 1, 9, 3, 7, 8, 2, 5,
                    4, 9, 4], device='cuda:0')




```python
# avg over 64 images, avg over height, avg over width
# left with avg [R, G, B] value across all 64 image in the mini-batch
x.mean(dim=[0,2,3])
```




    TensorImage([0.4486, 0.4347, 0.3975], device='cuda:0')




```python
# same thing with std dev
x.std(dim=[0,2,3])
```




    TensorImage([0.2861, 0.2709, 0.2795], device='cuda:0')



We want to normalize pixel values with a mean of 0 and standard dev of 1.


```python
def get_dls(bs, size): # bs = batch size, size = height/width of square image (in px)
    dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                   get_items=get_image_files,
                   get_y=parent_label,
                   item_tfms=Resize(460),
                   batch_tfms=[*aug_transforms(size=size, min_scale=0.75),
                               Normalize.from_stats(*imagenet_stats)]) # added Normalize function to batch_tfms, takes imagenet normalization statistics, mean = 0 and std dev = 1
    return dblock.dataloaders(path, bs=bs)
```


```python
dls = get_dls(64, 224)
```


```python
x,y = dls.one_batch()
x.mean(dim=[0,2,3]),x.std(dim=[0,2,3])
```




    (TensorImage([-0.1592, -0.0580,  0.0059], device='cuda:0'),
     TensorImage([1.2053, 1.1977, 1.2667], device='cuda:0'))



Mean is now ~0, Std Dev is now ~1 across the channels


```python
# check accuracy of model now with the normalization process
model = xresnet50(n_out=dls.c)
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
learn.fit_one_cycle(5, 3e-3)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.623628</td>
      <td>2.986832</td>
      <td>0.348021</td>
      <td>01:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.235725</td>
      <td>1.177322</td>
      <td>0.622106</td>
      <td>01:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.942790</td>
      <td>1.284812</td>
      <td>0.602689</td>
      <td>01:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.738023</td>
      <td>0.717642</td>
      <td>0.761016</td>
      <td>01:01</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.594130</td>
      <td>0.593817</td>
      <td>0.809186</td>
      <td>01:01</td>
    </tr>
  </tbody>
</table>


Only helped a little in this case. But it is important for transfer learning. You have to use the same normalized statistics that the model was trained on.

In previous projects, the vision_learner constructer applied the normalization for us. But we trained from scratch here, so we had to add ImageNet normalization statistics.

## Progressive Resizing

Idea: start training with small images and the progress up to training with large images.

Why?

Training on small images for earlier layers (general image principles) speeds up training, training on larger images in later layers (more specific image attributes) improves accuracy.

Progressive re-sizing is also another form of data augmentation.


```python
# trains from scratch
dls = get_dls(128, 128) # bs = 128 images, size of images in tensors = 128 px (i.e., 128px x 128px).
learn = Learner(dls, xresnet50(n_out=dls.c), loss_func=CrossEntropyLossFlat(), 
                metrics=accuracy)
learn.fit_one_cycle(4, 3e-3)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.902943</td>
      <td>2.447006</td>
      <td>0.401419</td>
      <td>00:30</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.315203</td>
      <td>1.572992</td>
      <td>0.525765</td>
      <td>00:30</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.001199</td>
      <td>0.767886</td>
      <td>0.759149</td>
      <td>00:30</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.765864</td>
      <td>0.665562</td>
      <td>0.797984</td>
      <td>00:30</td>
    </tr>
  </tbody>
</table>



```python
learn.dls = get_dls(64, 224) 
# replaces the DataLoaders inside the Learner class / learn object
# all parameters (weights and biases) remain the same, now we're just changing the underlying training data
# the parameters are stored inside the learn object, not the dls, that's why we can replaced the dls (the underlying data)
# CNNs don't need a fixed input size, # of output nodes remain the same = the number of output categories


learn.fine_tune(5, 1e-3)
# phase 1 - freezes all layers except head for 1 epoch (trains head for 1 epoch), unless (freeze_epochs is specified)
# phase 2 - unfreezes all layers, trains for 5 epochs, applies discriminative learning rates internally
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.665387</td>
      <td>0.745566</td>
      <td>0.764376</td>
      <td>01:00</td>
    </tr>
  </tbody>
</table>




<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.551115</td>
      <td>0.567658</td>
      <td>0.825616</td>
      <td>01:01</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.561163</td>
      <td>0.606796</td>
      <td>0.803958</td>
      <td>01:01</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.496560</td>
      <td>0.458967</td>
      <td>0.849515</td>
      <td>01:01</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.462095</td>
      <td>0.431930</td>
      <td>0.858476</td>
      <td>01:02</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.411312</td>
      <td>0.414647</td>
      <td>0.866318</td>
      <td>01:01</td>
    </tr>
  </tbody>
</table>


Inside learn, there is:

learn.model → stores all the weights & biases

learn.dls → stores the DataLoaders

learn.opt → optimizer

learn.loss_func → loss function

learn.metrics → metrics

This is called self-transfer learning. We're training the model on small images to recognize base patterns common in all images (for speed and cost), saving those parameters, then switching the underlying data to large images, and fine-tuning the model on the larger images for accuracy. The fine-turning trains the head of the model, which detects more complex and image-specific patterns.

## Test Time Augmentation

The idea behind test time augmentation is augmenting the validation data set so that the model makes predictions on generalized forms of the images.

fast.ai by default takes center crop of validation images, but this can be problematic. For example, if there are multiple objects in the image, or if there are important features in an image that are cropped out. 

tta will take k data augmented crops and then take the average or max to make a prediction.

Mathematically:


```python
# assume three classes: cat, dog, bird

# get predictions for k-4 data augmented validation images:

    # Prediction 1: [0.20, 0.70, 0.10]
    # Prediction 2: [0.30, 0.60, 0.10]
    # Prediction 3: [0.10, 0.80, 0.10]
    # Prediction 4: [0.25, 0.65, 0.10]
    
# get the average prediction per class

    # cat:  mean(0.20, 0.30, 0.10, 0.25) = 0.2125
    # dog:  mean(0.70, 0.60, 0.80, 0.65) = 0.6875
    # bear: mean(0.10, 0.10, 0.10, 0.10) = 0.10

# or get the max prediction per class

    # cat:  max = 0.30
    # dog:  max = 0.80
    # bear: max = 0.10
```


```python
preds,targs = learn.tta()
accuracy(preds, targs).item()
```
























    0.8737863898277283



A note on learn, Learner, DataLoaders, etc.

learn has attribute called dls.


dls has attributes: dls.train and dls.valid

By default, fastAI calls dls.valid when calling dls. The full call is learn.dls.valid

.tta() requires that we pass in a dataloader, such as learn.tta(dl=learn.dls.valid, n_aug=4). If no dataloader is specified, learn.dls.valid is passed in by default.

## Mixup

Mixup is essentially blending training images (and labels) together. This helps with smoother decision boundaries, and is also a way to add more training data to small data sets. Without mixup, model can be overconfident and memorize.

Pick a random % lambda. lambda * Img1 + (1-labda) * Img2. Same thing with labels (which must be one-hot encoded). E.g., lambda * [1, 0, 0] + (1-lambda) * [0,0,1]

Mixup is implemented by adding a callback to the Leaner class / learn object in fastai.


```python
#hide_input
#id mixup_example
#caption Mixing a church and a gas station
#alt An image of a church, a gas station and the two mixed up.
church = PILImage.create(get_image_files_sorted(path/'train'/'n03028079')[0])
gas = PILImage.create(get_image_files_sorted(path/'train'/'n03425413')[0])
church = church.resize((256,256))
gas = gas.resize((256,256))
tchurch = tensor(church).float() / 255.
tgas = tensor(gas).float() / 255.

_,axs = plt.subplots(1, 3, figsize=(12,4))
show_image(tchurch, ax=axs[0]);
show_image(tgas, ax=axs[1]);
show_image((0.3*tchurch + 0.7*tgas), ax=axs[2]);
```


    
![png](/images/112525_output_36_0.png)
    


## Label Smoothing

The idea behind label smoothing is that output probabilities are never exactly 1 or 0. I.e., the sigmoid function asymptotes at 1 or 0 but never reaches these values. To approximate 1 or 0, the model must push the logits to the extreme, toward infinity or negative infinity. But we don't want that. That makes the model unstable and difficult to take the gradient. Therefore, the solution is to use label smoothing. I.e., create ground-truth labels that approximate the correct level of confidence in the labels, but removes that absolute certainty of 1 or 0. Mathematically, these labels are within the range of the sigmoid function. Therefore, the model wouldn't push parameters/weights towards the extremes.


```python
# Mathmatically, how it works:

# choose level of uncertainty e = 0.1 (10% uncertainty)
# assume labels are one-hot encoded: y = [0, 0, 1, 0, 0, ... 0] with N categories. Labels sum to 1.

# all 0s get e/N uncertainty
# since y must sum to 1: 
    # 1 = (1-N)(e/N) + x 
    # x = 1 - e + e/N
    
# so we're left with something like y = [0.01, 0.01, 0.91, 0.01, ... 0.01] if we have 10 labels (N = 10)
```


```python
model = xresnet50(n_out=dls.c)
learn = Learner(dls, model, loss_func=LabelSmoothingCrossEntropy(),  # note the change in loss function
                metrics=accuracy)
learn.fit_one_cycle(5, 3e-3)
```
