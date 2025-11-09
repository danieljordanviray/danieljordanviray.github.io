# Improving AI Models: Image Augmentation, Loss Function Engineering, and Fine-tuning Training Parameters

![png](/images/110925-output_13_0.png)

Just completed my next project in learning how to build and engineer AI systems! In this notebook, I built an AI model that classifies dog breeds from images with up to 95% accuracy. Technically, this project explores core AI-engineering optimization techniques: image augmentation, loss-function mathematics, and fine-tuning training parameters.

### Table of Contents

- Pre-sizing and batch transformation of images
- Mathematically understanding cross-entropy loss function for debugging and performance optimization
- Model optimization techniques: optimal/discriminative learning rates, transfer learning/leveraging pretrained models, selecting model architectures and number of epochs to train for

### Load Libraries


```python
# ! pip install -Uqq fastbook
# import fastbook
# fastbook.setup_book()
```


```python
# from fastbook import *
# from fastai.vision.all import *
```

### Import Data

Download and load data. Dataset info: https://www.robots.ox.ac.uk/~vgg/data/pets/


```python
# download pet breeds data set
path = untar_data(URLs.PETS) # FastAI helper function
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
  <progress value='811712512' class='' max='811706944' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [811712512/811706944 00:11&lt;00:00]
</div>




```python
Path.BASE_PATH = path # points path operations to PETS dataset folder
```


```python
# view data folder structure
path.ls()
```




    (#2) [Path('images'),Path('annotations')]




```python
# check file names in file path
(path/"images").ls()
```




    (#7393) [Path('images/wheaten_terrier_3.jpg'),Path('images/basset_hound_176.jpg'),Path('images/scottish_terrier_30.jpg'),Path('images/chihuahua_90.jpg'),Path('images/great_pyrenees_106.jpg'),Path('images/keeshond_35.jpg'),Path('images/Bombay_64.jpg'),Path('images/basset_hound_93.jpg'),Path('images/Abyssinian_215.jpg'),Path('images/keeshond_79.jpg'),Path('images/newfoundland_43.jpg'),Path('images/miniature_pinscher_37.jpg'),Path('images/english_cocker_spaniel_155.jpg'),Path('images/samoyed_136.jpg'),Path('images/basset_hound_160.jpg'),Path('images/Russian_Blue_123.jpg'),Path('images/Russian_Blue_76.jpg'),Path('images/leonberger_102.jpg'),Path('images/scottish_terrier_53.jpg'),Path('images/english_setter_95.jpg')...]



View a subset of the images for a sanity check:


```python
from fastai.vision.utils import *
import matplotlib.pyplot as plt

files = (path/"images").ls()[:16]  # first 16 image paths
images = [PILImage.create(f) for f in files]

fig, axes = plt.subplots(4, 4, figsize=(12,12))

for ax, img, f in zip(axes.flatten(), images, files):
    ax.imshow(img)
    ax.set_title(f.name, fontsize=8)  # show just the filename
    ax.axis('off')

plt.tight_layout()
plt.show()
```


    
![png](/images/110925-output_13_0.png)
    


Test RegEx to extract labels from file names:


```python
fname = (path/"images").ls()[0]
fname
```




    Path('images/wheaten_terrier_3.jpg')




```python
re.findall(r'(.+)_\d+.jpg$', fname.name)
```




    ['wheaten_terrier']



Create DataBlock:

DataBlock = high-level FastAI object that gives instruction for how to create the data set.

- blocks = (Input type, Label type) (In this case, images and categorical labels) (single category label)
- get_itmes = get_image_files is just a function that gets a LIST of contents (images) from the file path (and all subfolders in the file path)
- get_y = how to get labels. using_attr() is using file attribute (file "name"), not the file itself. RegEx extracts label from file name.
- splitter = random 80/20% training/validation split. Returns two lists of indicies.
- item_tfms = see explanation below (item transformations). Each image is resized s.t. largest dimension is 460px.
- batch_tfms = see explanation below (batch transformations). Batch level transfomation applied on GPU after images are batched together. Random augmentations such as rotation, flip, warp, zoom, brightness, etc. Size = 224 x 224 px is image size fed to model. min_scale = how much random zooming/cropping to allow.


```python
pets = DataBlock(blocks = (ImageBlock, CategoryBlock),
                 get_items=get_image_files, 
                 splitter=RandomSplitter(seed=42),
                 get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'),
                 item_tfms=Resize(460), # default crop is "center crop"; shorter side is 460px, longer side is cropped to make is square (460px)
                 batch_tfms=aug_transforms(size=224, min_scale=0.75)) # min_scale = how much image can be cropped during random augmentation (b/w 75% to 100% of original size)
```

Actually create the data set:


```python
dls = pets.dataloaders(path/"images")
```

### Presizing and Batch Transformations

Theory:

Each image in the training data set has different pixel dimensions. We want to perform two actions on each image: (1) resize them such that each image is the same dimension, and (2) resize the image to very large dimensions (much larger than the dimensions we pass into the model).

The reason for step (2) is so that when we rotate, flip, and perform other augmentations on the training images, we can ensure the images are large enough to not lose image quality. For example, if we rotate the image 45 deg, we don't want black pixels in the corner if the image is too small.

We then batch these large images into a tensor. Then we perform data augmentation (e.g., rotation, flipping, etc.) on the entire batch. Performing these transformations as a batch in a tensor on the GPU ensures speed. Note that each image in the batch may receive different data augmentation functions; not all images in the batch will be transformed the same way.

For each training epoch in the model, the model will train on slighlty different "versions" of the training image. For example, say there is a specific image on a Beagle. In epoch 1, the model might train on a flipped version of the Beagle. In epoch 2, the model might train on a rotated version of the Beagle. This ensures the model generalizes the image rather than memorizing a specific pattern of pixels for the image. (Just like how we humans can detect the dog breed wheter it is laying upside down or is right side up).

Check dimensions of original images:


```python
from PIL import Image

# List image files
image_files = (path/"images").ls()

# Loop through the first 5 images
for img_path in image_files[:5]:
    with Image.open(img_path) as img:
        print(f"{img_path.name}: {img.size}")
```

    wheaten_terrier_3.jpg: (500, 313)
    basset_hound_176.jpg: (500, 375)
    scottish_terrier_30.jpg: (333, 500)
    chihuahua_90.jpg: (500, 333)
    great_pyrenees_106.jpg: (391, 500)



```python
# check training data set
dls.train_ds
```




    (#5912) [(PILImage mode=RGB size=375x500, TensorCategory(1)),(PILImage mode=RGB size=300x218, TensorCategory(3)),(PILImage mode=RGB size=500x375, TensorCategory(21)),(PILImage mode=RGB size=500x333, TensorCategory(16)),(PILImage mode=RGB size=500x326, TensorCategory(9)),(PILImage mode=RGB size=512x560, TensorCategory(3)),(PILImage mode=RGB size=500x376, TensorCategory(12)),(PILImage mode=RGB size=334x500, TensorCategory(21)),(PILImage mode=RGB size=333x500, TensorCategory(27)),(PILImage mode=RGB size=500x375, TensorCategory(34)),(PILImage mode=RGB size=500x375, TensorCategory(3)),(PILImage mode=RGB size=500x333, TensorCategory(13)),(PILImage mode=RGB size=500x351, TensorCategory(31)),(PILImage mode=RGB size=375x500, TensorCategory(16)),(PILImage mode=RGB size=245x300, TensorCategory(6)),(PILImage mode=RGB size=500x333, TensorCategory(13)),(PILImage mode=RGB size=500x375, TensorCategory(26)),(PILImage mode=RGB size=500x375, TensorCategory(30)),(PILImage mode=RGB size=377x500, TensorCategory(7)),(PILImage mode=RGB size=333x500, TensorCategory(11))...]




```python
# stored as tuples (image, category)
dls.train_ds[0][0] # index first tuple, first element in tuple (image)
```




    
![png](/images/110925-output_26_0.png)
    




```python
# check shape of image in dls before any transformation
dls.train_ds[0][0].shape
```




    (500, 375)




```python
# check dimensions of resized individual image (before batch transformation)
from fastai.vision.all import *

resize = Resize(460)
pil = PILImage.create((path/'images').ls()[0])
t = resize(pil)          # applies the item transform
print(t.shape)           # -> [3, 460, 460]
```

    (460, 460)



```python
# get one batch from the training set
# 64 images, 3 channels, 224 x 224 px
xb_train, yb_train = dls.train.one_batch()
print(xb_train.shape)
```

    torch.Size([64, 3, 224, 224])



```python
# view data augmentation in training set
dls.show_batch((xb, yb))
```


    
![png](/images/110925-output_30_0.png)
    



```python
# check dimensions of resized batched images in validation set
# 64 images, 3 channels (RBG), 224px x 224 px
xb_valid, yb_valid = dls.valid.one_batch()
print(xb_valid.shape)
```

    torch.Size([64, 3, 224, 224])



```python
# view sample of validation set
# notice how there are is not data augmentation here. Just re-sizing

import matplotlib.pyplot as plt
plt.rcParams['axes.titlesize'] = 6 

dls.show_batch((xb_valid, yb_valid), max_n=24, figsize=(8,8))
```


    
![png](/images/110925-output_32_0.png)
    


### Debugging Data Loaders

Check train/valid split:


```python
# check training / validation split (counts)
dls.train.n, dls.valid.n                 # counts per split
```




    (5912, 1478)




```python
# check training / validation split (percentage)
dls.train.n/7393, dls.valid.n/7393
```




    (0.7996753685919112, 0.1999188421479778)



Summary function to check everything loaded correctly in data loaders:


```python
# data loaders summary function
pets.summary(path/"images")
```

    Setting-up type transforms pipelines
    Collecting items from /root/.fastai/data/oxford-iiit-pet/images
    Found 7390 items
    2 datasets of sizes 5912,1478
    Setting up Pipeline: PILBase.create
    Setting up Pipeline: partial -> Categorize -- {'vocab': None, 'sort': True, 'add_na': False}
    
    
    Building one sample
      Pipeline: PILBase.create
        starting from
          /root/.fastai/data/oxford-iiit-pet/images/Bengal_92.jpg
        applying PILBase.create gives
          PILImage mode=RGB size=375x500
      Pipeline: partial -> Categorize -- {'vocab': None, 'sort': True, 'add_na': False}
    
        starting from
          /root/.fastai/data/oxford-iiit-pet/images/Bengal_92.jpg
        applying partial gives
          Bengal
        applying Categorize -- {'vocab': None, 'sort': True, 'add_na': False}
     gives
          TensorCategory(1)
    
    Final sample: (PILImage mode=RGB size=375x500, TensorCategory(1))
    
    
    Collecting items from /root/.fastai/data/oxford-iiit-pet/images
    Found 7390 items
    2 datasets of sizes 5912,1478
    Setting up Pipeline: PILBase.create
    Setting up Pipeline: partial -> Categorize -- {'vocab': None, 'sort': True, 'add_na': False}
    
    Setting up after_item: Pipeline: Resize -- {'size': (460, 460), 'method': 'crop', 'pad_mode': 'reflection', 'resamples': (<Resampling.BILINEAR: 2>, <Resampling.NEAREST: 0>), 'p': 1.0}
     -> ToTensor
    Setting up before_batch: Pipeline: 
    Setting up after_batch: Pipeline: IntToFloatTensor -- {'div': 255.0, 'div_mask': 1}
     -> Flip -- {'size': None, 'mode': 'bilinear', 'pad_mode': 'reflection', 'mode_mask': 'nearest', 'align_corners': True, 'p': 0.5}
     -> RandomResizedCropGPU -- {'size': (224, 224), 'min_scale': 0.75, 'ratio': (1, 1), 'mode': 'bilinear', 'valid_scale': 1.0, 'max_scale': 1.0, 'mode_mask': 'nearest', 'p': 1.0}
     -> Brightness -- {'max_lighting': 0.2, 'p': 1.0, 'draw': None, 'batch': False}
    
    
    Building one batch
    Applying item_tfms to the first sample:
      Pipeline: Resize -- {'size': (460, 460), 'method': 'crop', 'pad_mode': 'reflection', 'resamples': (<Resampling.BILINEAR: 2>, <Resampling.NEAREST: 0>), 'p': 1.0}
     -> ToTensor
        starting from
          (PILImage mode=RGB size=375x500, TensorCategory(1))
        applying Resize -- {'size': (460, 460), 'method': 'crop', 'pad_mode': 'reflection', 'resamples': (<Resampling.BILINEAR: 2>, <Resampling.NEAREST: 0>), 'p': 1.0}
     gives
          (PILImage mode=RGB size=460x460, TensorCategory(1))
        applying ToTensor gives
          (TensorImage of size 3x460x460, TensorCategory(1))
    
    Adding the next 3 samples
    
    No before_batch transform to apply
    
    Collating items in a batch
    
    Applying batch_tfms to the batch built
      Pipeline: IntToFloatTensor -- {'div': 255.0, 'div_mask': 1}
     -> Flip -- {'size': None, 'mode': 'bilinear', 'pad_mode': 'reflection', 'mode_mask': 'nearest', 'align_corners': True, 'p': 0.5}
     -> RandomResizedCropGPU -- {'size': (224, 224), 'min_scale': 0.75, 'ratio': (1, 1), 'mode': 'bilinear', 'valid_scale': 1.0, 'max_scale': 1.0, 'mode_mask': 'nearest', 'p': 1.0}
     -> Brightness -- {'max_lighting': 0.2, 'p': 1.0, 'draw': None, 'batch': False}
    
        starting from
          (TensorImage of size 4x3x460x460, TensorCategory([ 1,  3, 21, 16], device='cuda:0'))
        applying IntToFloatTensor -- {'div': 255.0, 'div_mask': 1}
     gives
          (TensorImage of size 4x3x460x460, TensorCategory([ 1,  3, 21, 16], device='cuda:0'))
        applying Flip -- {'size': None, 'mode': 'bilinear', 'pad_mode': 'reflection', 'mode_mask': 'nearest', 'align_corners': True, 'p': 0.5}
     gives
          (TensorImage of size 4x3x460x460, TensorCategory([ 1,  3, 21, 16], device='cuda:0'))
        applying RandomResizedCropGPU -- {'size': (224, 224), 'min_scale': 0.75, 'ratio': (1, 1), 'mode': 'bilinear', 'valid_scale': 1.0, 'max_scale': 1.0, 'mode_mask': 'nearest', 'p': 1.0}
     gives
          (TensorImage of size 4x3x224x224, TensorCategory([ 1,  3, 21, 16], device='cuda:0'))
        applying Brightness -- {'max_lighting': 0.2, 'p': 1.0, 'draw': None, 'batch': False}
     gives
          (TensorImage of size 4x3x224x224, TensorCategory([ 1,  3, 21, 16], device='cuda:0'))


### Train a Simple Model for Debugging Purposes

Check for existence and use of GPU (cuda):


```python
learn.dls.device      # device used for the DataLoaders
```




    device(type='cuda', index=0)




```python
next(learn.model.parameters()).device # device the model is on
```




    device(type='cuda', index=0)



Purposes: (1) Get a quick baseline accuracy, (2) does the model train at all?


```python
learn = vision_learner(dls, resnet34, metrics=[accuracy, error_rate])
learn.fine_tune(2)
# fasiAI defaults to cross-entropy loss function; inferred from image data and categorical outcome
```

    Downloading: "https://download.pytorch.org/models/resnet34-b627a593.pth" to /root/.cache/torch/hub/checkpoints/resnet34-b627a593.pth
    100%|██████████| 83.3M/83.3M [00:00<00:00, 132MB/s]




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
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.555198</td>
      <td>0.338130</td>
      <td>0.894452</td>
      <td>0.105548</td>
      <td>00:30</td>
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
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.494421</td>
      <td>0.327485</td>
      <td>0.903248</td>
      <td>0.096752</td>
      <td>00:36</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.308087</td>
      <td>0.241630</td>
      <td>0.924222</td>
      <td>0.075778</td>
      <td>00:36</td>
    </tr>
  </tbody>
</table>


### Cross Entropy Loss

The purpose of this section is to better understand the output behind cross-entropy loss. Cross-entropy loss is used for multi-category classification problems.

There are two parts to Cross Entropy Loss: (1) Softmax and (2) Log Likelihood.

#### View Activations and Labels


```python
# select one batch (default to validation set)
# 64 images/labels in a batch
x,y = dls.valid.one_batch()
```


```python
# view what y (dependent variable) looks like
# they are categories / pet breeds
y
```




    TensorCategory([14, 14, 22, 29,  3, 24,  6, 17, 15, 26, 16, 25, 11, 35, 24, 33, 15, 21,  4, 22, 20, 36, 31,  5,  5, 20, 30, 11, 31, 24, 11, 28, 11, 33,  6,  9, 11, 18,  7,  3, 15, 17, 18, 17, 16, 12,
                     5, 23, 14,  4, 22, 17,  1,  4, 19,  5, 11, 27, 22, 36, 36, 33, 20,  2], device='cuda:0')



Obtain predictions (activations of final layer of neural net) for just one batch:


```python
preds,labels = learn.get_preds(dl=[(x,y)])
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







There are 64 predictions and 64 (actual) labels since our batch has 64 images/labels in the set.


```python
len(preds)
```




    64




```python
len(labels)
```




    64




```python
# same output as "y" above
labels
```




    tensor([14, 14, 22, 29,  3, 24,  6, 17, 15, 26, 16, 25, 11, 35, 24, 33, 15, 21,  4, 22, 20, 36, 31,  5,  5, 20, 30, 11, 31, 24, 11, 28, 11, 33,  6,  9, 11, 18,  7,  3, 15, 17, 18, 17, 16, 12,  5, 23,
            14,  4, 22, 17,  1,  4, 19,  5, 11, 27, 22, 36, 36, 33, 20,  2])




```python
# for each prediction element, there are 37 probabilities; one for each breed.
len(preds[0])
```




    37




```python
# here are the 37 probabilities for the first image:
preds[0]
```




    tensor([1.2690e-07, 3.5162e-06, 5.3713e-07, 1.0684e-07, 1.9571e-08, 3.5571e-07, 1.5125e-08, 4.4227e-08, 2.2894e-07, 3.2881e-08, 4.9321e-08, 9.2373e-08, 6.7784e-06, 3.0692e-07, 9.9918e-01, 3.4053e-04,
            3.8140e-06, 8.6007e-09, 1.9983e-06, 1.1052e-05, 2.4418e-05, 3.8791e-08, 2.5733e-07, 7.8192e-08, 1.3803e-07, 2.1454e-06, 3.5073e-07, 1.1836e-07, 6.9297e-08, 1.2144e-07, 4.1609e-04, 2.6710e-07,
            1.1337e-07, 4.2507e-07, 5.2886e-07, 1.2182e-07, 2.6713e-07])




```python
# the sum of the 37 probabilities sums to 1 (100% probability)
preds[0].sum()
```




    tensor(1.)



#### Part 1/2 of Cross Entropy Loss: Softmax


```python
# example sigmoid function
# softmax is similar to sigmoid, but for multi-class classification
plot_function(torch.sigmoid, min=-4,max=4)
```


    
![png](/images/110925-output_60_0.png)
    


Here is a toy example -- there are six images in a binary classification problem:


```python
# example: 6 images, 2 possible categories, std dev of 2 (so multiply rand by 2)
torch.random.manual_seed(42);
acts = torch.randn((6,2))*2
acts
```




    tensor([[ 0.6734,  0.2576],
            [ 0.4689,  0.4607],
            [-2.2457, -0.3727],
            [ 4.4164, -1.2760],
            [ 0.9233,  0.5347],
            [ 1.0698,  1.6187]])




```python
# we can't just take sigmoid of acts since predictions will not sum to 1 (at row-level)
acts.sigmoid()
```




    tensor([[0.6623, 0.5641],
            [0.6151, 0.6132],
            [0.0957, 0.4079],
            [0.9881, 0.2182],
            [0.7157, 0.6306],
            [0.7446, 0.8346]])




```python
# the activations indicate the relative confidence in the classification prediction
# i.e., how much higher/lower the activation score is in one class vs. the other
# since there are only two classes in the example, we can take the difference between the classes
(acts[:,0]-acts[:,1])
```




    tensor([ 0.4158,  0.0083, -1.8731,  5.6924,  0.3886, -0.5489])




```python
# now we can take the sigmoid of the differences in predictions
# we obtain probability that the class is Category 0, so 1-P(Cateogry 0) = P(Category 1)
(acts[:,0]-acts[:,1]).sigmoid()
```




    tensor([0.6025, 0.5021, 0.1332, 0.9966, 0.5959, 0.3661])




```python
# softmax is the generalized form of this for multi-class classification

# softmax function is: 
    # def softmax(x): return exp(x) / exp(x).sum(dim=1, keepdim=True)
    
# softmax returns same value as sigmoid for column 0
sm_acts = torch.softmax(acts, dim=1)
sm_acts
```




    tensor([[0.6025, 0.3975],
            [0.5021, 0.4979],
            [0.1332, 0.8668],
            [0.9966, 0.0034],
            [0.5959, 0.4041],
            [0.3661, 0.6339]])



Notes on the math/intuition behind softmax:
- e^(x) ensures that all numbers are positive
- e^(x) / e^(x).sum turns the logits into probabilities that sum to 1
- the exponential magnifies the relative distance between the logits; the distance is magnified exponentially
- in practice, this means that softmax is great at picking a single class from the option of classes; however, there may be cases where we want the model to say "does not recognize any of the classes. (In this case, use multiple binary output columns, each using a sigmoid activation.) 

Note on multiple binary output columns (good for multiple category classes or classes not recognized):

- In softmax, we're saying: Here are all the options of classses (pet breeds), choose the one that first the best.
- In multiple binary output columns, we're saying: Is it a beagle (Y/N) (i.e., >50% probability)? Is it a terrier (Y/N)? Is it a labrador (Y/N)? The model can find that it's none of the options, or a mix of options (e.g., beagle/terrier mix).

#### Part 2/2 of Cross Entropy Loss: Log Likelihood

Assume a binary classification problem where the labels of six images are as follows:


```python
# toy/sample targets/labels (6 images)
labels = tensor([0,1,0,1,1,0])
```

Assume these are our softmax activations for the images:


```python
# softmax activations
sm_acts
```




    tensor([[0.6025, 0.3975],
            [0.5021, 0.4979],
            [0.1332, 0.8668],
            [0.9966, 0.0034],
            [0.5959, 0.4041],
            [0.3661, 0.6339]])



Select corresponding softmax activation (probability) for each image label:


```python
# for all 6 rows in sm_acts, take targ index
idx = range(6)
sm_acts[idx, labels]
```




    tensor([0.6025, 0.4979, 0.1332, 0.0034, 0.4041, 0.3661])



Output in table format:


```python
from IPython.display import HTML
from IPython.display import display

df = pd.DataFrame(sm_acts, columns=["P(0)","P(1)"])

# df['idx'] = idx
df['Label'] = labels
df['P(Label)'] = sm_acts[range(6), labels]

#HTML(df.to_html())
display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>P(0)</th>
      <th>P(1)</th>
      <th>Label</th>
      <th>P(Label)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.602469</td>
      <td>0.397531</td>
      <td>0</td>
      <td>0.602469</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.502065</td>
      <td>0.497935</td>
      <td>1</td>
      <td>0.497935</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.133188</td>
      <td>0.866811</td>
      <td>0</td>
      <td>0.133188</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.996640</td>
      <td>0.003360</td>
      <td>1</td>
      <td>0.003360</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.595949</td>
      <td>0.404051</td>
      <td>1</td>
      <td>0.404051</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.366118</td>
      <td>0.633882</td>
      <td>0</td>
      <td>0.366118</td>
    </tr>
  </tbody>
</table>
</div>


The PyTorch function F.nll_loss that does the same thing above (and takes the negative):


```python
# PyTorch function : negative log likelihood (note: does not take the log). Assumes you've already taken the log
# log_softmax function takes log and softmax. nll_loss meant to be used AFTER log_softmax
F.nll_loss(sm_acts, labels, reduction='none')
```




    tensor([-0.6025, -0.4979, -0.1332, -0.0034, -0.4041, -0.3661])



### Taking the Log

We want to take the log of the negative log likelihoood numbers for practical mathematical reasons (easier calculations):

This is what the log function looks like:

x : [0, inf]

y : [-inf, inf]

As x -> 0, y -> -inf



```python
# log function
# log(1) = 0
# PyTorch log is log base e
plot_function(torch.log, min=0,max=1, ty='log(x) (loss)', tx='x (probability)')
```


    
![png](/images/110925-output_83_0.png)
    


For prediction purposes, assume true label = 1. 

When x -> 0 (predicted probability), we want loss function to be high (inf) 

And when x -> 1 (predicted probability), we want loss function to be low (close of 0).

Therefore, we take the NEGATIVE log function.


```python
# negative log function
# we take negative log function so that when prob -> 1, loss -> 0; when prob -> 0, loss -> inf
plot_function(lambda x: -1*torch.log(x), min=0,max=1, tx='x (probability)', ty='- log(x) (loss)', title = 'Log Loss when true label = 1')
```


    
![png](/images/110925-output_85_0.png)
    


Here is the updated table with the calculated Loss (the log of the predicted label):


```python
from IPython.display import HTML
from IPython.display import display

df = pd.DataFrame(sm_acts, columns=["P(0)","P(1)"])

# df['idx'] = idx
df['Label'] = labels
df['P(Label)'] = sm_acts[range(6), labels]
df['Loss = (Log of P(Label)'] = -torch.log(sm_acts[range(6), labels])

#HTML(df.to_html())
display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>P(0)</th>
      <th>P(1)</th>
      <th>Label</th>
      <th>P(Label)</th>
      <th>Loss = (Log of P(Label)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.602469</td>
      <td>0.397531</td>
      <td>0</td>
      <td>0.602469</td>
      <td>0.506720</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.502065</td>
      <td>0.497935</td>
      <td>1</td>
      <td>0.497935</td>
      <td>0.697285</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.133188</td>
      <td>0.866811</td>
      <td>0</td>
      <td>0.133188</td>
      <td>2.015990</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.996640</td>
      <td>0.003360</td>
      <td>1</td>
      <td>0.003360</td>
      <td>5.695763</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.595949</td>
      <td>0.404051</td>
      <td>1</td>
      <td>0.404051</td>
      <td>0.906213</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.366118</td>
      <td>0.633882</td>
      <td>0</td>
      <td>0.366118</td>
      <td>1.004798</td>
    </tr>
  </tbody>
</table>
</div>


Loss is high in index 2 and 3 where prediction is confident but inaccurate.

### Negative Log Likelihood

Taking the average of the "Loss" column above is the negative log likelihood loss AKA the cross entropy loss.


```python
# loss column
-torch.log(sm_acts[range(6), labels])
```




    tensor([0.5067, 0.6973, 2.0160, 5.6958, 0.9062, 1.0048])




```python
# mean of Loss column = negative log likelihood loss = cross entropy loss
-torch.log(sm_acts[range(6), labels]).mean()
```




    tensor(1.8045)



Here it is calculated using Cross Entropy Loss function in PyTorch:


```python
# instantiate cross entropy loss as function
loss_func = nn.CrossEntropyLoss()
```


```python
# calculate cross entropy loss using function
loss_func(acts, targ)
```




    tensor(1.8045)




```python
# can also use:
F.cross_entropy(acts, targ)
```




    tensor(1.8045)



PyTorch loss functions take mean of loss of all items by default. Can use reduction = 'none' to disable default aggregation. 

Note that these figures are the same figures in the Loss column.


```python
nn.CrossEntropyLoss(reduction='none')(acts, targ)
```




    tensor([0.5067, 0.6973, 2.0160, 5.6958, 0.9062, 1.0048])



### Cross Entropy Loss Recap/Summary

Implementing CrossEntropyLoss() manually:

1. Start with logits (raw scores from your model).

2. Apply softmax to turn them into probabilities.

3. Take negative log of those probabilities.

4. Select the negative log-prob of the correct class for each example.

5. Average them.

nn.CrossEntropyLoss() -> Pass in raw logits

F.nll_loss() -> Pass in log(softmax(logits))

1. Start with logits (activations = acts) (raw scores from your model).


```python
acts
```




    tensor([[ 0.6734,  0.2576],
            [ 0.4689,  0.4607],
            [-2.2457, -0.3727],
            [ 4.4164, -1.2760],
            [ 0.9233,  0.5347],
            [ 1.0698,  1.6187]])




```python
targ
```




    tensor([0, 1, 0, 1, 1, 0])



2. Apply softmax to turn them into probabilities.


```python
softmax = F.softmax(acts, dim=1)
softmax
```




    tensor([[0.6025, 0.3975],
            [0.5021, 0.4979],
            [0.1332, 0.8668],
            [0.9966, 0.0034],
            [0.5959, 0.4041],
            [0.3661, 0.6339]])



3. Take negative log of those probabilities.


```python
log_softmax = torch.log(softmax)
log_softmax
```




    tensor([[-5.0672e-01, -9.2248e-01],
            [-6.8903e-01, -6.9729e-01],
            [-2.0160e+00, -1.4293e-01],
            [-3.3658e-03, -5.6958e+00],
            [-5.1760e-01, -9.0621e-01],
            [-1.0048e+00, -4.5589e-01]])




```python
log_probs = F.log_softmax(acts, dim=1)
log_probs
```




    tensor([[-5.0672e-01, -9.2248e-01],
            [-6.8903e-01, -6.9729e-01],
            [-2.0160e+00, -1.4293e-01],
            [-3.3658e-03, -5.6958e+00],
            [-5.1760e-01, -9.0621e-01],
            [-1.0048e+00, -4.5589e-01]])



4. Select the negative log-prob of the correct class for each example.


```python
# log_softmax of true classes
log_softmax[range(6), targ]
```




    tensor([-0.5067, -0.6973, -2.0160, -5.6958, -0.9062, -1.0048])




```python
# take negative of log_softmax of true classes
-log_softmax[range(6), targ]
```




    tensor([0.5067, 0.6973, 2.0160, 5.6958, 0.9062, 1.0048])



5. Average them.


```python
# mean loss
-log_softmax[range(6), targ].mean()
```




    tensor(1.8045)



Here are the same results using the PyTorch F.nll_loss function:


```python
acts
```




    tensor([[ 0.6734,  0.2576],
            [ 0.4689,  0.4607],
            [-2.2457, -0.3727],
            [ 4.4164, -1.2760],
            [ 0.9233,  0.5347],
            [ 1.0698,  1.6187]])




```python
sm_acts
```




    tensor([[0.6025, 0.3975],
            [0.5021, 0.4979],
            [0.1332, 0.8668],
            [0.9966, 0.0034],
            [0.5959, 0.4041],
            [0.3661, 0.6339]])




```python
F.nll_loss(log_softmax, targ, reduction='none')
```




    tensor([0.5067, 0.6973, 2.0160, 5.6958, 0.9062, 1.0048])




```python
F.nll_loss(log_softmax, targ, reduction='none').mean()
```




    tensor(1.8045)



### Model Interpretation

Full confusion matrix (difficult to read if there are many classes):


```python
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
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








    
![png](/images/110925-output_121_4.png)
    


Obtain the top N most confused classes for easier interpretation:


```python
interp.most_confused(min_val=5)
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










    [('Ragdoll', 'Birman', 9),
     ('staffordshire_bull_terrier', 'american_pit_bull_terrier', 7),
     ('Egyptian_Mau', 'Bengal', 6)]



### Improving the Model

### Learning Rate Finder

Problem: We need to find an optimal learning rate that is not too low and not too high.

If it's too low, training will be too slow and the model will memorize the data.

If the learning rate is too high, the model will overshoot the minimum loss and lead to higher error rate (e.g., lr = 0.1):


```python
learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1, base_lr=0.1)
```

    Downloading: "https://download.pytorch.org/models/resnet34-b627a593.pth" to /root/.cache/torch/hub/checkpoints/resnet34-b627a593.pth
    100%|██████████| 83.3M/83.3M [00:00<00:00, 139MB/s]




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
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2.674522</td>
      <td>3.832227</td>
      <td>0.408660</td>
      <td>00:30</td>
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
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>3.075671</td>
      <td>1.445985</td>
      <td>0.460758</td>
      <td>00:36</td>
    </tr>
  </tbody>
</table>


Solution: Find the optimal learning rate by starting with a very small LR, gradually increase it, and track the loss in sucessive mini-batches.

We choose a LR at a point between where the loss is clearly decreasing and before the loss starts to increase:


```python
learn = vision_learner(dls, resnet34, metrics=error_rate)
lr_min,lr_steep = learn.lr_find(suggest_funcs=(minimum, steep))
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








    
![png](/images/110925-output_129_2.png)
    



```python
print(f"Minimum/10: {lr_min:.2e}, steepest point: {lr_steep:.2e}")
```

    Minimum/10: 1.00e-02, steepest point: 4.37e-03


In this case, choose a learning rate of 3 x 10^(-3) since the loss is decreasing at this LR:

(Middle point between 1e-3 and 1e-2 is between 3e-3 and 4e-3 because LR finder is plotted on a logarithmic scale).


```python
learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(2, base_lr=3e-3)
```

### Unfreezing and Transfer Learning

Purpose: fine tune the weights of a pre-trained model. Keep all existing weights from previously trained models but only change/train the final output layers. (E.g., we need 37 outputs specifically for 37 breeds of dogs. And we need to train the computer vision model specifically on detecting fur, eye balls, etc.).

The fine_tune function freezes all pre-trained layers and trains the randomly added layers for one epoch. Then it unfreezes all of the layers, and trains them for n number of epochs.

But we will probably want to manually call the underlying methods in fine_tune for more control/customization:


```python
learn.fine_tune??
```


    [0;31mSignature:[0m
    [0mlearn[0m[0;34m.[0m[0mfine_tune[0m[0;34m([0m[0;34m[0m
    [0;34m[0m    [0mepochs[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mbase_lr[0m[0;34m=[0m[0;36m0.002[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mfreeze_epochs[0m[0;34m=[0m[0;36m1[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mlr_mult[0m[0;34m=[0m[0;36m100[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mpct_start[0m[0;34m=[0m[0;36m0.3[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mdiv[0m[0;34m=[0m[0;36m5.0[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0;34m*[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mlr_max[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mdiv_final[0m[0;34m=[0m[0;36m100000.0[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mwd[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mmoms[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mcbs[0m[0;34m=[0m[0;32mNone[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mreset_opt[0m[0;34m=[0m[0;32mFalse[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m    [0mstart_epoch[0m[0;34m=[0m[0;36m0[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0;31mSource:[0m   
    [0;34m@[0m[0mpatch[0m[0;34m[0m
    [0;34m[0m[0;34m@[0m[0mdelegates[0m[0;34m([0m[0mLearner[0m[0;34m.[0m[0mfit_one_cycle[0m[0;34m)[0m[0;34m[0m
    [0;34m[0m[0;32mdef[0m [0mfine_tune[0m[0;34m([0m[0mself[0m[0;34m:[0m[0mLearner[0m[0;34m,[0m [0mepochs[0m[0;34m,[0m [0mbase_lr[0m[0;34m=[0m[0;36m2e-3[0m[0;34m,[0m [0mfreeze_epochs[0m[0;34m=[0m[0;36m1[0m[0;34m,[0m [0mlr_mult[0m[0;34m=[0m[0;36m100[0m[0;34m,[0m[0;34m[0m
    [0;34m[0m              [0mpct_start[0m[0;34m=[0m[0;36m0.3[0m[0;34m,[0m [0mdiv[0m[0;34m=[0m[0;36m5.0[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m
    [0;34m[0m    [0;34m"Fine tune with `Learner.freeze` for `freeze_epochs`, then with `Learner.unfreeze` for `epochs`, using discriminative LR."[0m[0;34m[0m
    [0;34m[0m    [0mself[0m[0;34m.[0m[0mfreeze[0m[0;34m([0m[0;34m)[0m[0;34m[0m
    [0;34m[0m    [0mself[0m[0;34m.[0m[0mfit_one_cycle[0m[0;34m([0m[0mfreeze_epochs[0m[0;34m,[0m [0mslice[0m[0;34m([0m[0mbase_lr[0m[0;34m)[0m[0;34m,[0m [0mpct_start[0m[0;34m=[0m[0;36m0.99[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m
    [0;34m[0m    [0mbase_lr[0m [0;34m/=[0m [0;36m2[0m[0;34m[0m
    [0;34m[0m    [0mself[0m[0;34m.[0m[0munfreeze[0m[0;34m([0m[0;34m)[0m[0;34m[0m
    [0;34m[0m    [0mself[0m[0;34m.[0m[0mfit_one_cycle[0m[0;34m([0m[0mepochs[0m[0;34m,[0m [0mslice[0m[0;34m([0m[0mbase_lr[0m[0;34m/[0m[0mlr_mult[0m[0;34m,[0m [0mbase_lr[0m[0;34m)[0m[0;34m,[0m [0mpct_start[0m[0;34m=[0m[0mpct_start[0m[0;34m,[0m [0mdiv[0m[0;34m=[0m[0mdiv[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
    [0;31mFile:[0m      /usr/local/lib/python3.11/dist-packages/fastai/callback/schedule.py
    [0;31mType:[0m      method


fit_one_cycle method is the suggested way to train models without calling fine_tune. 

fit_one_cycle starts training at a low learning rate, gradually increases LR for the first section of training, and then gradually decreases LR for last section of training.

Train randomly added layers for three epochs (using previously found learning rate):


```python
learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fit_one_cycle(3, 3e-3)
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
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.112472</td>
      <td>0.364155</td>
      <td>0.112314</td>
      <td>00:30</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.562948</td>
      <td>0.262153</td>
      <td>0.086604</td>
      <td>00:30</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.331478</td>
      <td>0.228014</td>
      <td>0.073072</td>
      <td>00:29</td>
    </tr>
  </tbody>
</table>


Unfreeze the model:


```python
learn.unfreeze()
```

Now we need to find a new learning rate because we have more layers to train and weights that have been trained for three epochs. This means the previously found learning rates are not appropriate anymore.

We won't have a decrease in loss because the model has been trained already. At some LR, the loss will just increase. Choose an LR well before the increase in loss. (In this case, 1e-5).


```python
learn.lr_find()
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










    SuggestedLRs(valley=5.248074739938602e-05)




    
![png](/images/110925-output_141_3.png)
    


Train for 6 epochs with max learning rate:


```python
learn.fit_one_cycle(6, lr_max=1e-5)
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
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.245999</td>
      <td>0.217690</td>
      <td>0.071042</td>
      <td>00:36</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.246839</td>
      <td>0.217532</td>
      <td>0.071042</td>
      <td>00:37</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.227155</td>
      <td>0.215544</td>
      <td>0.071719</td>
      <td>00:37</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.196656</td>
      <td>0.207717</td>
      <td>0.068336</td>
      <td>00:37</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.188031</td>
      <td>0.201533</td>
      <td>0.064953</td>
      <td>00:37</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.180559</td>
      <td>0.204955</td>
      <td>0.067659</td>
      <td>00:37</td>
    </tr>
  </tbody>
</table>


### Discriminative Learning Rates

The deepest layers of pre-trained models might need smaller learning rates than the later layers. Therefore, we want to adapt the learning rates to the depth of the layers (smaller LRs for deeper layers, larger LRs of later layers).

lr_max=slice(1e-6, 1e-4) means deepest layer uses LR of 1e-6 and last layer uses LR of 1e-4. Layers inbetween use LRs that sliced equidistant from 1e-6 to 1e-4.


```python
learn = vision_learner(dls, resnet34, metrics=error_rate)
learn.fit_one_cycle(3, 3e-3)
learn.unfreeze()
learn.fit_one_cycle(12, lr_max=slice(1e-6,1e-4))
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
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.154180</td>
      <td>0.359302</td>
      <td>0.110961</td>
      <td>00:29</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.516324</td>
      <td>0.270517</td>
      <td>0.087280</td>
      <td>00:30</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.334596</td>
      <td>0.238323</td>
      <td>0.069012</td>
      <td>00:30</td>
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
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.276683</td>
      <td>0.234703</td>
      <td>0.069689</td>
      <td>00:37</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.252736</td>
      <td>0.235514</td>
      <td>0.070365</td>
      <td>00:37</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.232454</td>
      <td>0.231764</td>
      <td>0.070365</td>
      <td>00:37</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.215191</td>
      <td>0.233608</td>
      <td>0.075101</td>
      <td>00:37</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.193059</td>
      <td>0.223943</td>
      <td>0.064953</td>
      <td>00:37</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.171077</td>
      <td>0.220571</td>
      <td>0.072395</td>
      <td>00:37</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.164041</td>
      <td>0.219061</td>
      <td>0.069012</td>
      <td>00:37</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.144370</td>
      <td>0.224508</td>
      <td>0.071719</td>
      <td>00:37</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.120082</td>
      <td>0.218150</td>
      <td>0.066982</td>
      <td>00:37</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.120558</td>
      <td>0.215837</td>
      <td>0.066982</td>
      <td>00:38</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.117585</td>
      <td>0.220407</td>
      <td>0.069012</td>
      <td>00:37</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.126130</td>
      <td>0.216361</td>
      <td>0.067659</td>
      <td>00:36</td>
    </tr>
  </tbody>
</table>


Plot training and validation loss on graph:


```python
learn.recorder.plot_loss()
```




    <Axes: title={'center': 'learning curve'}, xlabel='steps', ylabel='loss'>




    
![png](/images/110925-output_148_1.png)
    


### Selecting the Number of Epochs

Problem: How many epochs to train for?

Considerations:

- Usually limited by the amount of time you have to train
- Accuracy may get worse over time once the model starts to memorize data. If this happens, you'll want to train again from scratch and stop training before accuracy starts to decrease.
- If you have time to train for more epochs, time is better spent on training a deeper architecture with more layers.

### Deeper Architectures

In general, deeper architectures (more layers) can model data more accurately. However, training deeper architecures takes more time and computing power.

Pre-trained models (e.g., ResNet) come in different variants (18, 34, 50, 101, 152 layers).

A way to speed up training is to use less-precise numbers (half-precision floating point, also called fp16). This can speed up training by 2-3x. In fastAi, just call .to_fp16() after the Learner.


```python
from fastai.callback.fp16 import *
learn = vision_learner(dls, resnet50, metrics=error_rate).to_fp16()
learn.fine_tune(6, freeze_epochs=3)
```

    Downloading: "https://download.pytorch.org/models/resnet50-11ad3fa6.pth" to /root/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth
    100%|██████████| 97.8M/97.8M [00:00<00:00, 142MB/s]




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
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2.205635</td>
      <td>0.471300</td>
      <td>0.147497</td>
      <td>00:53</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.851554</td>
      <td>0.274626</td>
      <td>0.086604</td>
      <td>00:53</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.552624</td>
      <td>0.287259</td>
      <td>0.097429</td>
      <td>00:53</td>
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
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.297660</td>
      <td>0.235255</td>
      <td>0.083221</td>
      <td>01:05</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.279863</td>
      <td>0.231873</td>
      <td>0.071042</td>
      <td>01:06</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.199640</td>
      <td>0.221607</td>
      <td>0.064276</td>
      <td>01:05</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.136157</td>
      <td>0.218444</td>
      <td>0.062923</td>
      <td>01:05</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.098143</td>
      <td>0.199191</td>
      <td>0.056834</td>
      <td>01:05</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.062279</td>
      <td>0.197232</td>
      <td>0.058863</td>
      <td>01:05</td>
    </tr>
  </tbody>
</table>


Note that we used learn.fine_tune() instead of .fit_one_cycle(). freeze_epochs parameters tells fastai how many epochs to train for while the deeper layers in the model are frozen. This function in general will automatically select the correct learning rate for most data sets.

Note that error rate / accuracy didn't improve much from smaller model, so it is more efficient to start with small models first.

### Conclusion

This notebook covered:

- Preparing image data for AI modeling
- Mathematically understanding the cross-entropy loss function for debugging and performance optimization
- Optimizing AI model training: learning rates, architectures, and epochs
