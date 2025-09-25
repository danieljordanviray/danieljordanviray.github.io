### AI - Training an AI Model to Detect Emotions from Images

### Abstract / Summary

Built my first hand-coded AI model (using PyTorch)! The model can detect human emotions from a picture of a face. 

What's covered in this notebook:

- Data ingestion (loading/batching ~30k 48x48px grayscale images)
- Data preprocessing (addressing class imbalance and weighting loss functions accordingly)
- Hand-coding fundamental AI functions/processes (initializing parameters, linear/non-linear models, loss functions, forward/backward passes, batch and epoch accuracies)
- Training AI models using PyTorch modules
- Training AI models using FastAI's libraries
- Fitting ResNet18 architecture/model on the data
- Model performance evaluation (visual inspection and confusion matrices)

In sum, my hand-coded models achieved up to ~30% accuracy, whereas the ResNet18 model achieved up to ~60% accuracy. For reference, baseline accuracy was ~25% due to class imbalance, and the highest known performance on this dataset is ~70–75% accuracy.

However, my goal for this project wasn't necessarily to build the most accurate model, but rather to learn the fundamental principles of AI-engineering. 

Follow along as I dive deeper into this journey!

### Load libraries / dependencies


```python
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision
import torch.nn.functional as F
import torch.nn as nn

from collections import Counter # for checking class balance
import matplotlib.pyplot as plt
import numpy as np
```


```python
!pip install -Uqq fastbook
import fastbook
fastbook.setup_book()
```


```python
from fastai.vision.all import *
from fastbook import *
```


```python
# set seeds for re-producibility 
import random
import numpy as np
import torch

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Check if GPU is available


```python
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current device index:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("Running on CPU")
```

    CUDA available: True
    CUDA device count: 1
    Current device index: 0
    Device name: Quadro P5000


### Load images

Here we are transforming the images into tensors of pixels, and then splitting into train and test data sets.


```python
# Define transformation: convert image to tensor and ensure it's grayscale
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to 1 channel if needed
    transforms.Resize((48, 48)),                  # Resize to 48x48 (FER2013 standard)
    transforms.ToTensor(),                        # Convert to tensor with shape (1, 48, 48) # also normalized by being bounded [0, 1]
])

# Load training and test datasets
# reads in .jpg images and takes class names from folder structure
train_dataset = ImageFolder(root='train', transform=transform)
test_dataset = ImageFolder(root='test', transform=transform)
```

### Conduct data quality check/preview

View class labels:


```python
# View class labels assigned
print("Class-to-Index mapping (training data set):", train_dataset.class_to_idx)
print("Class-to-Index mapping (testing data set):", test_dataset.class_to_idx)
```

    Class-to-Index mapping (training data set): {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
    Class-to-Index mapping (testing data set): {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}


Preview image:


```python
# Grab image and label
image, label = train_dataset[0]

# image is a tensor of shape (1, 48, 48). We squeeze out the channel dimension to make it (48, 48)
plt.figure(figsize=(1,1))
plt.imshow(image.squeeze(0), cmap='gray')  # cmap='gray' to show it in grayscale
plt.title(f"Label Index: {label}")
plt.axis('off')
plt.show()
```


    
![png](/images/092525-output_17_0.png)


Check for and remove corrupt files: 

(Checked separately)

Check if pixels have been normalized from [0,1]:


```python
# squeeze channel so it's (48, 48)
arr = image.squeeze(0).numpy()

print(arr.shape)     # (48, 48)
print(arr[:48, :48])
print("min:", arr.min(), "max:", arr.max())
```

    (48, 48)
    [[0.19607843 0.1254902  0.05882353 ... 0.52156866 0.5921569  0.3372549 ]
     [0.22352941 0.13333334 0.08627451 ... 0.5411765  0.5921569  0.34901962]
     [0.23921569 0.11764706 0.09411765 ... 0.5568628  0.58431375 0.34901962]
     ...
     [0.40392157 0.39215687 0.39215687 ... 0.58431375 0.40784314 0.33333334]
     [0.41960785 0.43529412 0.44313726 ... 0.5921569  0.47058824 0.3372549 ]
     [0.40784314 0.40784314 0.4392157  ... 0.56078434 0.53333336 0.3254902 ]]
    min: 0.011764706 max: 0.77254903


### Check for and address class imbalances


```python
train_counts = Counter(train_dataset.targets)   # ImageFolder exposes .targets
test_counts  = Counter(test_dataset.targets)

print("train per-class counts:", train_counts)
print("test  per-class counts :", test_counts)

# Imbalance ratio (max/min)
imb_ratio = max(train_counts.values()) / max(1, min(train_counts.values()))
print(f"train imbalance ratio: {imb_ratio:.2f}")
```

    train per-class counts: Counter({3: 7215, 4: 4965, 5: 4830, 2: 4097, 0: 3995, 6: 3171, 1: 436})
    test  per-class counts : Counter({3: 1774, 5: 1247, 4: 1233, 2: 1024, 0: 958, 6: 831, 1: 111})
    train imbalance ratio: 16.55


Since classes are imbalanced, use loss reweighting. This means that classes with fewer samples are penalized more heavily than classes with lots of samples. This prevents the model from getting "lazy" and just predicting the class with the most samples. It causes the model to "pay attention" to the smaller classes.

Use inverse frequency for weights.


```python
# Count samples per class
train_counts = Counter(train_dataset.targets) # number of samples per class
num_classes = len(train_dataset.classes) # 7 classes

counts = torch.tensor([train_counts[c] for c in range(num_classes)], dtype=torch.float) # number of samples per class in a tensor

# Inverse frequency
class_weights = 1.0 / counts # 1 / number of samples per class in a tensor
class_weights_normalized = class_weights / class_weights.sum() * num_classes  # normalize

# counts: tensor([3995.,  436., 4097., 7215., 4965., 4830., 3171.])
# class weights: # tensor([0.0003, 0.0023, 0.0002, 0.0001, 0.0002, 0.0002, 0.0003]) # higher weights to rarer classes
# class weights normalized # tensor([0.4800, 4.3982, 0.4681, 0.2658, 0.3862, 0.3970, 0.6047]) 
        # normalized s.t. avg weight is 1 (so weights are not too small) 
        # percent * 7 so total is 7, avg is 1.

# Cross-entropy with weights
criterion = torch.nn.CrossEntropyLoss(weight=class_weights_normalized)
```

### Prep Data

Load dataset into DataLoaders. DataLoaders is from the PyTorch library (torch.utils.data).
It allows for batching, suffling, processing, etc. instead of having to write loops.

In this case:
- I batched images into sets of 64. 
- I set num_workers = 8, since loading ~30k images was too slow to do without parallel processing
- I dropped the last batch since if the size was not exactly 64 images, it was causing downstream errors


```python
# Create DataLoaders
batch_size = 64
dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = 8, drop_last=True) # drop last or re-shaping gets messed up if batch doesn't have 64 images
valid_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = 8, drop_last=True) # keep unshuffled for reproducability
```

Iterate through each batch of images in the dataloders and re-shape the training image.

I want the batch of training images to be tensor of shape (64, 2304). I.e., each row is an image, each column is a pixel. 

Without re-shaping, the tensor would have shape (64, 1, 48, 48) (Batch Size, # of Channels, Length in Px, Width in Px). Since these are grayscale images, there is only 1 channel. 

When using pre-trained resnet18 models later on, these model expect 3 channels (e.g., RBG).

The tensor of labels has rank 1 and just a set of 64 labels (structurally directionless).


```python
# intermediate step -- read images/labels from data loader, but they will need to be re-shaped
train_images, train_labels = next(iter(dl))
valid_images, valid_labels = next(iter(valid_dl))
# 64 images, 64 labels

# reshape image pixels into a matrix (64 rows/images, 2304 columns/pixels)
xb = train_images.view(batch_size, 1*48*48)
yb = train_labels # just re-naming

valid_xb = valid_images.view(batch_size, 1*48*48)
valid_yb = valid_labels # just re-naming
```

### Create Functions

Create function to initialize random parameters (weights and biases):


```python
# initialize parameters
# random weights w/ mean = 0, std = 1
def init_params(size, std=1.0): return (torch.randn(size)*std).requires_grad_()
```

Create a linear function: 

logit = f(x) = x@w + b

[64 images (batch size), 2304 pixels for each image] @ [2304 pixels in one image, 7 weights/classes for each pixel] (+ bias) = [64 images, 7 logits (one for each class)]


```python
def linear1(xb): return xb@weights + bias
# [64, 2304] @ [2304, 7] = [64, 7] (64 rows/images, 7 columns/predictions (one for each class) + [7]
```

Create loss function:


```python
def loss_func(predictions, labels, num_classes):
    trgts = F.one_hot(labels, num_classes).float() # convert label classes (e.g., 2, 3, 4) to one-hot encoded values (.e.,g [0, 1, 0, 0, 0, 0, 0] for 2) 
    logp = F.log_softmax(predictions, dim=1) # take the log softmax of the logits
    logp_weighted = logp * class_weights_normalized # weight the loss due to class imbalance
    correct_class = -(trgts * logp_weighted) # keep only the log softmax / logits of the correct class (all other log softmax / logits are zero-ed out) # raw logp's are negative
    weighed_nll = correct_class.sum(dim=1) # keep the only non-zeroed out logits (negative log likelihood)
    return weighed_nll.mean() # return negative log likelihood across entire batch
```

Create back-propogation / gradient calculation function:


```python
def calc_grad(model, xb, labels, num_classes):
    predictions = model(xb)
    loss = loss_func(predictions, labels, num_classes)
    # try weighted CrossEntropyLoss
    # loss = criterion(predictions, labels)
    loss.backward()
```

Create function to train model (i.e., loop through each batch of images, calculate gradient/loss, and update parameters (weights and biases)):


```python
def train_epoch(model, lr, params, num_classes):
    for train_images, train_labels in dl: # iterate through each batch in dl
        xb = train_images.view(64, 1*48*48) # re-shape batch
        yb = train_labels # re-name labels
        calc_grad(model, xb, yb, num_classes)
        for p in params:
            p.data -= p.grad*lr # update parameter by gradient * learning rate
            p.grad.zero_() # reset gradient to zero to avoid accumulation
```

Create function to measure accuracy (not loss) of the batch of images:


```python
def batch_accuracy(model, valid_xb, valid_yb):
    logits = model(valid_xb)
    argmax_predictions = torch.argmax(logits, dim=1) # take the maximum logit (i.e., highest prediction for correct class. logits vs. probabilities don't matter)
    correct = argmax_predictions == valid_yb # calculate accuracy for each element/image
    return correct.float().mean() # calculate overall accuracy for a single batch (not the entire validation set)
```

Create function to measure accuracy (not loss) across all batches:

This function iterates through each batch, assesses the accuracy of the batch, stores each batch accuracy in a list, and then takes the average accuacy across batches to obtain the mean accuracy for one epoch.


```python
def validate_epoch(model):
    accs = [] # create empty list
    for valid_images, valid_labels in valid_dl: # iterate through each batch in valid_dl
        valid_xb = valid_images.view(batch_size, 1*48*48) # re-shape batch
        valid_yb = valid_labels # re-name labels
        accs.append(batch_accuracy(model, valid_xb, valid_yb)) # create a list of batch accuracies
    return round(torch.stack(accs).mean().item(), 4) #.item takes number out of tensor, round to 4 decimal places

# note: mean of batch accuracies is equivalent to global accuracy if all batches are the same size; in this case, we dropped last batch when creating dataloaders, so all batches will be the same size.
```

### Train and Validate Linear Model

Initialize random parameters (weights and biases):


```python
# initialize random parameters
weights = init_params((48*48,7))
bias = init_params(7)
```

Set learning rate and cosolidate weights and bias into params to pass into the train_epoch function.

Note: I originally used lr = 1. but it caused divergence in accuracy (i.e., overshooting), so I decreased the learning rate to 0.03.


```python
lr = 0.01 # learning rate # must be float
params = weights, bias # pack variables into a single tuple to loop over them as a group
```

Train model for 10 epochs:


```python
# train for 10 epochs, print accuracy after each epoch
for i in range(10):
    train_epoch(linear1, lr, params, num_classes=7)
    print(validate_epoch(linear1), end=' ')
```

    0.1477 0.1535 0.1469 0.1592 0.1611 0.1549 0.1554 0.1763 0.1699 0.1722 

Baseline is ~25% accuracy due to class imbalance (Class 4, ~7.2k samples of ~29k samples)

### Train a Simple Neural Network

Let's try a simple neural network, with two linear linears and a non-linear (relu) function in between.

In theory, this simple neural net could approximate any function (universal approximation theorem).


```python
# def simple_net(xb): 
#     res = xb@w1 + b1
#     res = res.max(tensor(0.0))
#     res = res@w2 + b2
#     return res
```


```python
def simple_net(xb):
    h = xb @ w1 + b1
    h = torch.relu(h)
    out = h @ w2 + b2
    return out  # logits
```

Initialize random parameters (weights and biases):


```python
w1 = init_params((48*48,30))
b1 = init_params(30)
w2 = init_params((30,7))
b2 = init_params(7)

# layer 1 - input: 48 x 48; output: 30 (flexible model complexity)
# layer 2 - input: 30; output: 7 (to predict 7 classes)
```

Set learning rate and consolidate parameters into a tuple:


```python
lr = 0.001 # learning rate # must be float
params = w1, b1, w2, b2 # pack variables into a single tuple to loop over them as a group
```

Train model for 10 epochs:


```python
# train for 5 epochs, print accuracy after each epoch
for i in range(10):
    train_epoch(simple_net, lr, params, num_classes=7)
    print(validate_epoch(simple_net), end=' ')
```

    0.1614 0.1578 0.1476 0.1551 0.1476 0.136 0.147 0.15 0.161 0.1643 

Baseline is ~25% accuracy due to class imbalance (Class 4, ~7.2k samples of ~29k samples)

### Creating an Optimizer Using PyTorch Modules

Create Linear Model using PyTorch module


```python
# replace linear1 function with nn.Linear module in PyTorch
# nn.Linear combines init_params and linear together

# nn.Linear(in_features, out_features, bias=True) # adds bias parameter by default
linear_model = nn.Linear(48*48,7)
```

Linear model parameters:


```python
w,b = linear_model.parameters()
w.shape,b.shape
```




    (torch.Size([7, 2304]), torch.Size([7]))



Create basic optimizer:


```python
class BasicOptim:
    def __init__(self,params,lr): self.params,self.lr = list(params),lr

    def step(self, *args, **kwargs):
        for p in self.params: p.data -= p.grad.data * self.lr

    def zero_grad(self, *args, **kwargs):
        for p in self.params: p.grad = None
```


```python
# create optimizer
lr = 0.001 # set learning rate
opt = BasicOptim(linear_model.parameters(), lr)
```

Create new train_epoch function that uses the optimizer:


```python
# re-write train_epoch function with new optimizer
def train_epoch_optim(model, num_classes):
    for train_images, train_labels in dl:
        xb = train_images.view(64, 1*48*48) # re-shape batch
        yb = train_labels # re-name labels
        calc_grad(model, xb, yb, num_classes)
        opt.step()
        opt.zero_grad()
```

Create new train_model function that uses the optimizer:


```python
# combine into training loop
def train_model_optim(model, epochs, num_classes):
    for i in range(epochs):
        train_epoch_optim(model, num_classes)
        print(validate_epoch(model), end=' ')
```

Train model using the optimizier. 

(Note: nn.Linear might use different parameter initializations (Kaiming uniform initialization); so learning rate that worked before might need to change now.)


```python
train_model_optim(model=linear_model, epochs=10, num_classes=7)
```

    0.2607 0.321 0.3089 0.3221 0.2866 0.3277 0.3313 0.3244 0.3358 0.3214 

Baseline is ~25% accuracy due to class imbalance (Class 4, ~7.2k samples of ~29k samples)

### Training the Model Using FastAi Libraries

FastAi provides the SGD class which, by default, does the same thing as BasicOptim:

Train model again but replace BasicOptim function with SGD class from FastAi:


```python
lr = 0.001
opt = SGD(linear_model.parameters(), lr)
train_model_optim(model=linear_model, epochs=10, num_classes=7)
```

    0.1826 0.2054 0.1402 0.1745 0.2312 0.1862 0.2028 0.2278 0.2518 0.2482 

Baseline is ~25% accuracy due to class imbalance (Class 4, ~7.2k samples of ~29k samples)

### Training the Model Using FastAI Learner

FastAI also provides Learner.fit, which we can use instead of train_model_optim. 

To create a Learner without using an application (such as vision_learner) we need to pass in: DataLoaders, the model, the optimization function (which will be passed the parameters), the loss function, and optionally any metrics to print:

Create DataLoaders:


```python
dls = DataLoaders(dl, valid_dl)
```

Note: Since I am pulling data directly from dls, I have to write new function that re-shape the data appropriately. (I.e., eliminate the batch channel (1) since this causes errors).

In the hand-rolled functions above, I re-shaped the data in the loops.

Create new linear model that flattens the data first:


```python
linear_model_flattened = nn.Sequential(              # container to run each module sequentially: output -> input
    nn.Flatten(start_dim=1),        # (B,1,48,48) -> (B, 2304) # (batch, channel, height, width) # collapses everything from dimension 1 onwards into single vector
    nn.Linear(48*48, 7)
)
```

Create new loss function which only takes in predictions and labels arguments:


```python
def loss_func_learner(predictions, labels):
    num_classes = 7
    trgts = F.one_hot(labels, num_classes).float() # convert label classes (e.g., 2, 3, 4) to one-hot encoded values (.e.,g [0, 1, 0, 0, 0, 0, 0] for 2) 
    logp = F.log_softmax(predictions, dim=1) # take the log softmax of the logits
    logp_weighted = logp * class_weights_normalized # weight the loss due to class imbalance
    correct_class = -(trgts * logp_weighted) # keep only the log softmax / logits of the correct class (all other log softmax / logits are zero-ed out) # raw logp's are negative
    weighed_nll = correct_class.sum(dim=1) # keep the only non-zeroed out logits (negative log likelihood)
    return weighed_nll.mean() # return negative log likelihood across entire batch
```

Create new batch_accuracy function that only takes in predictions and targets arguments:


```python
def batch_accuracy_learner(preds, targs):
    return (preds.argmax(dim=1) == targs).float().mean()

# passing in predictions and targets, not validation images and labels
# no need to run model on images and labels in this function
```

Train linear model via FastAI's learner


```python
learn_linear = Learner(dls, linear_model_flattened, opt_func=SGD, loss_func=loss_func_learner, metrics=batch_accuracy_learner)
```


```python
lr = 0.001
learn_linear.fit(10, lr=lr)
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
      <th>batch_accuracy_learner</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.892790</td>
      <td>0.881668</td>
      <td>0.233956</td>
      <td>00:51</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.867472</td>
      <td>0.865273</td>
      <td>0.227958</td>
      <td>00:52</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.867647</td>
      <td>0.866190</td>
      <td>0.201869</td>
      <td>00:51</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.833845</td>
      <td>0.856630</td>
      <td>0.307059</td>
      <td>00:47</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.831241</td>
      <td>0.852241</td>
      <td>0.313616</td>
      <td>00:45</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.827353</td>
      <td>0.834707</td>
      <td>0.284319</td>
      <td>00:50</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.835906</td>
      <td>0.855426</td>
      <td>0.287667</td>
      <td>00:50</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.825696</td>
      <td>0.829679</td>
      <td>0.326730</td>
      <td>00:47</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.829982</td>
      <td>0.840178</td>
      <td>0.274275</td>
      <td>00:51</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.828381</td>
      <td>0.838342</td>
      <td>0.220982</td>
      <td>00:47</td>
    </tr>
  </tbody>
</table>


Baseline is ~25% accuracy due to class imbalance (Class 4, ~7.2k samples of ~29k samples).

Adding non-linearity (Two linear layers with non-linear relu inbetween):


```python
w1 = init_params((48*48,30))
b1 = init_params(30)
w2 = init_params((30,7))
b2 = init_params(7)

# layer 1 - input: 48 x 48; output: 30 (flexible model complexity)
# layer 2 - input: 30; output: 7 (to predict 7 classes)
```


```python
simple_net = nn.Sequential(
    nn.Flatten(start_dim=1),
    nn.Linear(48*48,30),
    nn.ReLU(),
    nn.Linear(30,7)
)
```


```python
learn_simple_net = Learner(dls, simple_net, opt_func=SGD,
                           loss_func=loss_func_learner, metrics=batch_accuracy_learner)
```


```python
lr = 0.01
learn_simple_net.fit(15, lr=lr)
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
      <th>batch_accuracy_learner</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.885921</td>
      <td>0.896152</td>
      <td>0.193359</td>
      <td>00:53</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.872616</td>
      <td>0.884268</td>
      <td>0.214983</td>
      <td>00:53</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.863653</td>
      <td>0.875042</td>
      <td>0.237723</td>
      <td>00:55</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.867592</td>
      <td>0.866453</td>
      <td>0.268276</td>
      <td>00:54</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.861051</td>
      <td>0.863232</td>
      <td>0.281110</td>
      <td>00:51</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.851210</td>
      <td>0.852763</td>
      <td>0.273577</td>
      <td>00:54</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.839206</td>
      <td>0.845273</td>
      <td>0.270229</td>
      <td>00:52</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.839644</td>
      <td>0.845651</td>
      <td>0.306920</td>
      <td>00:55</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.853809</td>
      <td>0.837213</td>
      <td>0.274275</td>
      <td>00:54</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.853221</td>
      <td>0.834087</td>
      <td>0.288504</td>
      <td>00:56</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.819846</td>
      <td>0.837474</td>
      <td>0.338309</td>
      <td>00:54</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.835406</td>
      <td>0.828407</td>
      <td>0.258650</td>
      <td>00:54</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.824843</td>
      <td>0.834651</td>
      <td>0.234933</td>
      <td>00:56</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.806644</td>
      <td>0.835307</td>
      <td>0.353934</td>
      <td>00:58</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.840299</td>
      <td>0.822892</td>
      <td>0.252232</td>
      <td>00:50</td>
    </tr>
  </tbody>
</table>


Baseline is ~25% accuracy due to class imbalance (Class 4, ~7.2k samples of ~29k samples).

### Try Pre-trained Resnet 18

Reload the data with new transformations that ResNet expects. May not be necessary if pretrained=False.


```python
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),   # replicate gray → RGB
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])

# Load training and test datasets
# reads in .jpg images and takes class names from folder structure
train_dataset = ImageFolder(root='train', transform=transform)
test_dataset = ImageFolder(root='test', transform=transform)
```


```python
# Create DataLoaders
batch_size = 64
dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = 8, drop_last=True) # drop last or re-shaping gets messed up if batch doesn't have 64 images
valid_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = 8, drop_last=True) # keep unshuffled for reproducability
```


```python
dls = DataLoaders(dl, valid_dl)
```


```python
learn_resnet = vision_learner(dls, resnet18, pretrained=False,
                    loss_func=F.cross_entropy, metrics=accuracy, n_out=7)
```

Move learn_resnet to GPU to train faster. Training on CPU takes too long.


```python
learn_resnet.dls.to('cuda')         # move DataLoaders to GPU
learn_resnet.to_fp32().to('cuda')   # move model to GPU
```




    Sequential(
      (0): Sequential(
        (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        (4): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicBlock(
            (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (5): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (downsample): Sequential(
              (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): BasicBlock(
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (6): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (downsample): Sequential(
              (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): BasicBlock(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (7): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (downsample): Sequential(
              (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): BasicBlock(
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (1): Sequential(
        (0): AdaptiveConcatPool2d(
          (ap): AdaptiveAvgPool2d(output_size=1)
          (mp): AdaptiveMaxPool2d(output_size=1)
        )
        (1): fastai.layers.Flatten(full=False)
        (2): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): Dropout(p=0.25, inplace=False)
        (4): Linear(in_features=1024, out_features=512, bias=False)
        (5): ReLU(inplace=True)
        (6): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (7): Dropout(p=0.5, inplace=False)
        (8): Linear(in_features=512, out_features=7, bias=False)
      )
    )



Check if learn_resnet is on GPU:


```python
xb, yb = learn_resnet.dls.one_batch()
print("xb:", xb.device, "yb:", yb.device)
print("model:", next(learn_resnet.model.parameters()).device)
```

    xb: cuda:0 yb: cuda:0
    model: cuda:0


Train resnet18 model:


```python
learn_resnet.fit(10,0.01)
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
      <td>2.070354</td>
      <td>1.848544</td>
      <td>0.226004</td>
      <td>01:41</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.834533</td>
      <td>1.817640</td>
      <td>0.253627</td>
      <td>01:42</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.646091</td>
      <td>14.449536</td>
      <td>0.340402</td>
      <td>01:43</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.576655</td>
      <td>1.509799</td>
      <td>0.413225</td>
      <td>01:43</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1.479596</td>
      <td>1.412009</td>
      <td>0.454381</td>
      <td>01:44</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1.346888</td>
      <td>1.324958</td>
      <td>0.494280</td>
      <td>01:43</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1.310471</td>
      <td>1.279084</td>
      <td>0.504743</td>
      <td>01:43</td>
    </tr>
    <tr>
      <td>7</td>
      <td>1.298034</td>
      <td>1.493636</td>
      <td>0.498465</td>
      <td>01:44</td>
    </tr>
    <tr>
      <td>8</td>
      <td>1.201692</td>
      <td>1.154518</td>
      <td>0.556083</td>
      <td>01:44</td>
    </tr>
    <tr>
      <td>9</td>
      <td>1.172897</td>
      <td>1.206853</td>
      <td>0.556780</td>
      <td>01:45</td>
    </tr>
  </tbody>
</table>


Baseline is ~25% accuracy due to class imbalance (Class 4, ~7.2k samples of ~29k samples). 70-75% is the best known accuracy on this data set.

### Train ResNet18 with Custom Loss & Accuracy Functions

Try resnet with my custom functions (e.g., class-imbalance adjusted accuracy formula and accuracy function:

Since we're using custom metric for loss function, we must also move normalized weights to GPU:


```python
class_weights_normalized = class_weights_normalized.to('cuda')
```


```python
def loss_func_learner_gpu(predictions, labels):
    num_classes = 7
    trgts = F.one_hot(labels, num_classes).float() # convert label classes (e.g., 2, 3, 4) to one-hot encoded values (.e.,g [0, 1, 0, 0, 0, 0, 0] for 2) 
    logp = F.log_softmax(predictions, dim=1) # take the log softmax of the logits
    logp_weighted = logp * class_weights_normalized # weight the loss due to class imbalance
    logp_weighted = logp_weighted.to('cuda')
    correct_class = -(trgts * logp_weighted) # keep only the log softmax / logits of the correct class (all other log softmax / logits are zero-ed out) # raw logp's are negative
    weighed_nll = correct_class.sum(dim=1) # keep the only non-zeroed out logits (negative log likelihood)
    return weighed_nll.mean() # return negative log likelihood across entire batch
```

Create new learner:


```python
learn_resnet_custom = vision_learner(dls, resnet18, pretrained=False,
                    loss_func=loss_func_learner_gpu, metrics=batch_accuracy_learner, n_out=7)
```

Move model to cuda/gpu + check:


```python
learn_resnet_custom.to_fp32().to('cuda')   # move model to GPU
```




    Sequential(
      (0): Sequential(
        (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        (4): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (1): BasicBlock(
            (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (5): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (downsample): Sequential(
              (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): BasicBlock(
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (6): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (downsample): Sequential(
              (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): BasicBlock(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (7): Sequential(
          (0): BasicBlock(
            (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (downsample): Sequential(
              (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): BasicBlock(
            (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
      )
      (1): Sequential(
        (0): AdaptiveConcatPool2d(
          (ap): AdaptiveAvgPool2d(output_size=1)
          (mp): AdaptiveMaxPool2d(output_size=1)
        )
        (1): fastai.layers.Flatten(full=False)
        (2): BatchNorm1d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): Dropout(p=0.25, inplace=False)
        (4): Linear(in_features=1024, out_features=512, bias=False)
        (5): ReLU(inplace=True)
        (6): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (7): Dropout(p=0.5, inplace=False)
        (8): Linear(in_features=512, out_features=7, bias=False)
      )
    )




```python
xb, yb = learn_resnet_custom.dls.one_batch()
print("xb:", xb.device, "yb:", yb.device)
print("model:", next(learn_resnet_custom.model.parameters()).device)
```

    xb: cuda:0 yb: cuda:0
    model: cuda:0


Train resnet18 model with custom loss/accuracy functions:


```python
learn_resnet_custom.fit(20,0.001)
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
      <th>batch_accuracy_learner</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.993915</td>
      <td>0.943394</td>
      <td>0.151786</td>
      <td>01:40</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.982552</td>
      <td>1.281932</td>
      <td>0.186942</td>
      <td>01:39</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.914101</td>
      <td>0.847980</td>
      <td>0.190709</td>
      <td>01:40</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.823805</td>
      <td>0.806978</td>
      <td>0.343471</td>
      <td>01:40</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.746186</td>
      <td>0.789271</td>
      <td>0.396903</td>
      <td>01:41</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.709247</td>
      <td>0.672420</td>
      <td>0.452427</td>
      <td>01:42</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.651308</td>
      <td>0.689773</td>
      <td>0.453544</td>
      <td>01:41</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.632137</td>
      <td>0.615518</td>
      <td>0.495257</td>
      <td>01:40</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.605025</td>
      <td>0.604091</td>
      <td>0.497210</td>
      <td>01:41</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.569155</td>
      <td>0.608184</td>
      <td>0.507115</td>
      <td>01:41</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.541548</td>
      <td>0.580396</td>
      <td>0.547015</td>
      <td>01:41</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.526743</td>
      <td>0.567788</td>
      <td>0.550781</td>
      <td>01:41</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.503343</td>
      <td>0.588713</td>
      <td>0.553850</td>
      <td>01:40</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.498826</td>
      <td>0.596876</td>
      <td>0.557478</td>
      <td>01:41</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.460374</td>
      <td>0.565684</td>
      <td>0.584682</td>
      <td>01:41</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.433578</td>
      <td>0.583819</td>
      <td>0.582450</td>
      <td>01:41</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.444924</td>
      <td>0.600782</td>
      <td>0.587891</td>
      <td>01:40</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.419934</td>
      <td>0.585855</td>
      <td>0.589007</td>
      <td>01:41</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.405850</td>
      <td>0.639646</td>
      <td>0.592494</td>
      <td>01:41</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.396318</td>
      <td>0.570333</td>
      <td>0.602679</td>
      <td>01:41</td>
    </tr>
  </tbody>
</table>


Baseline is ~25% accuracy due to class imbalance (Class 4, ~7.2k samples of ~29k samples). 70-75% is the best known accuracy on this data set.

### Validate Model

Obtain overall accuracy across total validation set:


```python
from fastai.metrics import accuracy

preds,targs = learn_resnet_custom.get_preds(dl=learn_resnet_custom.dls.valid)
# opens model in evaluation mode (no dropout/grad)
# dl=learn_resnet_custom.dls.valid : set dl to validation data loader
# preds = tensor of probabilities/predictions. accuracy library takes argmax
# targs = class labels

print("Val accuracy:", accuracy(preds, targs).item())
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







    Val accuracy: 0.6010030508041382


Check/Validate the first batch of the validation set (64 images):

The validation set is ordered, so the first batch is all class 0 (angry).

{'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}


```python
# read in the first batch of validation set (data and labels)
dl = learn_resnet_custom.dls.valid
xb, yb = dl.one_batch()
```


```python
# check model class predictions
yhat = learn_resnet_custom.model(xb).argmax(1) # runs xb (one batch) through the model and takes highest prediction
yhat
```




    tensor([0, 0, 5, 5, 0, 2, 1, 0, 5, 0, 2, 4, 0, 5, 4, 0, 0, 1, 5, 0, 0, 0, 0, 0, 5, 1, 6, 5, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 2, 0, 5, 0, 0, 2, 0, 0, 5, 5, 4, 0, 0, 0, 0, 0, 0, 0, 3, 0, 5, 0, 5, 0, 6, 0],
           device='cuda:0')




```python
# compare to actual labels of the validation data set
yb
```




    tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           device='cuda:0')




```python
# calculate batch accuracy
(yhat==yb).float().mean()
```




    tensor(0.5938, device='cuda:0')



Visually inspect the data. All the images in the batch should be angry (class 0).


```python
import torchvision, matplotlib.pyplot as plt, torch

# recreate ImageNet normalization stats used during pre-processing
mean = torch.tensor([0.485,0.456,0.406], device=xb.device).view(1,3,1,1) # device = xb.device to put on same device as batch #.view (batch, channels, height, width)
std  = torch.tensor([0.229,0.224,0.225], device=xb.device).view(1,3,1,1)

# un-normalize batch to put into real pixel scale
# clamp(0,1) for friendly display
# .cpu() removes data off gpu so matplotlib can render it
imgs = (xb*std + mean).clamp(0,1).cpu() # here is where we're telling matplotlib to visualize xb

grid = torchvision.utils.make_grid(imgs, nrow=8) # tiles batch into single image with 8 images per row

plt.figure(figsize=(10,10)); plt.imshow(grid.permute(1,2,0)); plt.axis('off') # permute converts PyTorch's [C,H,W] to Matplotlib's [H,W,C]
```




    (-0.5, 1809.5, 1809.5, -0.5)




    
![png](/images/092525-output_141_1.png)
    


For good measure, let's check a different batch as well. In this case, I chose the 100th batch. It starts with class 5 (sad), but is mostly class 6 (surprise).

{'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}


```python
from itertools import islice

dl = learn_resnet_custom.dls.valid
xb, yb = next(islice(iter(dl), 99, None)) 
```


```python
yhat = learn_resnet_custom.model(xb).argmax(1)
yhat
```




    tensor([5, 5, 5, 0, 0, 4, 5, 4, 4, 5, 5, 6, 6, 6, 6, 3, 2, 6, 0, 6, 6, 6, 2, 4, 6, 6, 6, 6, 4, 6, 6, 6, 4, 6, 6, 6, 6, 6, 0, 6, 6, 3, 6, 6, 6, 6, 6, 2, 2, 6, 4, 6, 6, 6, 6, 3, 6, 2, 6, 6, 6, 6, 6, 6],
           device='cuda:0')




```python
yb
```




    tensor([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
           device='cuda:0')




```python
(yhat==yb).float().mean()
```




    tensor(0.7031, device='cuda:0')




```python
mean = torch.tensor([0.485,0.456,0.406], device=xb.device).view(1,3,1,1)
std  = torch.tensor([0.229,0.224,0.225], device=xb.device).view(1,3,1,1)
imgs = (xb*std + mean).clamp(0,1).cpu()

grid = torchvision.utils.make_grid(imgs, nrow=8)
plt.figure(figsize=(10,10)); plt.imshow(grid.permute(1,2,0)); plt.axis('off')
```




    (-0.5, 1809.5, 1809.5, -0.5)




    
![png](/images/092525-output_147_1.png)
    


### Evaluate Model Using Confusion Matrix


```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch

# preds/targs from valid loader
preds, targs = learn_resnet_custom.get_preds(dl=learn_resnet_custom.dls.valid)
y_true = targs.cpu().numpy()
y_pred = preds.argmax(1).cpu().numpy()

# class names — use ImageFolder's classes
labels = learn_resnet_custom.dls.valid.dataset.classes  # <-- key change

# confusion matrix (counts)
cm = confusion_matrix(y_true, y_pred)

# optional: normalized (row-wise)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=labels)
disp.plot(xticks_rotation=45, cmap='Blues', values_format='.2f', colorbar=True)
plt.gcf().set_size_inches(7,7)
plt.show()

# plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(xticks_rotation=45, cmap='Blues', values_format='d', colorbar=False)
plt.gcf().set_size_inches(7,7)
plt.show()
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








    
![png](/images/092525-output_149_2.png)
    



    
![png](/images/092525-output_149_3.png)
    

