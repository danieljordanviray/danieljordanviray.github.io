### Deploying a Deep Learning Model as a Public Web Application


```python
# import libraries
from fastai.vision.all import *
import gradio as gr
```


```python
# display image of car
car_path = '/content/drive/MyDrive/AI Projects/gradio_model_deployment_1/car.jpg'
im_car = PILImage.create(car_path)
im_car.thumbnail((192,192))
im_car
```




    
![png](output_2_0.png)
    




```python
# display image of bike
bike_path = '/content/drive/MyDrive/AI Projects/gradio_model_deployment_1/bike.jpg'
im_bike = PILImage.create(bike_path)
im_bike.thumbnail((192,192))
im_bike
```




    
![png](output_3_0.png)
    




```python
# display image of tree
tree_path = '/content/drive/MyDrive/AI Projects/gradio_model_deployment_1/tree.jpg'
im_tree = PILImage.create(tree_path)
im_tree.thumbnail((192,192))
im_tree
```




    
![png](output_4_0.png)
    




```python
# display image of all three - (i.e., ambigious)
all_path = '/content/drive/MyDrive/AI Projects/gradio_model_deployment_1/all.jpg'
im_all = PILImage.create(all_path)
im_all.thumbnail((192,192))
im_all
```




    
![png](output_5_0.png)
    




```python
# display image of flower
flower_path = '/content/drive/MyDrive/AI Projects/gradio_model_deployment_1/flower.jpg'
im_flower = PILImage.create(flower_path)
im_flower.thumbnail((192,192))
im_flower
```




    
![png](output_6_0.png)
    




```python
# import previously trained model (.pkl)
learn = load_learner('/content/drive/MyDrive/AI Projects/gradio_model_deployment_1/export.pkl')
```

    /usr/local/lib/python3.11/dist-packages/fastai/learner.py:455: UserWarning: load_learner` uses Python's insecure pickle module, which can execute malicious arbitrary code when loading. Only load files you trust.
    If you only need to load model weights and optimizer state, use the safe `Learner.load` instead.
      warn("load_learner` uses Python's insecure pickle module, which can execute malicious arbitrary code when loading. Only load files you trust.\nIf you only need to load model weights and optimizer state, use the safe `Learner.load` instead.")
    


```python
learn.predict(im_car)
# returns: ('prediction', 'tensor position', 'probability(bike, car, tree)')
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










    ('car', tensor(1), tensor([1.3583e-06, 1.0000e+00, 6.1913e-09]))




```python
learn.predict(im_bike)
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










    ('bike', tensor(0), tensor([9.9999e-01, 7.5860e-06, 1.0868e-07]))




```python
learn.predict(im_tree)
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










    ('tree', tensor(2), tensor([0.1054, 0.0544, 0.8402]))




```python
learn.predict(im_all)
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










    ('car', tensor(1), tensor([4.7532e-02, 9.5226e-01, 2.1252e-04]))




```python
# gradio wants dictionary of categories
# and probablities associated with each one (must be float). Gradio cannot handle numpy or tensors
categories = ('bike', 'car', 'tree')

def classify_image(img):
  pred, idx, probs = learn.predict(img)
  return dict(zip(categories, map(float,probs)))

# map(float, probs) converts the probabilities from tensors to plain Python floats (which are easier to work with or print).

# zip(categories, ...) pairs each class label from your categories list (['bike', 'car', 'tree']) with the corresponding probability.

# dict(...) turns those pairs into a dictionary.

# So if the probabilities are [0.05, 0.90, 0.05], the return value will be:
# {'bike': 0.05, 'car': 0.90, 'tree': 0.05}
```


```python
# test classify_image() function
classify_image(im_car)
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










    {'bike': 1.358293275188771e-06,
     'car': 0.9999986886978149,
     'tree': 6.191251067377834e-09}




```python
# build a simple image classification demo where a user can upload or click example images, and model predicts the label using classify_image() function

# input PIL images
image = gr.Image(height=250,width=250)

# Create a label output component that displays the model's predictions.
# It shows the predicted class and probabilities for all classes in a clean format.
# Gradio parses the output from the classify_image() function, a dictionary of categories and probabilities
label = gr.Label()

# Add clickable examples to the interface.
examples = [car_path, bike_path, tree_path, all_path, flower_path]

# create gradio interface
# fn=classify_image: The function to call when the user inputs an image.
intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)

# launch Gradio interface
intf.launch(inline=False)

```

    It looks like you are running Gradio on a hosted a Jupyter notebook. For the Gradio app to work, sharing must be enabled. Automatically setting `share=True` (you can turn this off by setting `share=False` in `launch()` explicitly).
    
    Colab notebook detected. To show errors in colab notebook, set debug=True in launch()
    * Running on public URL: https://96a9364e817450727a.gradio.live
    
    This share link expires in 1 week. For free permanent hosting and GPU upgrades, run `gradio deploy` from the terminal in the working directory to deploy to Hugging Face Spaces (https://huggingface.co/spaces)
    




    



Code needed to create script / app on HuggingFace:


```python
# import libraries
from fastai.vision.all import *
import gradio as gr

# import previously trained model (.pkl)
learn = load_learner('/content/drive/MyDrive/AI Projects/gradio_model_deployment_1/export.pkl')

# gradio wants dictionary of categories
# and probablities associated with each one (must be float). Gradio cannot handle numpy or tensors
categories = ('bike', 'car', 'tree')

def classify_image(img):
  pred, idx, probs = learn.predict(img)
  return dict(zip(categories, map(float,probs)))

# build a simple image classification demo where a user can upload or click example images, and model predicts the label using classify_image() function

# input PIL images
image = gr.Image(height=250,width=250)

# Create a label output component that displays the model's predictions.
# It shows the predicted class and probabilities for all classes in a clean format.
# Gradio parses the output from the classify_image() function, a dictionary of categories and probabilities
label = gr.Label()

# Add clickable examples to the interface.
examples = [car_path, bike_path, tree_path, all_path, flower_path]

# create gradio interface
# fn=classify_image: The function to call when the user inputs an image.
intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)

# launch Gradio interface
intf.launch(inline=False)
```

    Collecting nbdev
      Downloading nbdev-2.4.2-py3-none-any.whl.metadata (10 kB)
    Requirement already satisfied: packaging in /usr/local/lib/python3.11/dist-packages (from nbdev) (24.2)
    Collecting fastcore>=1.8.0 (from nbdev)
      Downloading fastcore-1.8.2-py3-none-any.whl.metadata (3.7 kB)
    Collecting execnb>=0.1.12 (from nbdev)
      Downloading execnb-0.1.14-py3-none-any.whl.metadata (3.6 kB)
    Requirement already satisfied: astunparse in /usr/local/lib/python3.11/dist-packages (from nbdev) (1.6.3)
    Collecting ghapi>=1.0.3 (from nbdev)
      Downloading ghapi-1.0.6-py3-none-any.whl.metadata (13 kB)
    Collecting watchdog (from nbdev)
      Downloading watchdog-6.0.0-py3-none-manylinux2014_x86_64.whl.metadata (44 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m44.3/44.3 kB[0m [31m2.0 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting asttokens (from nbdev)
      Downloading asttokens-3.0.0-py3-none-any.whl.metadata (4.7 kB)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.11/dist-packages (from nbdev) (75.2.0)
    Requirement already satisfied: PyYAML in /usr/local/lib/python3.11/dist-packages (from nbdev) (6.0.2)
    Requirement already satisfied: ipython in /usr/local/lib/python3.11/dist-packages (from execnb>=0.1.12->nbdev) (7.34.0)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in /usr/local/lib/python3.11/dist-packages (from astunparse->nbdev) (0.45.1)
    Requirement already satisfied: six<2.0,>=1.6.1 in /usr/local/lib/python3.11/dist-packages (from astunparse->nbdev) (1.17.0)
    Collecting jedi>=0.16 (from ipython->execnb>=0.1.12->nbdev)
      Downloading jedi-0.19.2-py2.py3-none-any.whl.metadata (22 kB)
    Requirement already satisfied: decorator in /usr/local/lib/python3.11/dist-packages (from ipython->execnb>=0.1.12->nbdev) (4.4.2)
    Requirement already satisfied: pickleshare in /usr/local/lib/python3.11/dist-packages (from ipython->execnb>=0.1.12->nbdev) (0.7.5)
    Requirement already satisfied: traitlets>=4.2 in /usr/local/lib/python3.11/dist-packages (from ipython->execnb>=0.1.12->nbdev) (5.7.1)
    Requirement already satisfied: prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 in /usr/local/lib/python3.11/dist-packages (from ipython->execnb>=0.1.12->nbdev) (3.0.51)
    Requirement already satisfied: pygments in /usr/local/lib/python3.11/dist-packages (from ipython->execnb>=0.1.12->nbdev) (2.19.1)
    Requirement already satisfied: backcall in /usr/local/lib/python3.11/dist-packages (from ipython->execnb>=0.1.12->nbdev) (0.2.0)
    Requirement already satisfied: matplotlib-inline in /usr/local/lib/python3.11/dist-packages (from ipython->execnb>=0.1.12->nbdev) (0.1.7)
    Requirement already satisfied: pexpect>4.3 in /usr/local/lib/python3.11/dist-packages (from ipython->execnb>=0.1.12->nbdev) (4.9.0)
    Requirement already satisfied: parso<0.9.0,>=0.8.4 in /usr/local/lib/python3.11/dist-packages (from jedi>=0.16->ipython->execnb>=0.1.12->nbdev) (0.8.4)
    Requirement already satisfied: ptyprocess>=0.5 in /usr/local/lib/python3.11/dist-packages (from pexpect>4.3->ipython->execnb>=0.1.12->nbdev) (0.7.0)
    Requirement already satisfied: wcwidth in /usr/local/lib/python3.11/dist-packages (from prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0->ipython->execnb>=0.1.12->nbdev) (0.2.13)
    Downloading nbdev-2.4.2-py3-none-any.whl (70 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m70.1/70.1 kB[0m [31m3.3 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading execnb-0.1.14-py3-none-any.whl (13 kB)
    Downloading fastcore-1.8.2-py3-none-any.whl (78 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m78.2/78.2 kB[0m [31m6.4 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading ghapi-1.0.6-py3-none-any.whl (62 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m62.4/62.4 kB[0m [31m5.3 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading asttokens-3.0.0-py3-none-any.whl (26 kB)
    Downloading watchdog-6.0.0-py3-none-manylinux2014_x86_64.whl (79 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m79.1/79.1 kB[0m [31m6.3 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading jedi-0.19.2-py2.py3-none-any.whl (1.6 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.6/1.6 MB[0m [31m27.8 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: watchdog, jedi, fastcore, asttokens, ghapi, execnb, nbdev
      Attempting uninstall: fastcore
        Found existing installation: fastcore 1.7.29
        Uninstalling fastcore-1.7.29:
          Successfully uninstalled fastcore-1.7.29
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    fastai 2.7.19 requires fastcore<1.8,>=1.5.29, but you have fastcore 1.8.2 which is incompatible.[0m[31m
    [0mSuccessfully installed asttokens-3.0.0 execnb-0.1.14 fastcore-1.8.2 ghapi-1.0.6 jedi-0.19.2 nbdev-2.4.2 watchdog-6.0.0
    
