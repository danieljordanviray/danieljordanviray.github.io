### Computer Vision -- Lane Detection Using Hough Transform


```python
# Code adapted from Udemy's "The Complete Self-Driving Car Course - Applied Deep Learning course"
```

The objective of this AI program is to detect traffic lane lines using actual footage from my dashcam.

Here is a sample of the raw footage we are working with:


```python
# packages
import cv2
import numpy as np
import matplotlib.pyplot as plt
```


```python
# Load raw image
raw = cv2.imread('test_image2_w_shadows.jpg')
```


```python
# Create display_image() function
def display_image(img, title):
    
    # Resize the image (width, height)
    resized = cv2.resize(img, (1206, 643)) 
    
    # Convert BGR to RGB (OpenCV loads as BGR, matplotlib expects RGB)
    resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Display inline
    plt.imshow(resized_rgb)
    plt.axis('off')  # Hide axes
    plt.title(title)
    plt.show()
```


```python
# Show raw image
display_image(raw,"raw")
```


    
![png](output_7_0.png)
    



```python
# Convert image to HSV
hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)

display_image(hsv, "hsv")
```


    
![png](output_8_0.png)
    



```python
# Define the range for shadow detection
lower_shadow = np.array([0, 0, 0])
upper_shadow = np.array([180, 255, 120])

# Create a mask for shadow regions
shadow_mask = cv2.inRange(hsv, lower_shadow, upper_shadow)

display_image(shadow_mask, "shadow_mask")
```


    
![png](output_9_0.png)
    


Next step is to transform the raw footage for better edge (i.e., traffic lane lines) detection:

(1) Convert 


```python

```


```python

```
