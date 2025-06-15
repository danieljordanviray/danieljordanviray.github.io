### Building a Self-Driving Car - Removing Shadows from Images to Prevent Lane Detection Errors

![png](/images/rsi_output_8_0.png)

*NB: Code adapted from Udemy's "The Complete Self-Driving Car Course - Applied Deep Learning course"*

*NB: Some text and code generated with AI for speed and convenience.*

### Objective
I am trying to build a self-driving car. 

The first step to that is understanding computer vision. 

While creating my first traffic lane detection pipeline, I noticed that shadows were interfering with the edge detection algorithms. Hence, I created a pipeline that removes shadows from images and then lets the computer process these new images without shadows. 

This prevents errors from the computer mistaking shadows for traffic lanes. We definitely would not want that happening in a self-driving car. 

### Raw image

Here is the raw image we're working with (we'll evolve to videos later):


```python
# load packages
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


    
![png](/images/rsi_output_8_0.png)
    


### Transforming and preparing the image

The goal is to have the computer detect the traffic lane lines using edge detection algorithms. 

However, you can see that there are shadows on the road from the lamp posts. This will interfere with the edge detection algorithms; the computer may interpret the shadows as a lane line because it shows up as a sharp edge in the image.

So we will need to transform the image to get rid of the shadows and prepare it for edge detection.

First, we convert the image pixel information from BGR/RGB (Blue, Green, Red) to HSV (Hue, Saturation, Value).


```python
# Convert image to HSV
hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)

display_image(hsv, "hsv")
```


    
![png](/images/rsi_output_12_0.png)
    


Why?

In BGR, each pixel is defined by how much blue, green, and red it contains. BGR is great for rendering images, but not ideal for tasks like detecting shadows. This is because RBG mixes color and intensity in a way that doesn’t reflect how humans perceive color.

In HSV:

- Hue (H) = The actual color (e.g., red, blue, green) — measured in degrees [0–180] in OpenCV.

- Saturation (S) = How pure the color is (0 is gray; 255 is fully saturated).

- Value (V) = Brightness of the color (0 is black; 255 is bright).

Why Use HSV for Shadow Detection?

Shadows mainly affect the brightness (Value channel) — not necessarily the hue or saturation. By separating the color (Hue, Saturation) from the brightness (Value), HSV allows you to:

- Isolate shadows: Shadows appear as low values in the V channel.

- Ignore color shifts caused by shadows: Even if a shadow changes how a color looks in BGR, in HSV you can still detect that it has low brightness.

Now what we will do is create a "mask" on the HSV image.

A mask is grayscale image that we put on top of the HSV image. Each pixel in the mask is binary (keep or discard), which essentially allows us to filter through/out pixels in the hsv image we want to keep/discard. 

In the mask:

 - Pixels with value 255 (white) indicate areas to keep or process.

 - Pixels with value 0 (black) indicate areas to ignore or discard.

So, now, we will tell OpenCV which pixels we want to let through in the mask (remember, we are trying to get rid of the shadow):


```python
# Define the range for shadow detection
# [H, S, V]
lower_shadow = np.array([0, 0, 0])
upper_shadow = np.array([180, 255, 110])
```

Here, we allow: 

- Any H value to flow through [0-180]

- Any S value to flow through [0-255]

- Only dim V values (shadows) to flow through [0-110] from a [0-255] range

(Remember, we converted from RBG to HSV specifically so that we could isolate the V).

Now, we create the mask:

The inRange() function checks each pixel from the hsv image and compares it against the shadow range we defined above.

- If the hsv pixel is within range, then that pixel in the output mask is set to 255 (white).

- If the hsv pixel is out of range, then it's set to 0 (black).

Remember, that we are only allowing dark pixels to flow through, so they should show up as white on the mask.


```python
# Create a mask for shadow regions
shadow_mask = cv2.inRange(hsv, lower_shadow, upper_shadow)

display_image(shadow_mask, "shadow_mask")
```


    
![png](/images/rsi_output_20_0.png)
    


Now, we can see that the shadow does not appear as an edge in the mask. 

Nevertheless,
- the dark pixels in the raw image are white in the mask

- and the light pixels in the raw image are black in the mask.

Therefore, we want to invert the mask (black -> white and white -> black pixels) so that we can keep the light pixels in the raw image of further processing. (I.e., we won't let the shadow pixels flow through the mask).


```python
# Invert the mask to select non-shadow regions
non_shadow_mask = cv2.bitwise_not(shadow_mask)

display_image(non_shadow_mask, "non_shadow_mask")
```


    
![png](/images/rsi_output_22_0.png)
    


Now we put the mask on top of the raw/original image. This effectively blocks out all the dark pixels in the original image. 


```python
# Apply the mask to the original image

# bitwise operation: raw * raw[with mask] since OpenCV doesn't have a native mask function
masked_image = cv2.bitwise_and(raw, raw, mask=non_shadow_mask)

display_image(masked_image, "masked_image")
```


    
![png](/images/rsi_output_24_0.png)
    


Next, we convert the image to gray scale because most image processing algorithms work on pixel intensity, not color. This simplifies the image processing.




```python
# convert to grayscale
gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)

display_image(gray, "gray")
```


    
![png](/images/rsi_output_26_0.png)
    


Next, we feed the image through the Canny edge detection algorithm.

Canny is used to detect edges — places in the image where pixel intensity changes sharply (like the boundary of an object).

The result is a binary image where:

White pixels (255) represent edges

Black pixels (0) represent non-edges


```python
# canny = cv2.Canny(image, threshold1, threshold2)
t1 = 30
t2 = 3*t1
canny = cv2.Canny(gray, t1, t2)

display_image(canny, "canny")
```


    
![png](/images/rsi_output_29_0.png)
    


Canny edge detection works by looking at intensity gradients (how quickly pixel values change). Then it uses thresholding (2nd and 3rd arguments in the function) to classify pixels:

- If pixel gradient ≥ threshold2: Pixel is kept -- it's considered a strong edge. 


- If pixel gradient is between threshold1 and threshold2: Pixel is conditionally kept -- it's kept only if connected to a strong edge.


- If pixel gradient < threshold1: Pixel is discarded -- it is not considered an edge but rather noise.

Common threshold1 Values in Practice:

- Natural scenes; 50–100; Lots of soft edges and lighting variations

- Text, printed docs; 100–150; Sharp, high-contrast lines

- Industrial/technical; 150–200+; Very crisp, low-noise edges (e.g., CAD diagrams)


- Auto (adaptive);	~66% of median;	Based on image histogram for dynamic detection

NB: threshold2 = 3*threshold1 is a common heuristic that gives a good edge/noise tradeoff.

Now, notice how there is no shadow in the image that the computer sees!

### Applying the image processing and edge detection algorithm to video

And here is the pre-processing steps applied to video! Video is just a series of images played in series.


```python
# create canny function
def canny(image):
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for shadow detection (adjust as needed)
    lower_shadow = np.array([0, 0, 0])
    upper_shadow = np.array([180, 255, 110])

    # Create a mask for shadow regions
    shadow_mask = cv2.inRange(hsv, lower_shadow, upper_shadow)

    # Invert the mask to select non-shadow regions
    non_shadow_mask = cv2.bitwise_not(shadow_mask)

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(image, image, mask=non_shadow_mask)

    # convert to grayscale
    gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)

    # apply canny edge detection algorithm
    t1 = 30
    t2 = 3*t1
    canny = cv2.Canny(gray, t1, t2)

    return canny

# capture video object
cap = cv2.VideoCapture("test_video.mov")
while(cap.isOpened()):
    _, frame = cap.read()
    small_frame = cv2.resize(frame, (1206, 643))

    # transform image
    canny_image = canny(small_frame)

    # show image
    cv2.imshow("result", canny_image)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```
### Next steps

In the next project, we will use the processed image to detect lane lines accurately. 
