### Building a Self-Driving Car - Lane Detection Using Hough Transform (Inferring Straight Lines in an Image)

*NB: Code adapted from Udemy's "The Complete Self-Driving Car Course - Applied Deep Learning course"*

*NB: Some text and code generated with AI for speed and convenience.*

#### In the Previous Blog Post:

We processed my raw dashcam image into something that the computer can use for detecting lane lines accurately. 

Namely, (1): We removed shadows from the image that interfere with edge detection algorithms, and (2): we applied the Canny function to the image in (1) to detect edges in the image. (As a reminder, the Canny function finds edges by looking a rapid change in pixel intensity).

Here is the link to the previous blog post: https://danieljordanviray.github.io/2025/06/02/Building-a-Self-Driving-Car-Removing-Shadows-from-Images-to-Prevent-Lane-Detection-Errors.html

Here is what the raw dashcam image looked like:


```python
# load packages
import cv2
import numpy as np
import matplotlib.pyplot as plt
```


```python
# Load raw image
raw = cv2.imread('test_image2_w_shadows.jpg')

# dimensions of raw image
# height, width, channels (e.g., 3 for RBG, 1 for gray scale)
# raw.shape
```


```python
# Create display_image() function
def display_image(img, title):

    # Convert BGR to RGB (OpenCV loads as BGR, matplotlib expects RGB)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Display inline
    plt.imshow(rgb)
    plt.axis('on')  # Hide axes
    plt.title(title)
    plt.show()
```


```python
display_image(raw,"raw_image")
```


    
![png](output_9_0.png)
    


And here is what the Canny image looked like:


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
```


```python
# Show canny image
canny_image = canny(raw)
display_image(canny_image,"canny_image")
```


    
![png](output_12_0.png)
    


Now, in this blog post, we are going to use this image to detect the traffic lane lines using the Hough Transform algorithm.

#### What is the Hough Transform Algorithm?

The Hough Transform is a classical algorithm in computer vision used to detect straight lines in images — it's especially useful when the lines are partially occluded, broken, or not perfectly continuous.

When analyzing real-world images, especially those with noise or partial edges, it can be hard to detect straight lines directly from pixels.

How a human might do it:
If you printed the image and looked at it, you’d scan the image and say:

“Ah, I can see there’s a line starting here and going that way.”
Even if the line is dashed or broken, you can imagine the full line.

That’s what the Hough Transform tries to do — find full straight lines, even when parts of them are missing.

#### Region of Interest

So now, the first action we are going to perform is cropping the image so that the computer only sees the road ahead. Just like putting blinders on a horse.

Get image dimensions because we need the dimensions to crop the view correctly:


```python
# Define image dimensions
print("Img dimensions : ", canny_image.shape)
# height, width, channels (e.g., 3 for RBG, 1 for gray scale)
# If only two values, then image is already in grayscale
```

    Img dimensions :  (1177, 2553)
    

Now, we're going to create a triangle in the computer's field of view by giving the computer three coordinates. We're going to let the dashcam image within the triangle be visible, and we're going to block out the rest of the dashcam image. 
(Note: I manually adjusted the coordinates of the triangle based on visual inspection of the image).

Now the image looks like this:


```python
# Create polygon (triangle)
# coodinate1, coordinate2, coordinate3

# Obtain height of the image in pixels (note that the computer inverts the y-axs)
height = canny_image.shape[0]

# Create a polygon (triangle) with three coordinates
polygons = np.array([
[(500,height), (1850, height), (1400,890)]
])

# Create a blank (black) single-channel mask the same size as canny_image
mask = np.zeros_like(canny_image)

# Fill the triangle area in the mask with white (255)
# 255 = white = basically means "1" or "keep" when we use the bitwise_and function
# 0 = back = "discard" the pixel when we use the bitwise_and function
cv2.fillPoly(mask, polygons, 255)

# Apply the mask to keep only the triangle region in the image (i.e., the region of interest)
masked_image = cv2.bitwise_and(canny_image, mask)

# Display the result
display_image(masked_image,"region_of_interest")
```


    
![png](output_21_0.png)
    


Now, we'll feed in this cropped image into the Hough Transform algorithm function to detect the lines:


```python
lines = cv2.HoughLinesP(masked_image, 2, np.pi/180, 40, np.array([]), minLineLength=40, maxLineGap=100)
```

Here are the paramters to the HoughLinesP function in cv2:

| Parameter          | Definition                                                                                     | Explanation                                                                                       |
| ------------------ | ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `masked_image`     | Input binary image (typically edges from `cv2.Canny`)                                       | The black-and-white image where you're trying to find straight lines                                        |
| `2`                | `rho`: distance resolution in pixels (e.g., 2 pixels)                                       | Scans every 2 pixels; smaller number = finer scan, but slower                                               |
| `np.pi/180`        | `theta`: angle resolution in radians (1° here)                                              | Checks lines at 1° angle steps; smaller steps = more precise angle detection                                |
| `40`               | `threshold`: minimum number of intersecting votes in the Hough accumulator to detect a line | Needs at least 40 edge points to agree before calling it a real line                                        |
| `np.array([])`     | Placeholder for output array (not used anymore, can just be omitted or `None`)              | Just a placeholder—doesn't matter for modern OpenCV                                                         |
| `minLineLength=40` | Minimum length of line (in pixels) to keep                                                  | Only keeps lines that are at least 40 pixels long                                                           |
| `maxLineGap=60`    | Maximum gap between line segments to connect them into a single line                        | If two pieces of a line are close (≤ 60 pixels apart), they're combined into one line                       |



#### Displaying the Hough Transform Lines on a Blank (Black) Canvas

Now, we will create a blank (black) canvas of the same dimensions of our image. And then draw the lines detected by the Hough Transform onto the blank canvas.


```python
# parameters of display_lines() function:
# image = image where we want to detect lines
# lines = A list or array of lines, typically returned by cv2.HoughLinesP, where each line is in the format [[x1, y1, x2, y2]]

def display_lines(image, lines):
    # create black canvas with same dimensions as image
    line_image = np.zeros_like(image)

    # convert line_image into a 3 channel (colored) image, so that I can draw lines in color. 
    # If the canvas is one channel, my lines will only be one channel (i.e., white).
    if len(image.shape) == 2: # if the image is in grayscale (only 2 values, height and width, no color channel), convert it to color
        line_image_colored = cv2.cvtColor(line_image, cv2.COLOR_GRAY2BGR)
    else: # if it's already colored, keep it in color
        line_image_colored = line_image

    # Prevents an error in case no lines were detected (i.e., lines is None).
    # If no lines, it will skip drawing.
    if lines is not None:

        # Iterate through each detected line
        for line in lines:
        
            # reshape from 2d array (i.e., matrix) to 1d array (i.e., row).
            # "line" comes in as a 2D array, e.g., [[x1, y1, x2, y2]]
            # .reshape(4) flattens it into a 1D array: [x1, y1, x2, y2]    
            # important for unpacking data
            x1, y1, x2, y2 = line.reshape(4)
            
            # draw lines onto black canvas
            # 4th argument = color of the line in BGR format, not RGB. e.g., (255, 0, 0) means blue in BGR (Blue, Green, Red).
            # 5th argument = thickness (# of pixels wide). 1 or 2 = thin, useful for high-res images or overlays.
            cv2.line(line_image_colored, (x1,y1), (x2,y2), (0, 0, 255), 5)

    return line_image_colored

line_image = display_lines(canny_image, lines)

display_image(line_image,"line_image")
```


    
![png](output_28_0.png)
    


#### Overlaying the Hough Transform Lines on the Canny Image

Now, let's overlay these lines onto the canny image:

(Note: when combining images, images must be of the same dimensions and channel (i.e., RBG or grayscale)). 

So we will convert the canny_image (currently in grayscale) to color. The line_image was converted to color already.


```python
print("dimensions: height, width, channels (3 for RGB/BGR, 1 for grayscale)")
print("canny img shape:", canny_image.shape)
print("line img shape:", line_image.shape)
```

    dimensions: height, width, channels (3 for RGB/BGR, 1 for grayscale)
    canny img shape: (1177, 2553)
    line img shape: (1177, 2553, 3)
    


```python
# overlay/blend canny image with lines images
canny_img_w_lines = cv2.addWeighted(cv2.cvtColor(canny_image, cv2.COLOR_GRAY2BGR), 0.75, line_image, 1, 1)
```

| Parameter | Meaning                                                     |
| --------- | ----------------------------------------------------------- |
| `src1`    | First image (here: `canny_image`)                           |
| `alpha`   | Weight for the first image (0.75 = 75%)                     |
| `src2`    | Second image (here: `line_image`)                           |
| `beta`    | Weight for the second image (1 = 100%)                      |
| `gamma`   | Scalar added to each pixel value after blending (here: `1`) |



```python
# display canny image with lines
display_image(canny_img_w_lines,"canny_img_w_lines")
```


    
![png](output_35_0.png)
    


Now, let's overlay the lines on the original image:


```python
# display raw image with lines
raw_w_lines = cv2.addWeighted(raw, 0.75, line_image, 1, 1)

display_image(raw_w_lines,"raw_w_lines")
```


    
![png](output_37_0.png)
    


Now, we can see that the traffic lanes are detected. Especially notice the broken traffic line on the right. The Hough Transform algorithm inferred that this is a single, continuous line, just as a human would do.

However, notice that there are multiple lines generated from the Hough Transform. We want to smooth out those lines into just a single line. Therefore, we will take the averages of these lines and create a singe line for each traffic line.

#### Averaging the Multiple Hough Transform Lines into One Line for Each Side


```python
# Converts slope & intercept into pixel coordinates for drawing a line on the image
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters

    # Set y1 to be the bottom of the image
    y1 = image.shape[0]

    # y2 is how far we're going to draw the line up the image
    # I did a visual inspection of the image and see what if the traffic lanes best
    # Since Hough Transform lines are straight, and real traffic lanes are curved, I truncated the lines up until the point they were valid in context.
    y2 = int(y1 * (83 / 100))

    # Solve for x1 and x2 using y = mx + b → x = (y - b) / m
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    # Return coordinates as array [x1, y1, x2, y2]
    return np.array([x1, y1, x2, y2])

# Processes multiple lines and returns one average left line and one average right line
def average_slope_intercept(image, lines):
    left_fit = []   # will hold slope & intercept for left-leaning lines
    right_fit = []  # will hold slope & intercept for right-leaning lines

    for line in lines:
        # reshape from 2D array [[x1, y1, x2, y2]] to flat array [x1, y1, x2, y2]
        x1, y1, x2, y2 = line.reshape(4)

        # Fit a 1st-degree polynomial (i.e., line) to the x,y points → returns [slope, intercept]
        parameters = np.polyfit((x1, x2), (y1, y2), 1) # third parameter is polynomial degree (i.e., degree 1 is a line)
        slope = parameters[0]
        intercept = parameters[1]

        # Classify line based on slope: left (negative), right (positive)
        # Note: in image, y-axis starts from 0 at the top (hence what looks like a pos/neg slope is a actually a neg/pos slope
        if slope < 0: # neg slope = looks like pos slope on left traffic lane
            left_fit.append((slope, intercept))
        else: # pos slope = looks like neg slope on right traffic lane
            right_fit.append((slope, intercept))

    # Average the slopes and intercepts of left and right lines
    # Axis = 0 means to average each column (slope and intercept) independently. 
    # For example, if left_fit = [(-0.7, 100), (-0.65, 105), (-0.72, 98)] (list of tuples), then average first and second item in each tuple separately.
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    # Convert average slope/intercept to drawable pixel coordinates
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)

    # Return both lines as a NumPy array of shape (2, 4)
    return np.array([left_line, right_line])

```

So, here are the [x1, y1, x2, y2] coordinates for the generated lines that we are going to overlay on the raw dashcam image:


```python
# create average of lines based on output lines from the Hough Transform algo
averaged_lines = average_slope_intercept(raw, lines)

# print coordinates
print("averaged lines coordinates: \n", averaged_lines)

print("left_line coordinates: \n", averaged_lines[0])

print("right_line coordinates: \n",averaged_lines[1])
```

    averaged lines coordinates: 
     [[ 743 1177 1150  976]
     [1660 1177 1466  976]]
    left_line coordinates: 
     [ 743 1177 1150  976]
    right_line coordinates: 
     [1660 1177 1466  976]
    

Now, let's draw these lines on top of the raw dashcam footage:


```python
# create line image (i.e., lines on black canvas)
# the only reason for passing in an image here is to create a black canvas of the same dimensions in the display_lines() function
average_line_image = display_lines(raw, averaged_lines)

# combine dashcam image and average_line_image
raw_w_avg_lines = cv2.addWeighted(raw, 0.65, average_line_image, 1, 1)
```


```python
display_image(raw_w_avg_lines,"raw_w_avg_lines")
```


    
![png](output_45_0.png)
    


#### Is Hough Transform Used in the Real-World? Limitations to Hough Transform

Now, we have one line each for the left and right lane. However, notice that the right lane is not perfectly straight. It curves off towards the right in the image. 

This is a limitation of the Hough Transform algo, which predicts straight lines. If I were to extend the right Hough Transform line further into the image (or to infinity), it would veer off course. 

Here are other limitations to the Hough Transform, which is now rarely used in real, modern self-driving cars:

| Limitation                 | Why It Matters in Driving                                                                                        |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **Assumes straight lines** | Most roads are curved, and Hough Transform detects only **straight** lines.                                      |
| **Sensitive to noise**     | Shadows, worn-out paint, and other cars create **false edges**.                                                  |
| **Hard-coded thresholds**  | It needs careful tuning of parameters like angle resolution, vote thresholds, etc., which don’t generalize well. |
| **Can't track or predict** | It doesn’t understand temporal context — every frame is treated separately.                                      |


#### Is the Hough Transform Algo Artifical Intelligence?

For the record, this lane detection system is not AI. It is classical computer vision where we are actually hand-coding the rules-based algorithm. 

AI would be feeding the computer data and having the computer learn the rules itself for lane detection.

Nevertheless, this is a great foundation in computer vision to get started in more advanced AI lane detection systems.
