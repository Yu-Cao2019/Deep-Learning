# Loading Data
1. `wget <link>` command can be used to download the data into cloud.
2. Use package cv2 to load image file (see the example).

#### Example of package cv2
```
import cv2

src = cv2.imread('/home/ex.png', cv2.IMREAD_UNCHANGED)

# percent by which the image is resized
scale_percent = 50

# calculate the 50 percent of original dimensions
width = int(src.shape[1] * scale_percent / 100)
height = int(src.shape[0] * scale_percent / 100)

#dsize = (width, height)

# resize image
output = cv2.resize(src, dsize)

cv2.imwrite('/home/resize_ex.png', output)
```
- `cv2.imread(path)` reads the given file in `cv2.IMREAD_UNCHANGED` and returns a numpy array
- `scr.shape[1]` gives the width of the source image.
- `cv2.resize(scr, dsize)` resize the image `src` to the size `dsize` and returns numpy array.
- `cv2.imwrite(path)` writes the output to a file.

Source: [Python OpenCV cv2 Resize Image](https://pythonexamples.org/python-opencv-cv2-resize-image/)

3. Use package cv2 to load image file (see the example).
