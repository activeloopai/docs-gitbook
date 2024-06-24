---
description: >-
  Converting an object detection dataset to Deep Lake format is a great way to
  get started with datasets of increasing complexity.
---

# Creating Object Detection Datasets

## How to convert a YOLO object detection dataset to Deep Lake format

#### This tutorial is also available as a [Colab Notebook](https://colab.research.google.com/drive/1ExJsPHoqrs0XS3KzVPZymkGPHgZ1o6hx?usp=sharing)

Object detection using bounding boxes is one of the most common annotation types for Computer Vision datasets. This tutorial demonstrates how to convert an object detection dataset from YOLO format to Deep Lake, and a similar method can be used to convert object detection datasets from other formats such as COCO and PASCAL VOC.

### Create the Deep Lake Dataset

The first step is to download the small dataset below called _animals object detection_.

{% file src="../../../../.gitbook/assets/animals_od.zip" %}
animals object detection dataset
{% endfile %}

The dataset has the following folder structure:

```
data_dir
|_images
    |_image_1.jpg
    |_image_2.jpg
    |_image_3.jpg
    |_image_4.jpg
|_boxes
    |_image_1.txt
    |_image_2.txt
    |_image_3.txt
    |_image_4.txt
    |_classes.txt
```

Now that you have the data, let's **create a Deep Lake Dataset** in the `./animals_od_deeplake`folder by running:

```python
import deeplake
from PIL import Image, ImageDraw
import numpy as np
import os

ds = deeplake.empty('./animals_od_deeplake') # Create the dataset locally
```

Next, let's specify the folder paths containing the images and annotations in the dataset. In YOLO format, images and annotations are typically matched using a common filename such as `image -> filename.jpeg` and `annotation -> filename.txt` . It's also helpful to create a list of all of the image files and the class names contained in the dataset.

```python
img_folder = './animals_od/images'
lbl_folder = './animals_od/boxes'

# List of all images
fn_imgs = os.listdir(img_folder)

# List of all class names
with open(os.path.join(lbl_folder, 'classes.txt'), 'r') as f:
    class_names = f.read().splitlines()
```

Since annotations in YOLO are typically stored in text files, it's useful to write a helper function that parses the annotation file and returns numpy arrays with the bounding box coordinates and bounding box classes.

```python
def read_yolo_boxes(fn:str):
    """
    Function reads a label.txt YOLO file and returns a numpy array of yolo_boxes 
    for the box geometry and yolo_labels for the corresponding box labels.
    """
    
    box_f = open(fn)
    lines = box_f.read()
    box_f.close()
    
    # Split each box into a separate lines
    lines_split = lines.splitlines()
    
    yolo_boxes = np.zeros((len(lines_split),4))
    yolo_labels = np.zeros(len(lines_split))
    
    # Go through each line and parse data
    for l, line in enumerate(lines_split):
        line_split = line.split()
        yolo_boxes[l,:]=np.array((float(line_split[1]), float(line_split[2]), float(line_split[3]), float(line_split[4])))
        yolo_labels[l]=int(line_split[0]) 
         
    return yolo_boxes, yolo_labels
```

Finally, let's create the tensors and iterate through all the images in the dataset in order to upload the data in Deep Lake. Boxes and their labels will be stored in separate tensors, and for a given sample, the first axis of the boxes array corresponds to the first-and-only axis of the labels array (i.e. if there are 3 boxes in an image, the labels array is 3x1 and the boxes array is 3x4).

```python
with ds:
    ds.create_tensor('images', htype='image', sample_compression = 'jpeg')
    ds.create_tensor('labels', htype='class_label', class_names = class_names)
    ds.create_tensor('boxes', htype='bbox')
    
    # Define the format of the bounding boxes
    ds.boxes.info.update(coords = {'type': 'fractional', 'mode': 'LTWH'})

    for fn_img in fn_imgs:

        img_name = os.path.splitext(fn_img)[0]
        fn_box = img_name+'.txt'
        
        # Get the arrays for the bounding boxes and their classes
        yolo_boxes, yolo_labels = read_yolo_boxes(os.path.join(lbl_folder,fn_box))
        
        # Append data to tensors
        ds.append({'images': deeplake.read(os.path.join(img_folder, fn_img)),
                   'labels': yolo_labels.astype(np.uint32),
                   'boxes': yolo_boxes.astype(np.float32)
                   })
```

{% hint style="warning" %}
In order for Activeloop Platform to correctly visualize the labels, `class_names` must be a list of strings, where the numerical labels correspond to the index of the label in the list.
{% endhint %}

### Inspect the Deep Lake Dataset&#x20;

Let's check out the third sample from this dataset, which contains two bounding boxes.

```python
# Draw bounding boxes for the fourth image

ind = 3
img = Image.fromarray(ds.images[ind ].numpy())
draw = ImageDraw.Draw(img)
(w,h) = img.size
boxes = ds.boxes[ind].numpy()

for b in range(boxes.shape[0]):
    (xc,yc) = (int(boxes[b][0]*w), int(boxes[b][1]*h))
    (x1,y1) = (int(xc-boxes[b][2]*w/2), int(yc-boxes[b][3]*h/2))
    (x2,y2) = (int(xc+boxes[b][2]*w/2), int(yc+boxes[b][3]*h/2))
    draw.rectangle([x1,y1,x2,y2], width=2)
    draw.text((x1,y1), ds.labels.info.class_names[ds.labels[ind].numpy()[b]])
```

```python
# Display the image and its bounding boxes
img
```

![](../../../../.gitbook/assets/dog\_and\_cat\_boxes.png)

Congrats! You just created a beautiful object detection dataset! 🎉

{% hint style="info" %}
**Note:** For optimal object detection model performance, it is often important for datasets to contain images with no annotations (See the 4th sample in the dataset above). Empty samples can be appended using:

`ds.boxes.append(None)`

or by specifying an empty array whose `len(shape)` is equal to that of the other samples in the tensor:

`ds.boxes.append(np.zeros(0,4))` #`len(sample.shape) == 2`
{% endhint %}
