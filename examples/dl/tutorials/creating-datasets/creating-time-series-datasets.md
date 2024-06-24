---
description: >-
  Deep Lake is a powerful tool for easily storing and sharing time-series
  datasets with your team.
---

# Creating Time-Series Datasets

## How to use Deep Lake to store time-series data

#### This tutorial is also available as a [Colab Notebook](https://colab.research.google.com/drive/1MX2PjzfPRXaxymWtGP81-81pm8E61sZQ?usp=sharing)

Deep Lake is intuitive format for storing large time-series datasets and it offers compression for reducing storage costs. This tutorial demonstrates how to convert a time-series data to Deep Lake format and load the data for plotting.&#x20;

### Create the Deep Lake Dataset

The first step is to download the small dataset below called _sensor data._

{% file src="../../../../.gitbook/assets/sensor_data.zip" %}

This is a subset of a [dataset available on kaggle](https://www.kaggle.com/malekzadeh/motionsense-dataset), and it contains the iPhone x,y,z acceleration for 24 users (subjects) under conditions of walking and jogging. The dataset has the folder structure below. `subjects_info.csv` contains metadata such as `height`, `weight`, etc. for each subject, and the `sub_n.csv` files contains the time-series acceleration data for the `nth` subject.

```python
data_dir
|_subjects_into.csv
|_motion_data
    |_walk
        |_sub_1.csv
        |_sub_2.csv
        ...
        ...
    |_jog
        |_sub_1.csv
        |_sub_2.csv
        ...
        ...
```

Now that you have the data, let's **create a Deep Lake Dataset** in the `./sensor_data_deeplake` folder by running:&#x20;

```python
import deeplake
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

ds = deeplake.empty('./sensor_data_deeplake') # Create the dataset locally
```

Next, let's specify the folder path containing the existing dataset, load the subjects metadata to a Pandas DataFrame, and create a list of all of the time-series files that should be converted to Deep Lake format.

```python
dataset_path= './sensor_data'

subjects_info = pd.read_csv(os.path.join(dataset_path, 'subjects_info.csv'))

fns_series = []
for dirpath, dirnames, filenames in os.walk(os.path.join(dataset_path, 'motion_data')):
    for filename in filenames:
        fns_series .append(os.path.join(dirpath, filename))
```

Next, let's create the `tensors` and add relevant metadata, such as the dataset source, the tensor units, and other information. We leverage `groups` to separate out the primary acceleration data from other user data such as the weight and height of the subjects.&#x20;

```python
with ds:
    #Update dataset metadata
    ds.info.update(source = 'https://www.kaggle.com/malekzadeh/motionsense-dataset', 
                   notes = 'This is a small subset of the data in the source link')

    #Create tensors. Setting chunk_compression is optional and it defaults to None
    ds.create_tensor('acceleration_x', chunk_compression = 'lz4') 
    ds.create_tensor('acceleration_y', chunk_compression = 'lz4')
    
    # Save the sampling rate as tensor metadata. Alternatively,
    # you could also create a 'time' tensor.
    ds.acceleration_x.info.update(sampling_rate_s = 0.1)
    ds.acceleration_y.info.update(sampling_rate_s = 0.1)
    
    # Encode activity as text
    ds.create_tensor('activity', htype = 'text')
    
    # Encode 'activity' as numeric labels and convert to text via class_names
    # ds.create_tensor('activity', htype = 'class_label', class_names = ['xyz'])
    
    ds.create_group('subjects_info')
    ds.subjects_info.create_tensor('age')
    ds.subjects_info.create_tensor('weight')
    ds.subjects_info.create_tensor('height')
    
    # Save the units of weight as tensor metadata
    ds.subjects_info.weight.info.update(units = 'kg')
    ds.subjects_info.height.info.update(units = 'cm')
```

Finally, let's iterate through all the time-series data and upload it to the Deep Lake dataset.  &#x20;

```python
with ds:
    # Iterate through the time series and append data
    for fn in tqdm(fns_series):
        
        # Read the data in the time series
        df_data = pd.read_csv(fn)
        
        # Parse the 'activity' from the file name
        activity = os.path.basename(os.path.dirname(fn))
        
        # Parse the subject code from the filename  and pull the subject info from 'subjects_info'
        subject_code = int(os.path.splitext(os.path.basename(fn))[0].split('_')[1])
        subject_info = subjects_info[subjects_info['code']==subject_code]
        
        # Append data to tensors
        ds.activity.append(activity)
        ds.subjects_info.age.append(subject_info['age'].values)
        ds.subjects_info.weight.append(subject_info['weight'].values)
        ds.subjects_info.height.append(subject_info['height'].values)
                
        ds.acceleration_x.append(df_data['userAcceleration.x'].values)
        ds.acceleration_y.append(df_data['userAcceleration.y'].values)
```

### Inspect the Deep Lake Dataset

Let's check out the first sample from this dataset and plot the acceleration time-series.

{% hint style="success" %}
**It is noteworthy that the Deep Lake dataset takes 36% less memory than the original dataset due to `lz4` chunk compression for the acceleration tensors.**
{% endhint %}

```python
s_ind = 0 # Plot the first time series
t_ind = 100 # Plot the first 100 indices in the time series

#Plot the x acceleration
x_data = ds.acceleration_x[s_ind].numpy()[:t_ind]
sampling_rate_x = ds.acceleration_x.info.sampling_rate_s

plt.plot(np.arange(0, x_data.size)*sampling_rate_x, x_data, label='acceleration_x')

#Plot the y acceleration
y_data = ds.acceleration_y[s_ind].numpy()[:t_ind]
sampling_rate_y = ds.acceleration_y.info.sampling_rate_s

plt.plot(np.arange(0, y_data.size)*sampling_rate_y, y_data, label='acceleration_y')

plt.legend()
plt.xlabel('time [s]', fontweight = 'bold')
plt.ylabel('acceleration [g]', fontweight = 'bold')
plt.title('Weight: {} {}, Height: {} {}'.format(ds.subjects_info.weight[s_ind].numpy()[0],
                                               ds.subjects_info.weight.info.units,
                                               ds.subjects_info.height[s_ind].numpy()[0],
                                               ds.subjects_info.height.info.units),
         fontweight = 'bold')

plt.xlim([0, 10])
plt.grid()
plt.gcf().set_size_inches(8, 5)
plt.show()
```

![](../../../../.gitbook/assets/time\_series\_plot.png)

Congrats! You just converted a time-series dataset to Deep Lake format! 🎉
