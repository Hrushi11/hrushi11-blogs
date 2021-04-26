# Tensorflow GPU Problem:

Some of the devices that have succesfull running of TensorFlow version 2.0, still show GPU not visible.
This Juputer Notebook works on the same problem, more importantly how to solve the problem.

### Problem: 

> Jupyter Notebook not detecting GPU.

### Background:

The GPU exists, but is not detected by the Jupyter Notebook

### Predictions:

`tf.test.is_gpu_available()
 tf.test.is_built_with_cuda()`
 
☝ Run this code in your cell if it results in `False` you can continue to go ahead, but if it results in `True` you need to  look into your problem more closely.
 
### Recongnizing the devices:

`from tensorflow.python.client import device_lib
 print(device_lib.list_local_devices())`
 
☝ This code will give only stats about CPU if you are into the same problem as that of this Jupyter Notebook.

### Solution:

The TensorFlow which you have installed might be built only on CPU so you need to install the GPU based TensorFlow.

**Installation:** 

#### Try any one of the two:

**1. Run this code into your terminal (Anaconda Prompt) (Active environment) (Recommended):**  
>`conda install -c anaconda tensorflow-gpu`

**2. Using Pip install TensorFlow 2.0:**
>`pip install tensorflow-gpu==2.0`

Let the installation get finished. (This may take time as per your CPU and internet speed).

### After Installation:

Once the installation is finished try running these codes, and they should probably give the same output as shown but may be with different stats.


```python
import tensorflow as tf

# To check the current version of TensorFlow
tf.__version__
```




    '2.0.0'




```python
# To check if GPU is available and its detected 
tf.test.is_gpu_available()
tf.test.is_built_with_cuda()
```




    True



☝ This resulted in `True` hence, now our GPU is detected by the Jupyter Notebook.
Let's check the stats of our available devices on this local machine.

Run these codes for the same:


```python
# Check for CPU and GPU stats
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```

    [name: "/device:CPU:0"
    device_type: "CPU"
    memory_limit: 268435456
    locality {
    }
    incarnation: 7661865232900734680
    , name: "/device:GPU:0"
    device_type: "GPU"
    memory_limit: 1422723891
    locality {
      bus_id: 1
      links {
      }
    }
    incarnation: 17343463754814321842
    physical_device_desc: "device: 0, name: GeForce MX250, pci bus id: 0000:01:00.0, compute capability: 6.1"
    ]
    

#### If you look closely and carefully our GPU stats our visible which were not visible earlier hence, our problem is completely solved.✔

### To check how fast is your GPU over CPU:

This code is from the google colab documentation: <br>
https://colab.research.google.com/notebooks/gpu.ipynb#scrollTo=Y04m-jvKRDsJ

Try running this code block in your Jupyter cell.



```python
import tensorflow as tf
import timeit

TF_CUDNN_USE_AUTOTUNE=0
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print(
      '\n\nThis error most likely means that this notebook is not '
      'configured to use a GPU.  Change this in Notebook Settings via the '
      'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
  raise SystemError('GPU device not found')

def cpu():
  with tf.device('/cpu:0'):
    random_image_cpu = tf.random.normal((100, 100, 100, 3))
    net_cpu = tf.keras.layers.Conv2D(32, 7)(random_image_cpu)
    return tf.math.reduce_sum(net_cpu)

def gpu():
  with tf.device('/device:GPU:0'):
    random_image_gpu = tf.random.normal((100, 100, 100, 3))
    net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
    return tf.math.reduce_sum(net_gpu)
  
# We run each op once to warm up; see: https://stackoverflow.com/a/45067900
cpu()
gpu()

# Run the op several times.
print('Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images '
      '(batch x height x width x channel). Sum of ten runs.')
print('CPU (s):')
cpu_time = timeit.timeit('cpu()', number=10, setup="from __main__ import cpu")
print(cpu_time)
print('GPU (s):')
gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")
print(gpu_time)
print('GPU speedup over CPU: {}x'.format(int(cpu_time/gpu_time)))
```

    Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images (batch x height x width x channel). Sum of ten runs.
    CPU (s):
    1.372167300000001
    GPU (s):
    0.3259647000000001
    GPU speedup over CPU: 4x
    
