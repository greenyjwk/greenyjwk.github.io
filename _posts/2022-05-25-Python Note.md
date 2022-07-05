---
layout: single
title:  "Python Note"
categories: [Development]
---

### <br> [ERROR] logp = torch.gather(logp, 1, target.view(n, 1, H, W))



### <br>Adding subarry to   sing np.append()

```python
self.images = np.array([])
self.images = io.imread(os.path.join(folder, filename), plugin='pil')
#(512, 512)

self.images = np.expand_dims(self.images, axis=0)
#(1, 512, 512)

self.images = np.append(self.images, [io.imread(os.path.join(folder, filename), plugin='pil')] , 0)
#(n, 512, 512)
```



### <br>count_nonzero()

It returns the count of non zero values in given Numpy array.

```python
arr = np.array([True, False, True, True])
true_count = np.count_nonzero(arr)
print(true_count)
# 3
```



### <br>Access numpy array via slicing and direct access

Direct access and slicing array have different output shape.

```python
# x_train.shape = [73,3,128,128]
print(x_train[0].shape)
# (3, 128, 128)

print(x_train[0:1].shape)
# (1, 3, 128, 128)
```

### <br>np.squeeze()

Remove axes of length one from a

```python
# If axis is not specified, then it squeezes the axis that has one size axis
# one size axis here means that the outmost reduandant layer such as [[1, 2]]
a = np.array([[[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]]])
a.shape
# (1, 3, 3)

a_squeezed = np.squeeze(a) 
a_squeezed.shape
# (3, 3)
```





### <br>ndarray.astype()

Copy the array and cast the specified type

```python
batch_val_y = torch.from_numpy(y_val[ i * batch_size : (i + 1) * batch_size ]).astype(int).long()
```

```python
ndarray = np.array([ 1.0, 2.0, 3.8 ])
int_ndarray = ndarray.astype(int)
int_ndarray
# array([1,2,3])
```



### np.maximum()

```python
arr1 = [2, 8, 125]
arr2 = [3, 3, 15]
out_arr = np.maximum(in_arr1, in_arr2) 
# [3, 8, 125]
```

```python
arr1 = [np.nan, 0, np.nan]
arr2 = [np.nan, np.nan, 0]
out_arr = np.maximum(in_arr1, in_arr2)
# [nan, nan, nan]
```

```python
bool_arr1 = np.array([1, 0.5, 0, None, 'a', '', True, False], dtype=bool)
bool_arr2 = np.array([False, False, False, False, False, False, False, False], dtype=bool)
print(bool_arr1)
print(bool_arr2)
# [True  True  False False True  False True  False]
# [False False False False False False False False]

mask = np.maximum(bool_arr1, bool_arr2)
print(mask)
# [True True False False True False True False]
```



### <br>np.expand_dims()

axis = 0 then add dimension to the outmost layer<br>axis = -1 then add dimension to the innermost layer

```python
# x1 = np.array([1, 2])
# x2 = np.array([1, 2])
# x.shape
# (2,)

y2 = np.expand_dims(x2, axis=0)
y2
# array([[1, 2]])
y.shape
# (1, 2)

y1 = np.expand_dims(x1, axis=1)
y1
# array([[1],
#       [2]])
# (2, 1)


y.shape
# (2, 1)
mask_ = np.expand_dims(resized_dummy_img, axis=-1)
```



### <br>skimage.resize()

#### <br>Aliasing artifacts

Aliasing artifacts occur in the phase encoding direction when dimensions of the imaged object exceeds the field of view. In MRI, artifact means that pixels that fully depict anatomic structure.

<br>**mode** {‘constant’, ‘edge’, ‘symmetric’, ‘reflect’, ‘wrap’}, optional
Points outside the boundaries of the input are filled according to the given mode.

<br>**preserve_range** bool, optional

Whether to keep the original range of values. Otherwise, the input image is converted according to the conventions of *img_as_float*.

<br>**anti_aliasing** bool, optional

Whether to apply a Gaussian filter to smooth the image prior to downsampling. It is crucial to filter when downsampling the image to avoid aliasing artifacts. If not specified, it is set to True when downsampling an image whose data type is not bool.

```python
image_resized = resize(img, (image.shape[0]//4, image.shape[1]//4), anti_aliasing=True)
# Points outside the boundaries of the input are filled according to the given mode. Modes match the behaviour of numpy.pad

# preserve_range: bool, optional
# Whether to keep the original range of values. Otherwise, the input image is converted according to the conventions of img_as_float.
```



### <br>tqdm()

Argument 'total' is predictive stats.

```python
pbar = tqdm(total=100)
for i in range(10):
    sleep(0.1)
    pbar.update(10)
pbar.close()

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
  print(n, id_)
```



### <br>sys.std.flush()

It forces not buffering.

```
```



### <br>os.walk()

Python method walk() generates the file names in a directory tree by walking the tree either top-down or bottom-up.

```python
for (root, dirs, files) in os.walk('Test', topdown=true):
	print (root)
	print (dirs)
	print (files)
	print ('--------------------------------')

os.walk(".", topdown=False)
```



### <br>next()

It returns the next time in an iterator.

```python
test_ids = next(os.walk(TEST_PATH))[1]
```
