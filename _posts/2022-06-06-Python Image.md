---
layout: single
title:  "Python Image"
categories: [skimage]
---

### skimage

io.imread returns numpy ndarray

```python
from skimage import io, transform, color

path = drive_path + "/dummy_imageout.tif"
img = io.imread(path)[0:1]
print(type(img))
```

