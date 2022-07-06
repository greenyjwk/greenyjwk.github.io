---
layout: single
title:  "Pytorch Note"
categories: [Dev]
---





### <br>set_grad_enabled()

Function that sets gradient calculation on or off.

```python
# torch.set_grad_enabled(False)

x = torch.tensor([1.], requres_grad = True)
is_train = False
with torch.set_grad_enabled(is_train)
	y = x * 2
y.requires_grad
# False

torch.set_grad_enabled(True)
y = x * 2
y.requires_grad
# True

torch.set_grad_enabled(False)
y = x * 2
y.requires_grad
# False
```



### <br>torch.contiguous()

Returns a contiguous in memory tensor containing the same data as `self` tensor. If `self` tensor is already in the specified memory format, this function returns the `self` tensor.

### <br>torch.mm

Performs a matrix multiplication of the matrices inputs.

```python
mat1 = torch.randn(2, 3)
mat2 = torch.randn(3, 3)
torch.mm(mat1, mat2)
# tensor([[ 0.4851,  0.5037, -0.3633],
#        [-0.0760, -3.6705,  2.4784]])
```



### <br>t()

Expects that the input to be <=2-D tensor and transposes dimensions 0 and 1.

```python
x = torch.randn(())
x
# tensor(0.1995)
x_t = x.t()
# tensor(0.1995)

x2 = torch.randn(3)
x2 
# tensor([ 2.4320, -0.4608,  0.7702])
torch.t(x)
# tensor([ 2.4320, -0.4608,  0.7702])

x3 = torch.randn(2,3)
x3
# tensor([[ 0.4875,  0.9158, -0.5872],
#         [ 0.3938, -0.6929,  0.6932]])

x3 = torch.t(x3)
x3
# tensor([[ 0.4875,  0.3938],
#        [ 0.9158, -0.6929],
#        [-0.5872,  0.6932]])
```



### <br>torch.exp()

Returns a new tensor with the exponential of the elements of the input tensor.

```python
tensor1 = torch.exp(torch.tensor([0, math.log(2.)]))
output = torch.exp(torch.tensor(1))
# tensor(2.7183)
```



### <br>torch.matmul

```python
tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(4, 5)
torch.matmul(tensor1, tensor2).size()
# torch.Size([10, 3, 5])
```





### <br>tqdm

```python
number_list = list(range(100))
for x in tqdm(number_list):
  sleep(0.05)
print("Completed!")
```

```python
def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
```





### <br>torch.view

Returns a new tensor with the same data as the self tensor but of a different shape

```python
x = torch.randn(4, 4)
x.size()
# torch.Size([4, 4])

z = x.view(-1, 8)
z.size()
# torch.Size([2, 8])
```



### <br>torch.argmax

Returns the indices of the maximum value of all elements in the input tensor

```python
a = torch.randn(4, 4)
a
# tensor([[ 1.3398,  0.2663, -0.2686,  0.2450],
#        [-0.7401, -0.8805, -0.3402, -1.1936],
#        [ 0.4907, -1.3948, -1.0691, -0.3132],
#        [-1.6092,  0.5419, -0.2993,  0.3195]])
torch.argmax(a, dim=1)
tensor([ 0,  2,  0,  1])
```



### <br>torch.nn.CrossEntropyLoss

It returns the average of the difference between label and predicted values in each class.

```python
# internal implementdation of the torch.nn.CrossEntropyLoss()

import torch
import torch.nn as nn
import numpy as np
output = [0.8982, 0.805, 0.6393, 0.9983, 0.5731, 0.0469, 0.556, 0.1476, 0.8404, 0.5544]
target = [1]
loss1 = np.log(sum(np.exp(output))) - output[target[0]]
output = [0.9457, 0.0195, 0.9846, 0.3231, 0.1605, 0.3143, 0.9508, 0.2762, 0.7276, 0.4332]
target = [5]
loss2 = np.log(sum(np.exp(output))) - output[target[0]]
print((loss1 + loss2)/2) # 2.351937720511233
```



### <br>Dataloader

It returns the tuple and also it returns torch tensor not numpy array.



### <br>Pytorch Tensor vs Numpy ndarray

Pytorch tensor can be operated on CUDA-capable NVIDA GPU, which requires heavy matrix computation.

### <br>TF.to_tensor

Convert a PIL image or numpy.ndarray to tensor. This function does not support torchscript.



### <br>model.train()

Changes to training mode of the model



### <br>tensorboard



### <br>cuda out of memory

It happens when GPU is out of memory.

### <br>IOError: decoder libtiff not available

Once I deleted splitter tif files(by online tif files splitter), it worked out.

### <br>RuntimeError: expected scalar type Double but found Float

Float is also counted as Scalar type double. 

https://discuss.pytorch.org/t/runtimeerror-expected-object-of-scalar-type-double-but-got-scalar-type-float-for-argument-2-weight/38961/14



### <br>torch.float, torch.double

torch.float32( aka torch.float)

torch.float64( aka torch.double)

The torch tensor default type is torch.float32( aka torch.float). The model's parameters are also torch.float32 type by default.

However, the default type of Numpy ndarray is Numpy.float64. So when I load data using numpy(float64) then convert it to torch tensor, then the data would be torch.float64 which is aka torch.double. For this reason, the code below is necessary in some cases.

```python
tensor.float() # is needed to convert data to torch.float32
```

https://stackoverflow.com/questions/60239051/pytorch-runtimeerror-expected-object-of-scalar-type-double-but-got-scalar-type

### <br>torchvision.compose

It doesn't generate multiple images. The augmentation methodology inside is applied all at once.

```python
Transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomGrayscale(p=0.5),
        transforms.CenterCrop(100),
        transforms.GaussianBlur(3),
        transforms.RandomHorizontalFlip(p=0.5)
        ])
```

### <br>loading trained model and switch to eval()

eval() turns off dropout layer, batch norm layer for evaluation.

```python
unet = UNet(in_channel=3,out_channel=2)
unet.load_state_dict(torch.load("/content/drive/MyDrive/Symmetry Nucleus Image Augmentation/U-Net/u-net_params.pt"))
unet.eval()
```

### <br>torch.detach()

- Detach the tensor from the current computational graph when we don't need to trace gradient computation.
- Need to detach tensor when we need to move the tensor from GPU to CPU.

```python
import torch

x = torch.tensor(2.0, requires_grad = True)
x_detach = x.detach()
print("Tensor with detach:", x_detach)
```

### <br>*

It is not dot product. It is element-wise multiplication.

```python
tensorA = torch.rand(2,2)
# [ 
#  [2, 3],
#	 [5, 6]
# ]
tensorB = torch.rand(2,2)
# [ 
#  [1, 2],
#	 [3, 4]
# ]

overlapped = tensorA * tensorB
print(overlapped)
# [ 
#  [2, 6],
#	 [15, 24]
# ]
```



### <br>torch.size()

returns the size of the tensor. If the dim is not specified then it returns the shape of the whole tensor.

```python
tensorA = torch.rand(4,6)
size(tensorA)
# torch.Size([4,6])
```



### torch.sigmoid()

```python
sigmoid_output = (torch.sigmoid(outputs) > 0.5).float()
sigmoid_output
# tensor([[1., 1., 1., 1.],
#        [1., 1., 1., 1.],
#        [1., 1., 1., 1.],
#        [1., 1., 1., 1.]])
```

### <br>torch.cat

Concatenates the given sequence of tensors in the given dimension

```python
original = torch.ones(4,4)
# axis is specified to determine where to tensors be concatenated
concatenated = torch.zeros(4,4)
merged_tensor = torch.cat( (original, concatenated), 1)
print(merged_tensor.shape)
# torch.Size([4,8])
```

### <br>torch.nn.functional.pad

Pad tensor

```python
tensor = torch.rand(4,4)
padded_tensor = F.pad(tensor, (2, 2, 2, 5))
print(padded_tensor.shape)
# torch.size([11,8])
```

### <br>scalar type should be long data type

long is synonymous with integer. PyTorch doesn't accept a Float tensor as categorical target, Need to cast the tensor as long datatype.

```python
# CORRECT CASE
input = torch.ones((5000, 2) , dtype = torch.float)
label = torch.ones(5000, dtype = torch.long)
criterion = nn.CrossEntropyLoss()
loss = criterion(input, label)
```

```python
# ERROR CASE
# label = torch.ones(5000, dtype = torch.float) 
# cuases error that is 'expected scalar type Long but found Float'
input = torch.ones((5000, 2) , dtype = torch.float)
label = torch.ones(5000, dtype = torch.float)
criterion = nn.CrossEntropyLoss()
loss = criterion(input, label)
```



#### <br>tensor.dtype

Access the datatype of tensor

```python
tensor = torch.rand(50,1,128,128)
tensor.dtype
# torch.float32
```



### <br>torch.from_numpy

Creates a tensor from a numpy.ndarry. The returned tensor and ndarray share the same memory. Modification to the tensor will be reflected in the ndarray and vice versa.

```python
a = np.array([1,2,3])
tensor_a = torch.from_numpy(a)
batch_train_x = torch.from_numpy(x_train[i * batch_size : (i + 1) * batch_size]).float()
```



### <br>tqdm -> trange

Trange can be used as a convenient shortcut

```python
for i in tqdm(range(1000))
```

```python
from tqdm import trange
import time

epochs = 10
t = trange(epochs, leav=True)
for i in t:
  print(i)
  time.sleep(2)
```



### <br>summary()

Provide the visualization of model

```python
from torchsummary import summary
unet = UNet(in_channel=3, out_channel=2) 
summary(unet, (3,128,128), device='cpu')
```



### <br>reshape()

Return the same data and tensor but with specified shape.

```python
output_resahped = output.reshape(batch_size*width_out*height_out, 2)
# output_resahped has a shape : torch.Size([batch_size * width_out * height_out, 2])
```

### <br>permute()

Rearrange the original tensor according to the desired order of the dimension. It returns the new tensor that has the same total number of tensors.

```python
outputs = outputs.permute(0,1,3,2)
```

### <br>

### <br>os.walk()

Python method walk() generates the file names in a directory tree by walking the tree either top-down or bottom-up.

```python
os.walk(".", topdown=False)
```



### <br>np.where()

Returns elements chosen from x or y depending on condition.

```
np.where()
```





### <br>masking

obj_ids is [2] shape array, and

```python
obj_ids[:, None, None]
# creates [1,1,2] shape array.

masks = mask == obj_ids[:, None, None]
# mask is [255,555] shape array, and masks will have [2,255,555] shape true/false array.
```



### <br>Split the color-encoded mask into a set

```python
obj_ids = [1,2]
temp = obj_ids[:, None, None]
print(temp)
# [ [ [1] ]
#   [ [2] ] ]
```



### <br>np.unique()

Find the unique elements of an array.

```python
mask = Image.open(mask_path)
mask = np.array(mask)
obj_ids = np.unique(mask)
```



### <br>natsorted()

Sorts an iterable naturally, not lexicographically. Returns a list containing a sorted copy of the iterable.

```python
imgs = list(natsorted(os.listdir(os.path.join(root, "image"))))
```



### <br>os.listdir()

The function is used to get the list of all files and directories in the specified directory.

```python
self.imgs = list(natsorted(os.listdir(os.path.join(root, "image"))))
```



### <br>os.path.join()

```python
path = "/home"
print(os.path.join(path, "User/Public/", "Documents", ""))
# /home/User/Public/Documents/
```



### <br>Summary()

Summarize the network when it comes to layer.

```python
mlp_model = SimpleMLP().cuda()
train_loss1 = []
test_accuracy = []
for epoch in tqdm(range(NUM_EPOCH)):
  train_loss1.append(fit(mlp_model, train_loader))
  test_accuracy1.append(eval(mlp_model, test_loader))
summary(mlp_model, input_size = (3,32,32) )
```



### <br>label.cpu()

Moves the parameters to CPU from GPU

```python
model.eval()
device = newxt(model.parameters()).device.index
pred_labels = []
# it should be detached(turned off required grad etc) to be a numpy
label.cpu().detach().numpy() 
```



### <br>argmax(axis = 1)

Extracts the maximum value from the arguments.

```python
pred_labels = pred_labels.argmax(axis=1)
accuracy = ((real_labels == pred_labels)/len(real_labels) ) * 100
```



### <br>model.eval()

This is such as the switch to make the model is suitable with inference. Dropout, batchNorm layers are switched to fit the testing. At the same time model.train() should be called before it switches to training.

```python
def eval(model, testdataLoader):
	model.eval()
```



### <br>torch.matmul

Matrix multiplication

- torch.matmul(tensor1, tensor2)
- torch.matmul(tensor1.view(1,-1), tensor2)[0]

```python
list1 = [1,2]
list2 = [3,4]

tensor1 = torch.tensor([1,2])
tensor2 = torch.tensor([3,4])

print(list1 + list2) # This is concatenation 

print(torch.matmul(tor1, tor2)) 								# Without view(), it also works.
print(torch.matmul(tor1.view(1,-1), tor2)[0])		# view() changes shape.
```



### <br>item()

It extracts the value in the tensor to make it scalar.

```python
temp1 = torch.sum(data)
print(temp1.item())
```



### <br>Pytorch Conditional Statement

```
torch.where(data, torch.ones())
```



### <br>Conversion between numpy.ndarray and torch.tensor

Operation between tensor in cuda and in CPU 

```python
tensor_in_cuda = torch.tensor([1,1]).cuda()
tensor_in_CPU = tensor_in_cuda.cpu()

print(tensor_in_cuda + tensor_in_CPU)	# cause error
```



<br>Batch size is added in front of the shape as well.

```python
X, y = datasets. make_circles(n_samples = npts, random_state = 123, noise = 0.2, factor = 0.3)
x_data = torch.Tensor(X)
y_data = torch.Tensor(y.reshape(500, 1))
print(x_data.shape)			# torch.Size([500,2]) -> 500 is aded because 500 is batch size.
print(y_data.shape)		# torch.Size([500,1]) -> 500 is aded because 500 is batch size.
```



### <br>reshape()

The functions returns the same number of datasets with changing the shape

```python
a = torch.tensor([1,2,3,3,4,5,6,7,8])
a_reshaped = a.reshape([8, 1])
print(a_reshaped)

# tensor([[1, 2],
#        [3, 4],
#        [5, 6],
#        [7, 8]])
# torch.size([8])
```

### <br>unsqueeze()

This is used to reshape a tensor by adding a new dimensions at given positions.

```python
a = torch.Tensor([[1,2], [3,4]])
a_squeezed = a.unsqueeze()
print(a_squeezed.shape)

a_squeezed_0 = a.unsqueeze(0)
# add dimension at 0
# torch.Size([1, 5])

a_squeezed_1 = a.unsqueeze(1)
# add dimension at 1
# torch.Size([5, 1])
```

### <br>cuda using Strategy

> Most variables are usually initialized and defined from the CPU. When they needs to be trained, they have to be relocated on the GPU. Once the training and operations are completed, then are moved back to the CPU again. This process will repeat.



### <br>retain_grad()



### <br>Considering the batch size 

An additional dimension for the batch_size should be added at the input.

```python
batch_size = 10
input = torch.randn(batch_size, 100)
output = net(input)

target = torch.randn(batch_size, 10)
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
```



### <br>gradient

Inheriting nn.Module means that all the layers have gradient =True configuration.



### <br>sub_

```python
learning_rate = 0.01
for param in net.parameters():
	param.data.sub_(param.grad.data * learning_rate)
```



```python
a = torch.tensor([[1,2]]).expand(3, -1)
b = torch.tensor([[10], [20], [30]])


a.sub(b) 
# tensor([[ -9,  -8],
#        [-19, -18],
#        [-29, -28]])
 
a.sub_(b)

# tensor([[-59, -58],
#         [-59, -58],
#        [-59, -58]])
        
```



### <br>zero_grad()

> In pytorch, during the single mini-batch, user want to explicitly set to gradient to zero before starting to do backpropagation, since the pytorch accumulates the gradients on subsequent backward passes. This accumulation is useful while training RNN or gradient of loss summation over multiple mini-batches are necessary.
>
> Because of this, when model starts to be trained, it should zero out the gradients, so that it corrects the parameters correctly. Otherwise, the gradients would be a combination of the old gradients that have already been used to update model parameters.



### <br>torch.optim

Rather than updating optimizer manually, torch.optim updates weight and bias using Adam, SGD etc.

```python
num_epochs = 100
learning_rate = 0.01
optimizer = optim.SGC(net.parameters(), lr = 0.01)
criterion = nn.MSELoss()

for epoch in range(num_epochs):
  optimizer.zero_grad()
	output = net(input)
  loss = creterion(target, output)
  print(loss)
  loss.backward() 	# Caculate the gradient from the model. 
  optimizer.step()	
  # Using the calculated gradient and learning rate, it updates the parameters from the model.
```





### <br>cuda

#### amp

Using amp in pytorch mostly mean that using **torch.cuda.amp.autocast** and **torch.cuda.amp.GradScale**.

They help the training time be lowered without affecting training performance -> Less GPU Usage, Better GPU Speed.



#### mixed precision

Using both datatype float16 and float32 to lower neural network's runtime and memory usages.

```python
import torch

scaler = torch.cuda.amp.autucast(enabled = True):
  outputs = model(inputs, targets)

loss = outputs["total_loss"]

opitimizer.zero_grads()
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```



------



![screenshot](../../../images/2022-05-09-tqdm/screenshot.png)



--------------

