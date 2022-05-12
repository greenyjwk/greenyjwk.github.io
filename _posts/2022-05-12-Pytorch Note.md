---
layout: single
title:  "Pytorch Note"
categories: pytorch
---

### torch.matmul

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





<br><br><br>



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

