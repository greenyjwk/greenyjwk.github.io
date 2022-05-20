---
layout: single
title:  "Today I learned"
---

### Difference between var and std



### <br>What is cuda



### <br>gradient and gradient from torch.requires

When model is being trained, it is being trained to reduce the error. The direction of this is called 'gradient'.



### <br>What is super(Net, self).__init__



### <br>How to inherit nn.Module



### <br>What do we mean by parameters in deep learning?

Weight and Bias are called as parameter in machine learning.

```python
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(in_features = 100, out_features = 100, bias = True)
    self.fc1_act = nn.ReLU()
    self.fc2 = nn.Linear(in_features = 100, out_features = 10, bias = True)

	def forward(self, x):
    out = self.fc1(x)
    out = self.fc1_act(out)
    out = self.fc2(out)  
    return out
  
params = list(net.parameters)
print(params)

print(len(params))		
# It turns out that 4, since the each layer(2 layers in our code above) has weight and bias.

print(params[o].size())
# It prints out 100 x 100 in weight.
# the first layer's weight.

print(params[1].size())
# It prints out 100 for bias.
# the first layer's bias.
```

#### Named_parameters()

```python
for name, param in net.named_parameters():
  print(name)
  print(param)
  
print(net.fc1.weight.data)
```



```python
learning_Rate = 0.01
for param in net.parameters():
  param.data.sub_(param.grad.data * learning_rate)
  
learning_rate = 0.01
for param in net.parameters(): 
```