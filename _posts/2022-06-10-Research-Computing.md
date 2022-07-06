---
layout: single
title:  "Research Computing"
categories: [GPU]
---

### nvidia-smi

It monitors the NVIDIA GPU devices.

```shell
nvidia-smi
```



### <br>Data Parallelism

Data is sometimes too large to be trained in a single GPU, for this reason, for example, each batch data are distributed in the different GPUs. And every forward and backward propagation is completed, the GPU shares the parameters to get the average of it, and shared the updated parameters with all GPUs. This process is called synchronization 

