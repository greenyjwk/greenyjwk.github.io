---
layout: single
title:  "Mask R-CNN"
categories: [Computer Vision]

---

## Main Idea

Faster R-CNN replaces selective search with Regional Proposal Network(RPN) which is also neural network, so that it implements the end-to-end training process.

Other than RPN and removal of selective search that was used in the Fast-R CNN it is same as Fast R-CNN.



#### RPN

Determining whether the anchor has background or object, 

GT Label  is IoU of ground truth and anchor.

