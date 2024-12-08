# GVKSense
source code for paper "Visual Knowledge-Aided Learning Framework for Cross-Task WiFi-Based Sensing"


## Introduction 

WiFi-based sensing technology can utilize Channel
State Information (CSI) to accomplish tasks such as activity detection, human identification, and gesture recognition. However,
when faced with a new WiFi-based sensing task, most of the
existing deep learning methods require training from scratch,
resulting in poor performance and more training epochs. In this
paper, we propose a general cross-task learning framework that
can address the aforementioned problem and be widely applied
in WiFi-based sensing. Deep models in the proposed framework
will be fine-tuned from a new perspective by leveraging prior
visual knowledge to enhance their performance. First, transforming functions are proposed to convert the multiform CSI
data into uniform CSI images. Then, we efficiently adapt a
visual Transformer-based model, which is pretrained on largescale visual datasets, to realize task-specific WiFi-based sensing
applications. The adaptation method employed differs from conventional fine-tuning techniques as it focuses on modifying only
a small part of the model

![image](https://github.com/user-attachments/assets/a6770f73-46b2-440b-a171-72ed3dd7f5fc)

## Proposed Architecture

![image](https://github.com/user-attachments/assets/d62ee988-2c38-4112-8f50-d71774f908e2)

The proposed cross-task framework. The first step is to transform the
CSI data into the CSI image. The second step is to finetune the ViT model
efficiently with an ”adapter”.


## Simulation Results


![image](https://github.com/user-attachments/assets/715abca6-af37-4d02-b18f-f13d04aa8bec)


