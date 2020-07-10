# Reflection Backdoor: A Natural Backdoor Attack on Deep Neural Networks


![Python 3.6](https://img.shields.io/badge/python-3.6-DodgerBlue.svg?style=plastic)
![Pytorch 1.10](https://img.shields.io/badge/pytorch-1.2.0-DodgerBlue.svg?style=plastic)
![CUDA 10.0](https://img.shields.io/badge/cuda-10.0-DodgerBlue.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-CC_BY--NC-DodgerBlue.svg?style=plastic)

 Our paper is accepted by **ECCV 2020**. 

We investigate the use of a natural phenomenon, i.e., reflection, as the backdoor pattern, and propose the reflection backdoor (*Refool*) attack to install stealthy and effective backdoor into DNN models.

<div align=center>  <img src="figures/teaser.png" alt="Teaser" width="500" align="bottom" /> </div>

**Picture:**  *Our reflection backdoors (rightmost column) are crafted based on the natural reflection phenomenon, thus need not to mislabel the poisoned samples on purpose (A - D, mislabels are in red texts), nor rely on obvious patterns (A - C, E), unpleasant blending (D), or suspicious stripes (F). Therefore, our reflection backdoor attacks are
stealthier.*



<div align=center>  <img src="./figures/pipeline.png" alt="Main image" width="800" align="center" /> </div>
**Picture:**  *The pipeline of proposed Refool.*
<br>


<div align=center>  <img src="./figures/optical_model.png" alt="MPI Results" width="800" align="center" /> </div>
**Picture:**  *The physical (left) and mathematical (right) models for three types of reflections.*


<div align=center>  <img src="./figures/vis_cam.png" alt="MPI Results" width="800" align="center" /> </div>
**Picture:**  *. Understandings of Refool with Grad-CAM [43] with two samples from PubFig(left) and GTSRB(right). In each group, the images at the top are the original input, CL [53], SIG [3] and our Refool (left to right), while images at the bottom are their corresponding attention maps.*



This repository contains the official PyTorch implementation of the following paper:

> **Reflection Backdoor: A Natural Backdoor Attack on Deep Neural Networks**<br>
>  Yunfei Liu, Xingjun Ma, James Bailey, and Feng Lu<br> https://arxiv.org/abs/2007.02343
> 
>**Abstract:**   Recent studies have shown that DNNs can be compromised by backdoor attacks crafted at training time. A backdoor attack installs a backdoor into the victim model by injecting a backdoor pattern into a small proportion of the training data. At test time, the victim model behaves normally on clean test data, yet consistently predicts a specific (likely incorrect) target class whenever the backdoor pattern is present in a test example. While existing backdoor attacks are effective, they are not stealthy. The modifications made on training data or labels are often suspicious and can be easily detected by simple data filtering or human inspection. In this paper, we present a new type of backdoor attack inspired by an important natural phenomenon: reflection. Using mathematical modeling of physical reflection models, we propose reflection backdoor (Refool) to plant reflections as backdoor into a victim model. We demonstrate on 3 computer vision tasks and 5 datasets that, Refool can attack state-of-the-art DNNs with high success rate, and is resistant to state-of-the-art backdoor defenses.

## Resources

Material related to our paper is available via the following links:

- Paper:  https://arxiv.org/abs/2007.02343
- Project: Coming soon!
- Code: https://github.com/DreamtaleCore/Refool

## System requirements

* Only Linux is tested, Windows is under test.
* 64-bit Python 3.6 installation. 
* PyTorch 1.2.0 or newer with GPU support.
* One or more high-end NVIDIA GPUs with at least 8GB of DRAM.
* NVIDIA driver 391.35 or newer, CUDA toolkit 9.0 or newer, cuDNN 7.3.1 or newer.

## Playing with pre-trained networks and training

Comming soon!

## Citation

If you find this work or code is helpful in your research, please cite:

```latex
@inproceedings{Liu2020Refool,
	title={Reflection Backdoor: A Natural Backdoor Attack on Deep Neural Networks},
	author={Yunfei Liu, Xingjun Ma, James Bailey, and Feng Lu},
	booktitle={ECCV},
	year={2020}
}
```

## Contact

If you have any questions, feel free to E-mail me via: `lyunfei(at)buaa.edu.cn`