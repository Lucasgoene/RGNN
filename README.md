# RGNN - Regularized Graph Neural Network
### Bartolomeo Caruso, Gabriele Costanzo, Giuseppe Fallica

---

This repository is an extention of the Python implementation of RGNN.

It's the result of a final assignment for the Cognitive Computing 2020/21 course at **UniCT - University of Catania**.

We refactored some code from other contributors, such as [PieraRiccio](https://github.com/PieraRiccio/RGNN)'s implementation of the **EmotionDL** regularizer, and added all the missing pieces necessary for a minimal execution of the model on the [SEED-IV](https://bcmi.sjtu.edu.cn/~seed/seed-iv.html) EEG dataset.

We provide the code both as a **Python module** and as a **Colab/Jupyter notebook**, and we provide additional files to be integrated with the SEED-IV dataset.

## Note:
In order to execute the code correctly, it may be necessary installing **torch_geometric** and all of its dependencies. Here is the code we used to import it in **Google Colab**:
```
TORCH_version = torch.__version__
TORCH = TORCH_version.split('+')[0]

CUDA_version = torch.version.cuda
CUDA =  'cu' + CUDA_version.replace('.', '')

!pip install torch-scatter     -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
!pip install torch-sparse      -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
!pip install torch-cluster     -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
!pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
!pip install torch-geometric 
```

Following, is the original README from [zhongpeixiang](https://github.com/zhongpeixiang/RGNN)'s repo:

---

This repo illustrates the RGNN model implementation in the paper *EEG-Based Emotion Recognition Using Regularized Graph Neural Networks*. The model is based on [torch geometric](https://github.com/rusty1s/pytorch_geometric) v1.2.1

The EmotionDL regularizer is easy to implement and thus not included in the repo. More details can be found in the [paper](https://arxiv.org/abs/1907.07835).


If you find the paper or this repo useful, please cite
```
@article{zhong2020eeg,
  title={EEG-Based Emotion Recognition Using Regularized Graph Neural Networks},
  author={Zhong, Peixiang and Wang, Di and Miao, Chunyan},
  journal={IEEE Transactions on Affective Computing},
  year={2020},
  publisher={IEEE}
}
```