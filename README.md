# NGCF(Neural Graph Collaborative Filtering)-Pytorch



## 1. NGCF_Pytorch

NGCF (Neural Graph Collaborative Filtering) is the recommendation framework proposed in the research paper in SIGIR ‘19. This method leverages high-order connectivity in the user-item graph by explicitly injecting  the collaborative signal into the embedding process. The authors conducted extensive experiments on public benchmarks, demonstrating significant improvements over various SOTA models.

This project is the Pytorch implementation of this paper

> *Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, and Tat-Seng Chua (2019). Neural Graph Collaborative Filtering. SIGIR 2019*

This implementation reproduces the accuracy results in the paper. This project implements not only Pytorch but also Horovod version to accelerate the training process of NGCF effectively by supporting multi-GPU training.



## 2. Environment

I ran this code on ubuntu 20.04 using python with CUDA 11.7. 

- Python == 3.8.6
- Pytorch == 1.13.1
- Horovod ==  0.26.0



## 3. How to run

- Pytorch Single-GPU implementation

  ```shell
  python main.py -data_path=[path] -dataset=[dataset]
  ```

- Horovod Multi-GPU implementation

  ```shell
  horovodrun -np [# of GPU] python main_horovod.py -data_path=[path] -dataset=[dataset]
  ```



## 4. Result

To train and evaluate NGCF_Pytorch implementation, I used the Gowalla dataset from [here](https://github.com/xiangwang1223/neural_graph_collaborative_filtering) with four NVIDIA A100 40GB GPUs which connected by NVLink.

|      | Sec per epoch (1 GPU) | Sec per epoch (4 GPU) | Recall@20 | NGCF@20 |
| :--- | --------------------- | --------------------- | --------- | ------- |
| NGCF | 49.44                 | 19.56                 | 0.1562    | 0.1310  |

