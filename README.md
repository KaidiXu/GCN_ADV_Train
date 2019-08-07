Optimization based GNN attack and defense 
-----------------------

In this work, we first propose a novel gradient-based graph neural networks (GNNs) attack method that facilitates the difficulty of tackling discrete graph data.
When comparing to current adversarial attacks on GNNs, the results show that by only perturbing a small number of edge perturbations (including addition and deletion), our optimization-based attack
can lead to a noticeable decrease in classification performance. Moreover, leveraging our gradientbased attack, we propose the first optimizationbased adversarial training for GNNs.

Cite this work:

Kaidi Xu\*, Hongge Chen\*, Sijia Liu, Pin-Yu Chen, Tsui-Wei Weng, Mingyi Hong and Xue Lin, ["Topology Attack and Defense for Graph Neural Networks:
An Optimization Perspective"](https://arxiv.org/abs/1906.04214), IJCAI 2019. (\* Equal Contribution)

```
@inproceedings{xu2019topology,
  title={Topology Attack and Defense for Graph Neural Networks: An Optimization Perspective},
  author={Xu, Kaidi and Chen, Hongge and Liu, Sijia and Chen, Pin-Yu and Weng, Tsui-Wei and Hong, Mingyi and Lin, Xue},
  booktitle = "International Joint Conference on Artificial Intelligence (IJCAI)",
  year={2019}
}
```

Prerequisites
-----------------------

The code is tested with python3.6 and TensorFlow v1.13. Please use miniConda to manage your Python environments.
The following Conda packages are required:

```
conda install python==3.6
conda install numpy scipy tensorflow-gpu 
grep 'AMD' /proc/cpuinfo >/dev/null && conda install nomkl
```

After installing prerequisites, clone this repository:

```
git clone https://github.com/KaidiXu/GCN_ADV_Train.git
cd GCN_ADV_Train
```

Train a Natural Model
-----------------------

To train a natural Cora model, simply run:
```
python train.py
```

This will train a natural GCN model on Cora dataset and save it at ```nat_cora``` directory.

Train a Robust Model
-----------------------

To train a robust Cora model, you first need to train a natural model to get the predicted labels. In ```train.py```, the predicted labels will be saved when the natural model training is done. Then simply run

```
python adv_train_pgd.py
```
This will train a robust GCN model using the method proposed in our paper  on Cora dataset and save it at ```rob_cora``` directory.

Attack a Model
-----------------------

To attack the model we just trained, run

```
python attack.py --model_dir=nat_cora
```

You can also change the value of ```--model_dir``` to attack models in other directories. You may use ```--method``` to choose the attack method. Default method is ```PGD```, we also have Carlini \& Wagner style attack, which is ```---method=CW```.



