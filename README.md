
# PoC of ACT imitation learning
- The aim of this repository is for reproduction and PoC of [action chunking transformer](https://github.com/tonyzhaozh/act)

# Example
- We train the policy that pick the small stapler box and place that in the other black box.
- The position of the box can be **random but within the range of robot motion**.
- The following video shows the ACT policy employed to controlling koch-v1-1.
- The box is **placed at the random position in each pick-and-place trials**.
- It takes about **30 minutes** to complete the 500 epochs training of this policy.

|||
|:-:|:-:|
|<video src="https://github.com/user-attachments/assets/a037ce06-af63-4127-963a-cf475e38a0a7"/>|<video src="https://github.com/user-attachments/assets/e8bfec96-4b04-4040-877a-f940c0df0512"/>|


- The ACT policy in this video are trained in the following conditions

  |||
  |----|----|
  |GPU|Intel Core i7| 
  |GPU|NVIDIA GeForce RTX 2070|
  |Size of memory|32 GB|
  |Number of epochs | 500 |
  |Training sample size|60|
  |Batch size|8|
  |Number of VAE encoder layers|4|
  |Number of encoder layers|4|
  |Number of decoder layers|5|
  |Backbone|ResNet18|
  |Number of hidenn dimension|512|
  |Number of feedforward MLP dimension|3027|
  |Episode length|1000|
  |Image size|640 × 480|
  |Number of camera|2|

# About this repository
- If you only want to know usage or installation, please read [Usage](#usage) or [Installation](#installation).
- In this section, we describe details of the implementations of this repository.
##  What contents contained
- Re-implementation of [action chunking transformer (ACT)](https://github.com/tonyzhaozh/act) which is originally developed by Tony Z. Zhao.
- The python scripts for model training, teleoperation, and model evaluation for real the robot arm.
- Robot client class for low cost robot arm [koch-v1-1](https://github.com/jess-moss/koch-v1-1).
- Dynamixel client for DIY robot arms which is compatible with XL430-W250 and XL330-M288-T. You can use this client for any low cost robot arms which is composed of these two types of dynamixel motors.

## Re-implementation of ACT
- We re-implemented the original [action chunking transformer](https://github.com/tonyzhaozh/act) from scratch.
- The original ACT are using some utils and transformer architectures which is partialy dapted from [detr](https://github.com/facebookresearch/detr).
- We use pytorch official implementation of transformer modules.
- We don't use the positional encoding of original ACT. Alternately, simple 1-dimensional sinusoidal positional encoding is employed. This is same positional encoding as the one orignally propose in ["Attention is all you need"]([https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html]).

## About python scripts
- [train.py](/train.py)
  - This script is responsible for training ACT. 
  - By default, [train_dataset](/train_dataset/) and [test_dataset](/test_dataset/) are expected to include all train and test dataset.
- [teleoperation_and_data_collection.py](/teleoperation_and_data_collection.py)
  - This script is responsible for teloperation.
  - The follower arm are controlled to synchronize with the joint angles of the leader arm.
  - So, by directly controlling the leader arm, you can see the follower arm follow the human demonstrations.
- [evaluate_policy.py](/evaluate_policy.py)
  - This script is responsible for evaluation of trained policy.
  - By giving specific checkpoint path, you can deploy the policy you trained to the real robot.

## Robot client for koch-v1-1
- We prepared robot client base which is abstract class for the client controlling robot arms.
- Because this client prepares some basic methods that is common for any robot arms, you can customize this class even for the robot arms which is composed of actuators other than DYNAMIXEL.
- However, this repository contains only the specific implementaion koch-v1-1.
- If want know more details, please refer to [robot_client.py](koch11/core/robot_client.py). 
- And, we prepare a few examples of usage of the dynamixel robot client class. Please refer to [examples/robot_client](examples/robot_client/)

## Dynamixel client for DIY robot arms

# Installation
- Make sure you have already installed [poetry](https://github.com/python-poetry/poetry) for package manager.
- Run the followeing commands to install all dependencies in the root of the repository:
```
poetry install
```

# Usage
## How we get the robot arms ?
- This repository assumes that you already have follower and leader arms of [koch-v1-1](https://github.com/jess-moss/koch-v1-1). Please refer to [koch-v1-1](https://github.com/jess-moss/koch-v1-1) to prepare your own robot arms.

## Collection of the data by teleoperation
- To collect the data by teloperation, run
```
python teleoperation_and_data_collection.py --dataset_dir ./train_dataset --initial_episode_id 0
```

## Push collected datasets to the huggingface dataset repository 
1. - Login to your huggingface account first by runnning
```
huggingface-cli login
```
2. - Run the following command to push `datasets` directory into the huggingface dataset repository:
```
huggingface-cli upload <your_repository_id> <local_path_to_your_dataset>  --repo-type dataset
```

3. - To clone the pushed datasets from huggingface dataset repository, run the following command:
```
huggingface-cli download <your_repository_id> --local-dir <local_path_to_your_dataset> --repo-type dataset
```

## ACT training
- To train ACT from scratch, run
```
python train.py --num_epochs 10000 --train_dataset_dir ./train_dataset --test_dataset_dir ./test_dataset --num_episodes_train 30
```

## Evaluate your policy
- To evaluate trained policym run
```
python evaluate_policy.py --checkpoint <checkpoints_path>
```

# Acknowledgement
I appreciate these refenrence implementations!
- We used the repository [ACT](https://github.com/Shaka-Labs/ACT) as a reference implementation for teleoperation of koch-v1-1.
- We adapted model arthictectures and hyper parameters for training and inferecne from [original ACT repository](https://github.com/tonyzhaozh/act).
- We used the low cost robot arm [koch-v1-1](https://github.com/jess-moss/koch-v1-1) as an actual robot for evaluating policy performacne.
