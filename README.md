# Slapo Artifact

This repository contains scripts for setting up environments and reproducing results presented in the ASPLOS 2024 paper entitled [Slapo: A Schedule Language for Progressive Optimization of Large Deep Learning Model Training](https://dl.acm.org/doi/10.1145/3582016.3582047). To access the core implementation of Slapo, please visit the [Slapo repository](https://github.com/awslabs/slapo). Additionally, we have documentation and tutorials on the [Slapo language](https://awslabs.github.io/slapo/). Please take a look if you are interested.

As our experiment requires a machine with 8xV100 GPUs, we provide an AWS [p3dn.24xlarge](https://aws.amazon.com/ec2/instance-types/p3/) EC2 instance for the reviewers to reproduce our results. Please contact the authors for access.

For AE reviewers, please directly go to the [Clone-the-Repository](#clone-the-repository) section and the [Kick-the-Tires](#kick-the-tires) section to run the pre-built docker image.

## Prerequisite

We require NVIDIA Container Toolkit to setup environments, please follow instructions from [installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), below is the installation script for Debian/Ubuntu (extracted from official guide):

```bash
curl https://get.docker.com | sh && sudo systemctl --now enable docker

distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

User can try the following command to test whether the installation was successful or not:
```bash
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
```

## Clone the Repository

```bash
git clone https://github.com/chhzh123/slapo-artifact.git --recursive
cd slapo-artifact
```

## Setup Docker Image

### Pull from Docker Hub

We provide a pre-built docker image available on Docker Hub which is compatible with Volta architecture NVIDIA GPUs, user can pull it with:
```bash
docker image pull chhzh123/slapo-ae:latest
docker tag chhzh123/slapo-ae:latest slapo
```

### Build from source

After these steps, user can run the following command to build docker container. It will take about 2 hours to build the docker image.
```bash
docker build -t slapo .
```

## Kick-the-Tires (Est. Time: 3 mins)

We will run some basic tests on the artifact to make sure the environment is set up correctly. User can run the following command to enter the docker container:
```bash
docker run -it --gpus all slapo /bin/bash -c 'cd /home/deepspeed/slapo && torchrun --nproc_per_node 2 -r 1:1 -m pytest -rxXs -p "no:randomly" tests'
```

You should see the following output.

```
=================================================================================== short test summary info ====================================================================================
SKIPPED [1] tests/test_shard.py:275: Flaky test
========================================================================= 56 passed, 1 skipped, 18 warnings in 16.83s ==========================================================================
```

## Run Experiments (Est. Time: 2.5 hours)

Below is the script to reproduce experiments in Slapo paper, each script would emit logging files and figures in pdf format.

We only provide scripts for reproducing the results of **Figure 7**, **Table 4**, and **Figure 9**, which constructs the main conclusion of our paper. For other experiments, since it may need multiple machines or take excessively long time to run, we do not provide end-to-end evaluation scripts, but users can still find the instructions in the folder.


### Figure 7 (Est. Time: 2 hours)
> Conclusion: Either Slapo-TP or Slapo-ZeRO3 can achieve the best performance among all the baselines.

We recommend using `tmux` to run the experiment in the background. Please make sure that when you launch the experiment, there are no other users using GPUs. You can check by typing `nvidia-smi`.

Some of the experiments may run into out-of-memory (OOM) issue due to the dynamic memory allocation nature of the baseline systems. Please rerun the experiment or reduce the batch size if it happens.

```bash
# Run single-node end-to-end experiments
docker run -it --gpus all -v $(pwd)/spmm/:/root/spmm sparsetir /bin/bash -c 'cd spmm && bash run.sh'
```

### Table 4 (Est. Time: 1 min)
> Conclusion: Users can write a few lines of schedule code to optimize model training.

```bash
docker run -it -v $(pwd)/usability/:/home/deepspeed/slapo/usability/script slapo /bin/bash -c 'cd /home/deepspeed/slapo/usability/script && python3 loc.py ../'
```

The lines of code (LoC) count is shown below.

```
wideresnet.schedule: 12
t5.schedule: 11
roberta.schedule: 21
llama.schedule: 11
opt.schedule: 10
bert.schedule: 21
gpt.schedule: 10
```

### Figure 9 (Est. Time: 5 mins)
> Conclusion: Users can leverage Slapo to progressively optimize the model performance.

```bash
docker run -it --gpus all --shm-size=150G -v $(pwd)/ablation/:/home/deepspeed/slapo/benchmark/script slapo /bin/bash -c 'cd /home/deepspeed/slapo/benchmark/script && bash run.sh'
```


### Figure 8 (Not for AE)
This experiment requires 8 x p3dn instances to evaluate the GPT-10B and LLaMA-7B models, and thus is not for artifact evaluation. The corresponding scripts are listed below:

* [GPT-10B](https://github.com/chhzh123/slapo/blob/asplos24/examples/gpt2/run.sh)
* [LLaMA-7B](https://github.com/chhzh123/slapo/blob/asplos24/examples/llama/run.sh)


### Figure 10 (Not for AE)
This experiment requires excessive computing resources to evaluate all the configuration points and summarize the data manually, and thus is not for artifact evaluation.

For the usage of autotuner, please refer to [tune_cfg.py](https://github.com/chhzh123/slapo/blob/asplos24/examples/opt/tune_cfg.py). For the figure plotting script, please refer to [autotune.py](https://github.com/chhzh123/slapo/blob/asplos24/benchmark/plot/autotune.py).


## Further customization
Please refer to the Slapo [documentation]() for the usage of Slapo.


## Reference
* [SparseTIR artifact](https://github.com/uwsampl/sparsetir-artifact)
* [nnSmith artifact](https://github.com/ganler/nnsmith-asplos-artifact)
