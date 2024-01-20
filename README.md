# Slapo Artifact

This repository contains scripts for setting up environments and reproducing results presented in the ASPLOS 2024 paper entitled [Slapo: A Schedule Language for Progressive Optimization of Large Deep Learning Model Training](https://arxiv.org/abs/2302.08005). If you wish to access the core implementation, documentation, and tutorials for the Slapo language, please refer to the following links. We encourage you to explore these resources if you are interested using Slapo for training other deep learning models that are not presented in our paper.

* Slapo repository: <https://github.com/awslabs/slapo>
* Slapo documentation: <https://awslabs.github.io/slapo/>

As our experiment requires a machine with 8 x V100 GPUs, we provide an AWS [p3dn.24xlarge](https://aws.amazon.com/ec2/instance-types/p3/) EC2 instance for the AE reviewers to reproduce our results. Please contact the authors for the server access.

In the following, we detail the steps to prepare the environment and reproduce the results. For the AE reviewers, please directly go to the [Clone-the-Repository](#clone-the-repository) section and the [Kick-the-Tires](#kick-the-tires) section to run the pre-built docker image on our server.

## Prerequisite

We require NVIDIA Container Toolkit to setup environments, please follow instructions from the [installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), below is the installation script for Ubuntu (extracted from official guide):

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

## Clone the Repository

Please clone the repository with the `--recursive` flag to include the submodules.

```bash
git clone https://github.com/chhzh123/slapo-artifact.git --recursive
cd slapo-artifact
```

## Setup Docker Image

We have already built the docker on our server, so the AE reviewers do *not* need to build the docker image again.

### Pull from Docker Hub

We provide a pre-built docker image available on Docker Hub which is compatible with Volta architecture NVIDIA GPUs, user can pull it with:
```bash
docker image pull chhzh123/slapo-ae:latest
docker tag chhzh123/slapo-ae:latest slapo
```

### Build from source

After these steps, user can run the following command to build docker container. It will take about 2 hours to build the docker image. The docker image contains the necessary environment, libraries, source code, and datasets for reproducing the results.
```bash
docker build -t slapo -f docker/Dockerfile .
```

## Kick-the-Tires (Est. Time: 3 mins)

We will run some basic tests on the artifact to make sure the environment is set up correctly. User can run the following command to run the unit tests using 2 GPUs:
```bash
docker run -it --gpus all slapo /bin/bash -c 'cd /home/deepspeed/slapo && torchrun --nproc_per_node 2 -r 1:1 -m pytest -rxXs -p "no:randomly" tests'
```

You are expected to see the following output.

```
=================================================================================== short test summary info ====================================================================================
SKIPPED [1] tests/test_shard.py:275: Flaky test
========================================================================= 56 passed, 1 skipped, 18 warnings in 16.83s ==========================================================================
```

## Run Experiments (Est. Time: 3 hours)

Below is the script to reproduce experiments in Slapo paper, each script would emit logging files and figures in pdf format.

We only provide scripts for reproducing the results of **Figure 7**, **Table 4**, and **Figure 9**, which constructs the main conclusion of our paper. For other experiments, since it may need multiple machines or take excessively long time to run, we do not provide end-to-end evaluation scripts, but users can still find the instructions in the folder.


### Figure 7 (Est. Time: 2.5 hours)
> Conclusion: Either Slapo-TP or Slapo-ZeRO3 can achieve the best performance among all the baselines.

We recommend using `tmux` to run the experiment in the background. Please make sure that when you launch the experiment, there are no other users using GPUs. You can check by typing `nvidia-smi`.

Some of the experiments may run into out-of-memory (OOM) issue due to the dynamic memory allocation nature of the baseline systems. Please rerun the experiment or reduce the batch size if it happens.

```bash
docker run -it --gpus all --shm-size=150G -v $(pwd)/end2end/:/home/deepspeed/slapo/benchmark/configs slapo /bin/bash -c 'cd /home/deepspeed/slapo/benchmark && cp configs/run.sh run.sh && bash run.sh'
```

Check the `end2end/single_node_v100_1b.pdf` figure.

### Table 4 (Est. Time: 1 min)
> Conclusion: Users can write a few lines of schedule code to optimize model training.

```bash
docker run -it -v $(pwd)/usability/:/home/deepspeed/slapo/usability/script slapo /bin/bash -c 'cd /home/deepspeed/slapo/usability/script && python3 loc.py ../'
```

The lines of code (LoC) count is output to the terminal.

```
wideresnet.schedule: 12
t5.schedule: 11
roberta.schedule: 21
llama.schedule: 11
opt.schedule: 10
bert.schedule: 21
gpt.schedule: 10
```

### Figure 9 (Est. Time: 10 mins)
> Conclusion: Users can leverage Slapo to *progressively* optimize the model performance.

```bash
docker run -it --gpus all --shm-size=150G -v $(pwd)/ablation/:/home/deepspeed/slapo/benchmark/script slapo /bin/bash -c 'cd /home/deepspeed/slapo/benchmark/script && bash run.sh'
```

Check the `ablation/ablation-1b.pdf` figure.


### Figure 8 (Not for AE)
This experiment requires 8 x p3dn instances (64 x V100 GPUs) to evaluate the GPT-10B and LLaMA-7B models, and thus is not for artifact evaluation. The corresponding scripts are listed below:

* [GPT-10B](https://github.com/chhzh123/slapo/blob/asplos24/examples/gpt2/run.sh)
* [LLaMA-7B](https://github.com/chhzh123/slapo/blob/asplos24/examples/llama/run.sh)


### Figure 10 (Not for AE)
This experiment requires excessive computing resources to evaluate all the configuration points and summarize the data manually, and thus is not for artifact evaluation.

For the usage of autotuner, please refer to [tune_cfg.py](https://github.com/chhzh123/slapo/blob/asplos24/examples/opt/tune_cfg.py). For the figure plotting script, please refer to [autotune.py](https://github.com/chhzh123/slapo/blob/asplos24/benchmark/plot/autotune.py).


## Further customization

### End-to-end Evaluation

Modify the script.


### Tutorials
Please refer to the Slapo [documentation](https://awslabs.github.io/slapo/) for the usage of Slapo.


## More information
For AE reviewers, please contact the authors through HotCRP for any questions. For other users, please open an [issue](https://github.com/chhzh123/slapo-artifact/issues) publicly or contact [chhzh123](mailto:hzchen@cs.cornell.edu) for any techinical questions.


## Reference
* [SparseTIR artifact](https://github.com/uwsampl/sparsetir-artifact)
* [nnSmith artifact](https://github.com/ganler/nnsmith-asplos-artifact)
