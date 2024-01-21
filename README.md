# Slapo Artifact

This repository contains scripts for setting up environments and reproducing results presented in the ASPLOS 2024 paper entitled [Slapo: A Schedule Language for Progressive Optimization of Large Deep Learning Model Training](https://arxiv.org/abs/2302.08005). If you wish to access the core implementation, documentation, and tutorials for the Slapo language, please refer to the following links. We encourage you to explore these resources if you are interested using Slapo for training other deep learning models that are not presented in our paper.

* Slapo repository: <https://github.com/awslabs/slapo>
* Slapo documentation: <https://awslabs.github.io/slapo/>

As our experiment requires a machine with 8 x V100 GPUs, we provide an AWS [p3dn.24xlarge](https://aws.amazon.com/ec2/instance-types/p3/) EC2 instance for the AE reviewers to reproduce our results. Please contact the authors for the server access.

In the following, we detail the steps to prepare the environment and reproduce the results. For the AE reviewers, please directly go to the [Clone-the-Repository](#clone-the-repository) section and the [Kick-the-Tires](#kick-the-tires) section to run the pre-built docker image on our server.

## Prerequisite

We require NVIDIA Container Toolkit to setup environments, please follow instructions from the [installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). For convenience, we also provide the installation script below (extracted from official guide):

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

We have already built the docker on our server, so the AE reviewers do *not* need to pull or build the docker image again.

### Pull from Docker Hub

We provide a pre-built docker image available on Docker Hub which is compatible with NVIDIA GPUs with CUDA 11.7 installed, user can pull it with:
```bash
docker image pull chhzh123/slapo-ae:latest
docker tag chhzh123/slapo-ae:latest slapo
```

### Build from source

Otherwise, users can run the following command to build docker container. It will take about **2 hours** to build the docker image. The docker image contains the necessary environment, libraries, source code, and datasets for reproducing the results.
```bash
cd 3rdparty/slapo/docker
docker build -t slapo -f docker/Dockerfile .
```

## Kick-the-Tires (Est. Time: 3 mins)

We will run some basic tests on the artifact to make sure the environment is set up correctly. User can use the following command to run the unit tests using 2 GPUs:
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

Below is the script to reproduce experiments in Slapo paper. Each script will emit logging files and figures in pdf format.

We only provide scripts for reproducing the results of **Figure 7**, **Table 4**, and **Figure 9**, which constructs the main conclusion of our paper. For other experiments, since they may require multiple machines or take excessively long time to run, we do not provide end-to-end evaluation scripts, but users can still find the instructions in our repository.


### Figure 7 (Est. Time: 2.5 hours)
> Conclusion: Either Slapo-TP or Slapo-ZeRO3 can achieve the best performance among all the baselines.

We recommend using `tmux` to run the experiment in the background. For the AE reviewers, please coordinate the time to run the experiments and make sure that when you launch the experiment, there are no other users using GPUs. You can check by typing `nvidia-smi`. The following command shows how to launch the experiment:

```bash
docker run -it --gpus all --shm-size=150G -v $(pwd)/end2end/:/home/deepspeed/slapo/benchmark/configs slapo /bin/bash -c 'cd /home/deepspeed/slapo/benchmark && cp configs/run.sh run.sh && bash run.sh'
```

Please check the `end2end/single_node_v100_1b.pdf` figure.

Note: Some of the experiments may run into out-of-memory (OOM) issue due to the dynamic memory allocation nature of the baseline systems. Please rerun the experiment or reduce the batch size if it happens. Also, some baseline systems (e.g., Megatron-LM and DeepSpeed) may have better performance than the ones presented in the paper due to the recent updates of their systems, but the above conclusion should still hold.


### Table 4 (Est. Time: 1 min)
> Conclusion: Users can write a few lines of schedule code to optimize model training in a distributed environment.

Please run the following command to count the lines of code (LoC) of the schedule. Those schedules are corresponding to the ones used in the Figure 7 experiment.

```bash
docker run -it -v $(pwd)/usability/:/home/deepspeed/slapo/usability/script slapo /bin/bash -c 'cd /home/deepspeed/slapo/usability/script && python3 loc.py ../'
```

The LoC count is output to the terminal.

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

Please run the following command to generate the figure.

```bash
docker run -it --gpus all --shm-size=150G -v $(pwd)/ablation/:/home/deepspeed/slapo/benchmark/script slapo /bin/bash -c 'cd /home/deepspeed/slapo/benchmark/script && bash run.sh'
```

Please check the `ablation/ablation-1b.pdf` figure.


### Figure 8 (Not for AE)
This experiment requires 8 x p3dn instances (64 x V100 GPUs) to evaluate the GPT-10B and LLaMA-7B models, and thus is not for artifact evaluation. However, we provide the corresponding scripts for users who may be interested in reproducing our results in a larger-scale environment:

* [GPT-10B](https://github.com/chhzh123/slapo/blob/asplos24/examples/gpt2/run.sh)
* [LLaMA-7B](https://github.com/chhzh123/slapo/blob/asplos24/examples/llama/run.sh)


### Figure 10 (Not for AE)
This experiment requires excessive computing resources to evaluate all the configuration points and summarize the data manually, and thus is not for artifact evaluation.

For the usage of autotuner, please refer to [tune_cfg.py](https://github.com/chhzh123/slapo/blob/asplos24/examples/opt/tune_cfg.py). For the figure plotting script, please refer to [autotune.py](https://github.com/chhzh123/slapo/blob/asplos24/benchmark/plot/autotune.py).


## Further Customization

The Slapo language and the artifact can be further customized for running more models or different settings.

### End-to-end Evaluation
You can run this artifact using your own machine once you have the GPU environment set up. For example, if you only have one GPU on your machine and still want to try out Slapo, you can still modify the benchmarking script to run the experiment.

We use a simple config file to control what to benchmark. A config file is composed of `N` lines, where each line indicates a benchmark case. The format is as follows. Note that you can add `"#"` in the beginning of any line to skip that configuration.

```
MODE MODEL GPUS SEQ_LEN DEC_SEQ_LEN BATCH_SIZE CKPT
```

* `MODE`: megatron, slapo-megatron, deepspeed, or slapo-deepspeed
* `MODEL`: HuggingFace model name (e.g., bert-large-uncased)
* `GPUS`: Number of GPUs (e.g., pow2, or 2,4,8)
* `SEQ_LEN`: Sequence length. In encoder-decoder model, this is the encoder length.
* `DEC_SEQ_LEN`: The decoder length. This is only used by encoder-decoder models.
* `BATCH_SIZE`: An expression that inputs GPU number and outputs batch size (e.g., "16*n" means batch size 16, 32, 64, 128 respecting to GPU number 1, 2, 4, 8).
* `CKPT`: Activation checkpointing. In Megatron it would be full or selective. In Slapo it is a floating point indicating the checkpoint ratio (e.g., 1.0 means full).

For example, if you only have one GPU and want to run the BERT-large model using Slapo-TP with batch size 16 and sequence length 512, you can write the following line in the config file named `end2end/test.cfg`.
```
slapo-megatron bert-large-uncased 1 512 0 16 0
```

And then launch the docker in the root folder:
```bash
docker run -it --gpus all --shm-size=150G -v $(pwd)/end2end/:/home/deepspeed/slapo/benchmark/configs slapo /bin/bash -c 'cd /home/deepspeed/slapo/benchmark && cp configs/test.cfg test.cfg && bash run_all_single_node.sh configs/test.cfg'
```

The output will be displayed in the terminal. Our script makes the artifact more reusable for different models and hardware enviroments.


### Tutorials
We also provide detailed documentation and tutorials for users who are interested in using Slapo for training other deep learning models. Please refer to this [webpage](https://awslabs.github.io/slapo/) for more information.


## More information
For AE reviewers, please contact the authors through [HotCRP](https://asplos24aec-summer.hotcrp.com/) for any questions. For other users, please open an [issue](https://github.com/chhzh123/slapo-artifact/issues) publicly or contact [chhzh123](mailto:hzchen@cs.cornell.edu) for any techinical questions.

Please cite our paper if you use Slapo for your research.
```bibtex
@inproceedings{chen2024slapo,
      title={Slapo: A Schedule Language for Progressive Optimization of Large Deep Learning Model Training}, 
      author={Hongzheng Chen and Cody Hao Yu and Shuai Zheng and Zhen Zhang and Zhiru Zhang and Yida Wang},
      year={2024},
      booktitle={Proceedings of the ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS)},
}
```


## Reference
* [SparseTIR artifact](https://github.com/uwsampl/sparsetir-artifact)
* [nnSmith artifact](https://github.com/ganler/nnsmith-asplos-artifact)
