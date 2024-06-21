# AWS installation

Current reposiory contains pre-built docker images to run in any cloud / on-premise platform.
This is the recommended way to run in in destineed containers, as they are compatible and tested in GPU and CPU setups,
and they are a basis for containerized distributed scheme.

In the given file you will find installation instructions to run in [Amazon SageMaker](https://aws.amazon.com/sagemaker/), but they are applicable to according EC2 instances.

## Pre-requisites

1. Familiriality with AWS cloud is assumed.
2. Root or IAM account is configured.
3. *****Disclaimer: AWS is a paid service, and any computations imply costs.**
4. Navigate to the [console](https://us-east-1.console.aws.amazon.com/console/home?region=us-east-1#) for your selected region.
5. Create or run your [SageMaker](https://us-east-1.console.aws.amazon.com/sagemaker/home?region=us-east-1#/notebook-instances) instance
6. Open jupyter lab
7. Upload your files.
8. Click Terminal among available options. Validate `nvidia-smi` to make sure that drivers are successfully installed.
9. Run docker compose build && docker compose up -d according to instructions.


Alternatively one may try to setup the appropriate image to EC2 together with drivers, and install application as per guide.

# CUDA ON EC2 FROM SCRATCH
This instruction helps to set up Pytorch with CUDA on an EC2 instance with plain, Ubuntu AMI.

## Pre-installation actions
1) Verify the instance has the CUDA-capable GPU
```
lspci | grep -i nvidia
```

2)  Install kernel headers and development packages
```
sudo apt-get install linux-headers-$(uname -r)
```

## NVIDIA drivers installation
1) Download a CUDA keyring for your distribution $distro and architecture $arch
```
wget https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-keyring_1.1-1_all.deb
```
i.e. for Ubuntu 22.04 with x86_64 the command would look as follows:
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
```
2) Add the downloaded keyring package
```
sudo dpkg -i cuda-keyring_1.1-1_all.deb
```
3) Update the APT repository cache
```
sudo apt-get update
```
4) Install the drivers
```
sudo apt-get -y install cuda-drivers
```
5) Reboot the instance
```
sudo reboot
```
6) Verify the installation
```
nvidia-smi
```
It is important to keep in mind CUDA Version is displayed in the upper-right corner, as PyTorch needs to be compatible with it.

**NOTE:** At this stage NVIDIA recommends following [Post-installation actions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#environment-setup). I didn't and it worked but some unexpected errors might occur.
## PyTorch installation

### Install package manager
I used conda but pip+venv *should* also work
1) Install conda
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```
2) Initialize conda 
```
~/miniconda3/bin/conda init bash
```
3) Reload bash
```
source ~/.bashrc
```
4) Create a new conda environment
```
conda create -n env 
```
5) Activate the newly created environment
```
conda activate env
```
