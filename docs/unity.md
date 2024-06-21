# Running an Apptainer container on Unity

Install `spython` utility to make an Apptainer definition file from Dockerfile:
```
pip install spython
spython recipe Dockerfile1 > Apptainer1.def 
```
Modify the definition file with the following environment settings:
```
# Make non-interactive environment.
export TZ='America/New_York'
export DEBIAN_FRONTEND=noninteractive
```
Build the Apptainer image (sif):
```
module load apptainer/latest 
unset APPTAINER_BINDPATH
apptainer build --fakeroot sifs/drlearner1.sif Apptainer1.def 
```
Allocate a computational node and load the required modules:
```
salloc -N 1 -n 1 -p gpu-preempt -G 1  -t 2:00:00 --constraint=a100
module load cuda/11.8.0
module load cudnn/8.7.0.84-11.8 
```
Run the container:
```
apptainer exec --nv sifs/drlearner1.sif bash
```
Export a user's WANDB key for logging the job (for illustrative purposes only!):
```
export WANDB_API_KEY=c5180d032d5325b08df49b65f9574c8cd59af6b1
```
Run the Atari example:
```
python3.10 examples/distrun_atari.py --exp_path experiments/apptainer_test_distrun_atari --exp_name apptainer_test_distrun_atari
```