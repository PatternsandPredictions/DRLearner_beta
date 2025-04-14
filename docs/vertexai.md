# Running on Vertex AI

## Installation and set-up

1. (Local) Install `gcloud`.
```
sudo apt-get install apt-transport-https ca-certificates gnupg curl

echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
sudo apt-get update && sudo apt-get install google-cloud-sdk
 ```

2. (Local) Set up GCP project.
```
gcloud init # choose the existing project or create a new one
export GCP_PROJECT=<GCP project ID>
echo $GCP_PROJECT # make sure it's the DRLearner project
conda env config vars set GCP_PROJECT=<GCP project ID> # optional
```
3. (Local) Authorise the use of GCP services by DRLearner.
```
gcloud auth application-default login # get credentials to allow DRLearner code calls to GC APIs
export GOOGLE_APPLICATION_CREDENTIALS=/home/<user>/.config/gcloud/application_default_credentials.json
conda env config vars set GOOGLE_APPLICATION_CREDENTIALS=/home/<user>/.config/gcloud/application_default_credentials.json # optional
```
4. (Local) Install and configure Docker.
```
sudo apt-get remove docker docker-engine docker.io containerd runc
sudo apt-get update && sudo apt-get install lsb-release
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
  
sudo groupadd docker
sudo usermod -aG docker <user>
  
gcloud auth configure-docker
```

5. (GCP console) Enable IAM, Enable Vertex AI, Enable Container Registry in `<GCP project ID>`.


6. (GCP console) Set up a xmanager service account.
- Create xmanager service account in `IAM & Admin/Service accounts` .
- Add 'Storage Admin', 'Vertex AI Administrator', 'Vertex AI User' , 'Service Account User' roles.
- If you will be building docker images on GCP and not locally, consider adding 'Cloud Build Editor' and 'Storage Object Viewer' roles as well
7. Set up a Cloud storage bucket.
- (GCP console) Create a Cloud storage bucket in Cloud Storage in `us-central1` region.
- (Local) `export GOOGLE_CLOUD_BUCKET_NAME=<bucket name>`
- (Local, optional) `conda env config vars set GOOGLE_CLOUD_BUCKET_NAME=<bucket name>`

8. (Local) Replace `envs/drlearner/lib/python3.10/site-packages/launchpad/nodes/python/xm_docker.py` with `./external/xm_docker.py`  (to get the correct Docker instructions)*

    *Can't rebuild launchpad package with those changes because the of complicated build process (requires Bazel...)


9. (Local) Replace `envs/drlearner/lib/python3.10/site-packages/xmanager/cloud/vertex.py` with `./external/vertex.py` (to add new machine types, allow web access to nodes from GCP console).


10. (Local) Tensorboard instructions:
- Use scripts/update_tb.py to download current tfevents file which is saved in `<bucket name>`
```
python update_tb.py <experiment name>/ <path to save> 
```
! We recommend syncing tf files regularly and keeping older versions as well, 
since Vertex AI silently restarts the workers which are down,
and they start writing logs in tf file from scratch !

## GCP Hardware Specs
The hardware requirements for running DRLearner on Vertex AI are specified in `drlearner/configs/resources/` - there are two setups: for easy environment (i.e. Atari Boxing) and a more complex one (i.e. Atari Montezuma Revenge). See the table below.


|               |                Simple   env                |                   Complex env                |
|---------------|:------------------------------------------:|---------------------------------------------:|
| Actor         |       e2-standard-4 (4 CPU, 16 RAM)        |                e2-standard-4 (4 CPU, 16 RAM) |
| Learner       | n1-standard-4 (4 CPU, 16 RAM + TESLA P100) | n1-highmem-16 (16 CPU, 104 RAM + TESLA P100) |
| Replay Buffer |        e2-highmem-8 (8 CPU, 64 RAM)        |              e2-highmem-16 (16 CPU, 128 RAM) |

New configurations can be added using the same xm_docker.DockerConfig and xm.JobRequirements classes. Available for use on Vertex AI machine types are listed here https://cloud.google.com/vertex-ai/pricing.
But it might require adding the new machine names to `external/vertex.py` i.e.  `'n2-standard-64': (64, 256 * xm.GiB),`.

## GCP Troubleshooting
In case of any 'Permission denied' issues, go to `IAM & Admin/` in GCP console and try adding 'Service Account User' role to your User, and
'Compute Storage Admin' role to 'AI Platform Custom Code Service Agent' Service Account.

## Running experiments
```
python ./examples/distrun_atari.py  --run_on_vertex --exp_path /gcs/$GOOGLE_CLOUD_BUCKET_NAME/test_pong/ --level PongNoFrameskip-v4 --num_actors_per_mixture 3 --colocate_learner_and_reverb=false
```
- add `--noxm_build_image_locally` to build Docker images with Cloud Build, otherwise it will be built locally.
- number of nodes running Actor code is `--num_actors_per_mixture` x `num_mixtures` - default number of mixtures for Atari is 32 - so be careful and don't launch the full-scale experiment before testing that everything works correctly.
