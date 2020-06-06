# siim melanome kaggle

## Conventions
* `data/` contains the data (not committed on git)
* `src/` contains the scripts and code, for reproducing results
* `notebooks/` contains the Jupyter notebooks. Notebooks should be personal, named `<USER>-<NAME>.ipynb`.
For instance: `julien-resnetstests.ipynb`.

## Docker setup (on Linux)
* Install the nvidia-docker engine for your distribution. See here for how to install: https://github.com/NVIDIA/nvidia-docker
* Install the latest nvidia drivers (440 or more recent for using Cuda 10.2). 
  On Ubuntu: `sudo apt-get install nvidia-driver-440`
* Build the Docker image: `./build_docker.sh`
* Run the Docker container: `./run_docker.sh`

This will run a Jupyter server, which you can access from your browser at `<YOUR_IP>:8888`,
and copy-pasting the token being displayed on your terminal.

Within the container, the data is available in `/workdir/data` and `src` in `/workdir/src`.
This is *not* a copy of the original files, so editing them from within the container will edit the original files.

You can run a script from Jupyter using `%run path-to-src/my_script.py`.

Edit `Dockerfile` and re-build if you need to add/remove packages. 
To just launch a command such as `bash`, you can connect to the running container using
`docker exec -it siim bash` 
