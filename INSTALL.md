# INSTALLATION GUIDE

## JAX INSTALLATION

This library can be used only with CPUs, but using GPUs would result in the best performance.  
Here, a typical installation steps of JAX library are described.

⇒ [Official guide (Installation - JAX documentation)](https://docs.jax.dev/en/latest/installation.html)

### Prerequisities

- Using relatively new nvidia graphic cards
- Using relatively new Ubuntu OS (or WSL2)

### 1. NVIDIA driver

You need to install NVidia driver for the cards.

#### i. Disable the default GPU driver

Nouveau is a open-source driver for nvidia graphic cards.  
This might be enabled by default, and can cause a problem if you added the proprietary drivers from NVIDIA (like the proper one is not loaded).

Adding the following file will disable default driver.

```conf
sudo vim /etc/modprobe.d/blacklist-nouveau.conf
```

Save the following lines:

```
blacklist nouveau
options nouveau modeset=0
```

and update the boot sequence:

```
sudo update-initramfs -u
```

Now you don't have the default driver.

#### ii. Find the right driver

(Optional) Remove old drivers (and cuda, cudnn) if you have:

```
dpkg -l | grep nvidia
dpkg -l | grep cuda
dpkg -l | grep cudnn
```

```
sudo apt-get --purge remove nvidia-*
sudo apt-get --purge remove cuda-*
```

List the appropriate drivers for your environment:

```
ubuntu-drivers devices
```

and use "recommended" one:

```
...
vendor   : NVIDIA Corporation
model    : AD102 [GeForce RTX 4090]
driver   : nvidia-driver-550 - distro non-free recommended
...
```

> NOTE: `server` is usually not for us, with consumer models (like GTX, RTX).  
  They are for high-end datacenter GPUs.

(Optional) If nothing was found or your GPU is very old, add PPA repository and try it again:

```
sudo add-apt-repository ppa:graphics-drivers/ppa
ubuntu-drivers devices
```

#### iii. Install the driver

```
sudo apt install nvidia-driver-550
sudo reboot
```

Now check if the GPU is recognized:

```
nvidia-smi
```

### 2. CUDA

Here is the official guide:

- [NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

Latest version can be easily installed from here: [Download the NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).

For Ubuntu 24.04, the easiest for CUDA Toolkit 12.8 is:

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
```

### 3. CUDNN

CUDNN is the library mostly for deep-learning, but JAX also requires this.

Follow the officical guide here: [NVIDIA cuDNN](https://developer.nvidia.com/cudnn).

Latest version can be easily installed from here: [Download cuDNN Library](https://developer.nvidia.com/cudnn-downloads).

Make sure you install the version for the CUDA you installed above.

For Ubuntu 24.04, 

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cudnn
```

### JAX and test

Now you can install JAX with CUDA12 by:

```
pip install -U "jax[cuda12]"
```

You can check if GPUs are correctly recognized by JAX:

```
python -c "import jax; print(jax.devices())"
```

If you see `CudaDevice`, congratulations!  
You have sucessfully installed JAX with GPU acceleration.
