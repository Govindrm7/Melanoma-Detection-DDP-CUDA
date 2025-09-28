# Melanoma Detection using Deep Learning, CUDA, and Parallelism

## Overview
This project leverages **deep learning** and **parallel computing (CPU & GPU)** to improve melanoma detection accuracy and training efficiency. By combining **PyTorch DistributedDataParallel (DDP)** with **CUDA acceleration** and **Dask-based CPU parallelism**, the system achieves scalable training across multiple GPUs while also optimizing large dataset preprocessing.

---

## Key Features
- **Deep Learning with Transfer Learning**: Modified CNN architecture with fine-tuned final layers for melanoma classification.
- **Multi-GPU Training with DDP**:
  - Gradient synchronization across GPUs.
  - `torch.multiprocessing.spawn` for process initialization.
  - All-reduce operation for averaging gradients and keeping weights consistent.
- **CUDA Acceleration**:
  - Leveraged **CUDA 11.2 / 11.4** on Tesla K80 GPUs.
  - Achieved significant reduction in epoch training time.
- **Data Parallelism with Dask**:
  - **Distributed Computation**: Used Dask to parallelize image enhancement tasks across multiple CPUs.
  - **Dynamic CPU Allocation**: Configured CPU usage (1, 2, 4, or 8) via `run_dask_with_cpus` function.
  - **Task Graph Execution**: Image processing tasks (train/val/test splits) executed in parallel using `compute(*tasks)`.
  - **Performance Metrics**:
    - **Speedup**: Compared single vs multi-CPU processing time.
    - **Efficiency**: Measured effective utilization of CPUs, with ideal efficiency = 1 for linear scaling.

---

## Dataset
- **Size**: ~12.2 GB of high-resolution dermoscopic images.
- **Classes**:
  - **Melanoma** (malignant, dangerous)
  - **Nevus** (benign mole)
  - **Seborrheic Keratosis** (benign, non-cancerous)
- Each image includes **metadata** with lesion type, diagnosis, and clinical features.

---

## Environment & Setup
- **Cluster**: Discovery (reservation-based)
- **System Architecture**: x86_64
- **Hardware**:
  - 4 × NVIDIA Tesla K80 GPUs  
  - 8 CPUs  
  - 32 GB RAM  
- **Software**:
  - Anaconda3 (2021.05 module)  
  - CUDA 11.2 / 11.4  
  - PyTorch with DDP support  
  - Dask for distributed CPU parallelism  
- **Job Config**:
  - Time limit: 4 hours  
  - Email on job start: Disabled  

---

## Results
- **CPU Parallelism with Dask**:
  - Improved preprocessing speed by distributing tasks.
  - Efficiency decreased slightly as CPU count increased, showing overhead but still major time savings.
- **GPU Parallelism with DDP**:
  - **1 GPU**: Accuracy ~75%, Epoch Time ~360–430s  
  - **2 GPUs**: Accuracy ~85%, Epoch Time ~232s  
  - **4 GPUs**: Accuracy ~90%, Epoch Time ~163s  
- **Observations**:
  - 2 GPUs gave the best trade-off between speedup and efficiency.
  - More GPUs introduced communication overhead and diminishing returns.
  - Dask + DDP together enabled end-to-end acceleration (preprocessing + training).

---

## References
- [PyTorch Distributed Data Parallel](https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html)  
- [Comprehensive Tutorial to PyTorch DDP](https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51)  
- [OSC Guide to PyTorch DDP](https://www.osc.edu/resources/getting_started/howto/howto_pytorch_distributed_data_parallel_ddp)  

---

## Authors  
- **Govind Mudavadkar**
- **Saiyam Doshi**

---

## Future Work
- Mixed-precision training (FP16) for faster GPU performance.  
- Model quantization for lightweight deployment.  
- Expand dataset with more dermatology images for robustness.  
