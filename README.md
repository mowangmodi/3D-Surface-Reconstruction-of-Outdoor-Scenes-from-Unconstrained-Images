

# 💻 Installation
This implementation only includes libraries that have been actually tested and used, along with their corresponding hardware and software environments. Components that were not used are not listed here.
# Hardware
- **OS**: Ubuntu 20.04  
- **GPU**: NVIDIA GPU with Compute Capability ≥ 8.6 and memory > 24GB  *(Tested on RTX 4090 with CUDA 11.7; other versions may also be compatible))* 
- **RAM**: 128GB 
#  Software

Make sure you have Git installed. Then run the following command to clone the repository:

```bash
git clone https://github.com/mowangmodi/3D-Surface-Reconstruction-of-Outdoor-Scenes-from-Unconstrained-Images.git.git
```
-  Python ≥ 3.8 is required. It is recommended to use [Anaconda](https://www.anaconda.com/) for environment management.

- You can create the Conda environment using either of the following methods:

```bash

conda env create -f environment.yaml
# Download the segmentation model
scripts/download_sem_model.sh
```
# Dataset setup
Download the [Heritage-Recon](https://drive.google.com/drive/folders/1eZvmk4GQkrRKUNZpagZEIY_z8Lsdw94v?usp=sharing) dataset and put it under `data`. You can also use [gdown](https://github.com/wkentaro/gdown) to download it in command line:
```bash
mkdir data && cd data
gdown --id 1eZvmk4GQkrRKUNZpagZEIY_z8Lsdw94v
```
Generate ray cache for all four scenes:
```bash
for SCENE_NAME in brandenburg_gate pantheon_exterior; do
  scripts/data_generation.sh data/heritage-recon/${SCENE_NAME}
done
```
# Training
To train scenes in our Heritage-Recon dataset:
```bash
python train.py --cfg_path /path/to/the/config/train_pantheon_exterior.yaml --num_gpus 1 \
				--num_nodes 1 --num_epochs 5 \
				--batch_size 512 --test_batch_size 512 \
				--num_workers 8 --exp_name exp_pe
```
# Evaluating
First, extracting mesh from a checkpoint you want to evaluate:
```bash
bash scripts/sdf_extract.sh gate config/train_pantheon_exterior.yaml path/to/the/model.ckpt 10
```
The reconstructed meshes will be saved to `PROJECT_PATH/results`.

Then run the evaluation pipeline:
- Reprojection Filtering
```bash
python utils/reproj_filter.py \
    --src_file     /path/to/the/results/extracted_mesh_level_10.ply \
    --target_file  /path/to/the/results/extracted_mesh_level_10.ply \
    --data_path    /path/to/the/data/heritage-recon/pantheon_exterior \
    --output_path  results/phototourism/ \
    --n_cpus      2 \
    --n_gpus      2
```
- Mesh Evaluation
```bash
python utils/eval_mesh.py \
  --file_pred           /path/to/the/results/extracted_mesh_level_10.ply \
  --file_trgt           /path/to/the/data/heritage-recon/pantheon_exterior/pantheon_exterior.ply \
  --scene_config_path   /path/to/the/data/heritage-recon/pantheon_exterior/config.yaml \
  --threshold           0.01,1,0.01 \
  --save_name           pantheon_exterior_reprojected.ply \
  --sfm_path            /path/to/the/data/heritage-recon/pantheon_exterior/neuralsfm \
  --track_lenth         12 \
  --reproj_error        1.4 \
  --voxel_size          0.1 \
  --bbx_name            eval_bbx
```

# Reconstructing custom data



# Acknowledgement
Part of our code is derived from [nerf_pl](https://github.com/kwea123/nerf_pl) , [NeuS](https://github.com/Totoro97/NeuS), [NeuralRecon-W](https://github.com/zju3dv/NeuralRecon-W),  [Neuralangelo](https://github.com/NVlabs/neuralangelo). We sincerely acknowledge and appreciate the outstanding contributions of their authors.

