**⚠️ This repository is no longer maintained**

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
# Pretrained models will be released soon. Stay tuned!

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

# Reconstructing Custom Data 
The COLMAP workspace should be looking like this
└── brandenburg_gate
  └── brandenburg_gate.tsv
  ├── cache_sgs
    └── splits
        ├── rays1_meta_info.json
        ├── rgbs1_meta_info.json
        ├── split_0
            ├── rays1.h5
            └── rgbs1.h5
        ├── split_1
        ├──.....
  ├── config.yaml
  ├── dense
    └── sparse
        ├── cameras.bin
        ├── images.bin
        ├── points3D.bin
  └── semantic_maps
      ├── 99119670_397881696.jpg
      ├── 99128562_6434086647.jpg
      ├── 99250931_9123849334.jpg
      ├── 99388860_2887395078.jpg
      ├──.....

- Obtain relevant parameters
note：Modify the path
```bash
bash scripts/preprocess_data.sh 
```
After running bash 'scripts/preprocess_data.sh', Create a file config.yaml into workspace to write metadata. The target scene needs to be normalized into a unit sphere, which require manual selection. One simple way is to use SFM key-points points from COLMAP to determine the origin and radius. Also, a bounding box is required, which can be set to [origin-raidus, origin+radius], or only the region you're interested in.
{
    name: brandenburg_gate, # scene name
    origin: [ 0.568699, -0.0935532, 6.28958 ], 
    radius: 4.6,
    eval_bbx: [[-14.95992661, -1.97035599, -16.59869957],[48.60944366, 30.66258621, 12.81980324]],
    voxel_size: 0.25,
    min_track_length: 10,
    # The following configuration is only used in evaluation, can be ignored for your own scene
    sfm2gt: [[1, 0, 0, 0],
            [ 0, 1, 0, 0],
            [ 0, 0, 1, 0],
            [ 0, 0, 0, 1]],
}

- Obtain the segmentation images (in case the ones above have incorrect resolution).
```bash
- python tools/prepare_data/prepare_semantic_maps.py --root_dir path/to/the/data/heritage-recon/custom_data --gpu 1
```
- Segment training and testing images
```bash
python tools/prepare_data/prepare_data_split.py --root_dir data/heritage-recon/stand --num_test 1 --min_observation -1 --roi_threshold 0 --static_threshold 0
```

- Generate training data cache
```bash
python tools/prepare_data/prepare_data_cache.py --root_dir data/heritage-recon/custom_data --dataset_name phototourism --cache_dir cache_sgs --img_downscale 1 --semantic_map_path semantic_maps --split_to_chunks 64 
```
- extracting mesh from a checkpoint:
```bash
bash scripts/sdf_extract.sh gate config/train_custom_data.yaml path/to/the/model.ckpt 10
```

# Acknowledgement
Part of our code is derived from [nerf_pl](https://github.com/kwea123/nerf_pl) , [NeuS](https://github.com/Totoro97/NeuS), [NeuralRecon-W](https://github.com/zju3dv/NeuralRecon-W),  [Neuralangelo](https://github.com/NVlabs/neuralangelo). We sincerely acknowledge and appreciate the outstanding contributions of their authors.

