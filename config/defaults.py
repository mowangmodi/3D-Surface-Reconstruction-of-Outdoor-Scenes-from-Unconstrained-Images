from matplotlib.colors import BoundaryNorm
from yacs.config import CfgNode as CN

_CN = CN()

##############  ↓  NEUS-W Pipeline  ↓  ##############
_CN.NEUCONW = CN()
_CN.NEUCONW.N_SAMPLES = 512
_CN.NEUCONW.N_IMPORTANCE = 512
_CN.NEUCONW.USE_DISP = False
_CN.NEUCONW.PERTURB = 1.0
_CN.NEUCONW.NOISE_STD = 1.0

_CN.NEUCONW.S_VAL_BASE = 0
_CN.NEUCONW.BOUNDARY_SAMPLES = 0
_CN.NEUCONW.NEAR_FAR_OVERRIDE = False
_CN.NEUCONW.VOXEL_SIZE = 0.0
_CN.NEUCONW.MIN_TRACK_LENGTH = 0
_CN.NEUCONW.SAMPLE_RANGE = 4
_CN.NEUCONW.SDF_THRESHOLD = 1e-3
_CN.NEUCONW.TRAIN_VOXEL_SIZE = 0.01
_CN.NEUCONW.UPDATE_FREQ = 2000

_CN.NEUCONW.N_VOCAB = 1500
_CN.NEUCONW.ENCODE_A = True
_CN.NEUCONW.N_A = 48
_CN.NEUCONW.N_STATIC_HEAD = 1
_CN.NEUCONW.ANNEAL_END = 50000

_CN.NEUCONW.RENDER_BG = True
_CN.NEUCONW.UP_SAMPLE_STEP = 4
_CN.NEUCONW.N_OUTSIDE = 32
_CN.NEUCONW.MESH_MASK_LIST = None
_CN.NEUCONW.RAY_MASK_LIST = None
_CN.NEUCONW.ENCODE_A_BG = True
_CN.NEUCONW.FLOOR_NORMAL = False
_CN.NEUCONW.FLOOR_LABELS = ['road']
_CN.NEUCONW.DEPTH_LOSS = False

# network config
_CN.NEUCONW.SDF_CONFIG = CN()
_CN.NEUCONW.SDF_CONFIG.d_in = 3
_CN.NEUCONW.SDF_CONFIG.d_out = 513
_CN.NEUCONW.SDF_CONFIG.d_hidden = 512
_CN.NEUCONW.SDF_CONFIG.n_layers = 8
_CN.NEUCONW.SDF_CONFIG.skip_in = (4,)
_CN.NEUCONW.SDF_CONFIG.multires = 6
_CN.NEUCONW.SDF_CONFIG.bias = 0.5
_CN.NEUCONW.SDF_CONFIG.scale = 1
_CN.NEUCONW.SDF_CONFIG.geometric_init = True
_CN.NEUCONW.SDF_CONFIG.weight_norm = True
_CN.NEUCONW.SDF_CONFIG.inside_outside = False
#nrw+nelo


_CN.NEUCONW.COLOR_CONFIG = CN()
_CN.NEUCONW.COLOR_CONFIG.d_in = 9
_CN.NEUCONW.COLOR_CONFIG.d_feature = 512
_CN.NEUCONW.COLOR_CONFIG.mode = "idr"
_CN.NEUCONW.COLOR_CONFIG.d_out = 3
_CN.NEUCONW.COLOR_CONFIG.d_hidden = 256 
_CN.NEUCONW.COLOR_CONFIG.n_layers = 4
_CN.NEUCONW.COLOR_CONFIG.head_channels = 128
_CN.NEUCONW.COLOR_CONFIG.static_head_layers = 2
_CN.NEUCONW.COLOR_CONFIG.weight_norm = True
_CN.NEUCONW.COLOR_CONFIG.multires_view = 4

_CN.NEUCONW.S_CONFIG = CN()
_CN.NEUCONW.S_CONFIG.init_val = 0.03

# loss config
_CN.NEUCONW.LOSS = CN()
_CN.NEUCONW.LOSS.coef = 1.0
_CN.NEUCONW.LOSS.igr_weight = 0.1
_CN.NEUCONW.LOSS.mask_weight = 0.1
_CN.NEUCONW.LOSS.depth_weight = 0.1
_CN.NEUCONW.LOSS.floor_weight = 0.01
_CN.NEUCONW.LOSS.curvature_weight = 0.0005


##############  Dataset  ##############
_CN.DATASET = CN()
_CN.DATASET.ROOT_DIR = None
_CN.DATASET.DATASET_NAME = None
_CN.DATASET.SPLIT = 'train'


_CN.DATASET.PHOTOTOURISM = CN()
_CN.DATASET.PHOTOTOURISM.IMG_DOWNSCALE = 1  # how much to downscale the images for phototourism dataset
_CN.DATASET.PHOTOTOURISM.USE_CACHE = True  # whether to use ray cache (make sure img_downscale is the same)
_CN.DATASET.PHOTOTOURISM.CACHE_DIR = 'cache'
_CN.DATASET.PHOTOTOURISM.CACHE_TYPE = 'npz'
_CN.DATASET.PHOTOTOURISM.SEMANTIC_MAP_PATH = 'semantic_maps'
_CN.DATASET.PHOTOTOURISM.WITH_SEMANTICS = True

############################
############################

# Create the configuration node
# Add NELO model configuration
_CN.NELO = CN()

# Model settings for NELO
_CN.NELO.model = CN()

# Object settings
_CN.NELO.model.object = CN()

# SDF settings for NELO
_CN.NELO.model.object.sdf = CN()

# Encoding settings for SDF
_CN.NELO.model.object.sdf.encoding = CN()
_CN.NELO.model.object.sdf.encoding.type = 'hashgrid'
_CN.NELO.model.object.sdf.encoding.levels = 16

#### warm_up_end
_CN.NELO.model.object.sdf.warm_up_end = 5000
# Hashgrid configuration
_CN.NELO.model.object.sdf.encoding.hashgrid = CN()
_CN.NELO.model.object.sdf.encoding.hashgrid.min_logres = 5
_CN.NELO.model.object.sdf.encoding.hashgrid.max_logres = 11
_CN.NELO.model.object.sdf.encoding.hashgrid.dict_size = 22
_CN.NELO.model.object.sdf.encoding.hashgrid.dim = 8
_CN.NELO.model.object.sdf.encoding.hashgrid.range = [-2, 2]

# coarse2fine
_CN.NELO.model.object.sdf.encoding.coarse2fine = CN()
_CN.NELO.model.object.sdf.encoding.coarse2fine.enabled = True
_CN.NELO.model.object.sdf.encoding.coarse2fine.init_active_level = 4
_CN.NELO.model.object.sdf.encoding.coarse2fine.step = 5000

# Gradient settings for NELO SDF
_CN.NELO.model.object.sdf.gradient = CN()
_CN.NELO.model.object.sdf.gradient.mode = 'numerical'
_CN.NELO.model.object.sdf.gradient.taps = 4



##############  Trainer  ##############
_CN.TRAINER = CN()
_CN.TRAINER.WORLD_SIZE = 1
_CN.TRAINER.CANONICAL_BS = 2048
_CN.TRAINER.CANONICAL_LR = 1e-3
_CN.TRAINER.SCALING = None  # this will be calculated automatically
_CN.TRAINER.SAVE_DIR = 'checkpoints'
_CN.TRAINER.VAL_FREQ = 0.125
_CN.TRAINER.SAVE_FREQ = 5000

# optimizer
_CN.TRAINER.OPTIMIZER = "adam"
_CN.TRAINER.LR = None  # this will be calculated automatically at runtime when train
_CN.TRAINER.WEIGHT_DECAY = 0.01  # adam weight decay 0, adamw weight decay 0.01

# step-based warm-up, , only applied when optimizer == 'adam'
_CN.TRAINER.WARMUP_EPOCHS = 0
_CN.TRAINER.WARMUP_MULTIPLIER = 1.0

# learning rate scheduler
_CN.TRAINER.LR_SCHEDULER = 'cosine'  # ['steplr', 'cosine', 'poly', 'none']
# for steplr
_CN.TRAINER.DECAY_STEP = []
_CN.TRAINER.DECAY_GAMMA = 0.1
# for poly
_CN.TRAINER.POLY_EXP = 0.9  # exponent for polynomial learning rate decay

# random seed
_CN.TRAINER.SEED = 66 #66


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()