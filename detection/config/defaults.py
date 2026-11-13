from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.MASK_ON = False

# ---------------------------------------------------------------------------- #
# Dataset options
# ---------------------------------------------------------------------------- #
_C.DATASETS = CN()
_C.DATASETS.TRAINS = ()
_C.DATASETS.TESTS = ()
_C.DATASETS.TARGETS = ()

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = "vgg16"

# -----------------------------------------------------------------------------
# INPUT options
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.TRANSFORMS_TRAIN = []
_C.INPUT.TRANSFORMS_TEST = []

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4

# ---------------------------------------------------------------------------- #
# RPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.RPN = CN()
_C.MODEL.RPN.ANCHOR_SIZES = (128, 256, 512)
_C.MODEL.RPN.ASPECT_RATIOS = (0.5, 1, 2)
_C.MODEL.RPN.ANCHOR_STRIDE = 16
_C.MODEL.RPN.NUM_CHANNELS = 512
_C.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
_C.MODEL.RPN.PRE_NMS_TOP_N_TRAIN = 12000
_C.MODEL.RPN.PRE_NMS_TOP_N_TEST = 6000
_C.MODEL.RPN.POST_NMS_TOP_N_TRAIN = 2000
_C.MODEL.RPN.POST_NMS_TOP_N_TEST = 300
_C.MODEL.RPN.NMS_THRESH = 0.7
_C.MODEL.RPN.MIN_SIZE = 1

# ---------------------------------------------------------------------------- #
# ROI HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_HEADS = CN()
_C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
_C.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
_C.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
_C.MODEL.ROI_HEADS.DETECTIONS_PER_IMG = 100

# ---------------------------------------------------------------------------- #
# Box Head
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_BOX_HEAD = CN()
_C.MODEL.ROI_BOX_HEAD.NUM_CLASSES = 9
_C.MODEL.ROI_BOX_HEAD.POOL_SPATIAL_SCALE = 1.0 / 16
_C.MODEL.ROI_BOX_HEAD.POOL_RESOLUTION = 7
_C.MODEL.ROI_BOX_HEAD.POOL_TYPE = "pooling"
_C.MODEL.ROI_BOX_HEAD.BOX_PREDICTOR = 'vgg16_predictor'

# ---------------------------------------------------------------------------- #
# Adversarial options
# ---------------------------------------------------------------------------- #
_C.ADV = CN()
_C.ADV.LAMBDA_FROM = 1.0
_C.ADV.LAMBDA_TO = 1.0

_C.ADV.LAYERS = [False, False, True]
_C.ADV.DIS_MODEL = [{'in_channels': 512}]
_C.ADV.LOSS_FUNC = 'focal_loss'  # ['cross_entropy', 'focal_loss']
_C.ADV.FOCAL_LOSS_GAMMA = 5
_C.ADV.LOSS_WEIGHT = 1.0

_C.ADV.INSTANCE_SOURCE_SPACE_LIST = [True, True]
_C.ADV.INSTANCE_TARGET_SPACE_LIST = [True, True]


# ---------------------------------------------------------------------------- #
# ROI_ONE_DIS options
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_ONE_DIS = CN()
_C.MODEL.ROI_ONE_DIS.DOM = CN()
_C.MODEL.ROI_ONE_DIS.DOM.LOSS_WEIGHT = 1.0
_C.MODEL.ROI_ONE_DIS.DOM.MON = False
_C.MODEL.ROI_ONE_DIS.DOM.DISCRIMINATOR = 'OneFCOSDiscriminator'
_C.MODEL.ROI_ONE_DIS.DOM.NUM_CONVS = 4
_C.MODEL.ROI_ONE_DIS.DOM.GRL_LAMBDA = 0.01

_C.MODEL.ROI_ONE_DIS.CLS = CN()
_C.MODEL.ROI_ONE_DIS.CLS.LOSS_WEIGHT = 1.0
_C.MODEL.ROI_ONE_DIS.CLS.MON = False
_C.MODEL.ROI_ONE_DIS.CLS.DISCRIMINATOR = 'OneFCOSDiscriminator_cc'
_C.MODEL.ROI_ONE_DIS.CLS.NUM_CONVS = 4
_C.MODEL.ROI_ONE_DIS.CLS.GRL_LAMBDA = 0.01
_C.MODEL.ROI_ONE_DIS.CLS.LOSS_DIRECT_W = 1.0
_C.MODEL.ROI_ONE_DIS.CLS.LOSS_GRL_W = 1.0
_C.MODEL.ROI_ONE_DIS.CLS.SAMPLES_THRESH = 0.5
_C.MODEL.ROI_ONE_DIS.CLS.NUM_CLASSES = 1

# ---------------------------------------------------------------------------- #
# ROI_TWO_DIS options
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_TWO_DIS = CN()
_C.MODEL.ROI_TWO_DIS.DOM = CN()
_C.MODEL.ROI_TWO_DIS.DOM.LOSS_WEIGHT = 1.0
_C.MODEL.ROI_TWO_DIS.DOM.MON = False
_C.MODEL.ROI_TWO_DIS.DOM.DISCRIMINATOR = 'VGG16TwoDiscriminator'
_C.MODEL.ROI_TWO_DIS.DOM.POOL_RESOLUTION = 7
_C.MODEL.ROI_TWO_DIS.DOM.GRL_LAMBDA = 0.01

_C.MODEL.ROI_TWO_DIS.CLS = CN()
_C.MODEL.ROI_TWO_DIS.CLS.LOSS_WEIGHT = 1.0
_C.MODEL.ROI_TWO_DIS.CLS.MON = False
_C.MODEL.ROI_TWO_DIS.CLS.DISCRIMINATOR = 'VGG16TwoDiscriminator_cc'
_C.MODEL.ROI_TWO_DIS.CLS.POOL_RESOLUTION = 7
_C.MODEL.ROI_TWO_DIS.CLS.GRL_LAMBDA = 0.01
_C.MODEL.ROI_TWO_DIS.CLS.LOSS_DIRECT_W = 1.0
_C.MODEL.ROI_TWO_DIS.CLS.LOSS_GRL_W = 1.0
_C.MODEL.ROI_TWO_DIS.CLS.SAMPLES_THRESH = 0.5
_C.MODEL.ROI_TWO_DIS.CLS.NUM_CLASSES = 1

# ---------------------------------------------------------------------------- #
# GENFEATURE Options
# ---------------------------------------------------------------------------- #
_C.MODEL.GENFEATURE = CN()
_C.MODEL.GENFEATURE.TWOMULTSCALE = True
_C.MODEL.GENFEATURE.LOCAL_GLOBAL_MERGE = True
_C.MODEL.GENFEATURE.FPN_STRIDES = [1.0/16]

# ---------------------------------------------------------------------------- #
# Network and ADV methods
# ---------------------------------------------------------------------------- #
_C.MODEL.MODE = CN()
_C.MODEL.MODE.USE_STUDENT = False
_C.MODEL.MODE.INSRANCE_ADV = False
_C.MODEL.MODE.UPDATE_TEACHER_ITERUM = 100
_C.MODEL.MODE.NET_MOMENTUM = 0.99
_C.MODEL.MODE.INIT_DIS_NET = 100

_C.MODEL.MODE.UPDATE_TEACHER_MARK = False
_C.MODEL.MODE.UPDATE_STUDENT_MARK = False
_C.MODEL.MODE.UPDATE_STUDENT_ITERUM = 20
_C.MODEL.MODE.UPDATE_TEACHER_LR = [0.99, 0.92, 0.99, 0.99, 0.99]
_C.MODEL.MODE.UPDATE_TEACHER_INTER = [1, 100, 1, 100, 100]

_C.MODEL.MODE.UPDATE_TP_MARK = False
_C.MODEL.MODE.UPDATE_TP_INTERNAL = 100

_C.MODEL.MODE.UPDATE_TP_MATRIX_MARK = False
_C.MODEL.MODE.UPDATE_TP_MATRIX_INTERNAL = 100

_C.MODEL.MODE.TRAIN_PROCESS = 'S0'
_C.MODEL.MODE.TEST_PROCESS = 'ST'
_C.MODEL.MODE.PJOTYPE = 'NBIAS'

_C.MODEL.MODE.IS_SAMPLE = 0
_C.MODEL.MODE.IS_SAMPLE_TEST = 0

_C.MODEL.MODE.USE_MOE_TEST = False
_C.MODEL.MODE.USE_MOE_FUSION_TEST = False


# ---------------------------------------------------------------------------- #
# Optional Selective-AHDR experiment (disabled for every legacy config)
# ---------------------------------------------------------------------------- #
_C.SELECTIVE_AHDR = CN()
_C.SELECTIVE_AHDR.ENABLED = False
_C.SELECTIVE_AHDR.ALPHA_MAX = 1.2
_C.SELECTIVE_AHDR.GATE_INIT = 0.5
_C.SELECTIVE_AHDR.HIDDEN_CHANNELS = 8
_C.SELECTIVE_AHDR.GATE_LR = 1e-5
_C.SELECTIVE_AHDR.CONS_WEIGHT = 0.01
_C.SELECTIVE_AHDR.OBJ_CONS_WEIGHT = 0.01
_C.SELECTIVE_AHDR.GATE_REG_WEIGHT = 0.001
_C.SELECTIVE_AHDR.LOG_INTERVAL = 50

# ---------------------------------------------------------------------------- #
# B2 source-only object-aware Gate supervision (disabled by default)
# ---------------------------------------------------------------------------- #
_C.B2_OBJECT_AWARE = CN()
_C.B2_OBJECT_AWARE.ENABLED = False
_C.B2_OBJECT_AWARE.OBJECT_FLOOR = 0.500
_C.B2_OBJECT_AWARE.SMALL_OBJECT_FLOOR = 0.525
_C.B2_OBJECT_AWARE.SMALL_AREA_RATIO = 0.020
_C.B2_OBJECT_AWARE.SMALL_WEIGHT_BETA = 1.0
_C.B2_OBJECT_AWARE.BOX_EXPAND_RATIO = 1.15
_C.B2_OBJECT_AWARE.OBJ_WEIGHT = 0.05
_C.B2_OBJECT_AWARE.SMALL_WEIGHT = 0.10
_C.B2_OBJECT_AWARE.BG_WEIGHT = 0.02
_C.B2_OBJECT_AWARE.TV_WEIGHT = 0.001
_C.B2_OBJECT_AWARE.ANCHOR_WEIGHT = 0.01
_C.B2_OBJECT_AWARE.LOG_INTERVAL = 50

# ---------------------------------------------------------------------------- #
# Solver options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.EPOCHS = 25
_C.SOLVER.STEPS = (16, 22)
_C.SOLVER.STEPS_EVAL = (16, 22)
_C.SOLVER.LR = 1e-5
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.MAX_ITERS_EARLY_STOP = 0
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.BATCH_SIZE = 1
_C.SOLVER.WARMUP = False
_C.SOLVER.WARMUP_ITERS = 500

# ---------------------------------------------------------------------------- #
# Test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.EVAL_TYPES = ('voc',)

_C.WORK_DIR = "./work_dir"

# C1 source-only appearance diversification (disabled for all legacy configs).
_C.C1_SOURCE_DIVERSIFICATION = CN()
_C.C1_SOURCE_DIVERSIFICATION.ENABLED = False
_C.C1_SOURCE_DIVERSIFICATION.APPLY_PROB = 0.8
_C.C1_SOURCE_DIVERSIFICATION.LOG_INTERVAL = 50
_C.C1_SOURCE_DIVERSIFICATION.PHOTOMETRIC = CN()
_C.C1_SOURCE_DIVERSIFICATION.PHOTOMETRIC.ENABLED = True
_C.C1_SOURCE_DIVERSIFICATION.PHOTOMETRIC.BRIGHTNESS = (0.72, 1.18)
_C.C1_SOURCE_DIVERSIFICATION.PHOTOMETRIC.CONTRAST = (0.72, 1.18)
_C.C1_SOURCE_DIVERSIFICATION.PHOTOMETRIC.GAMMA = (0.78, 1.28)
_C.C1_SOURCE_DIVERSIFICATION.PHOTOMETRIC.SATURATION = (0.75, 1.20)
_C.C1_SOURCE_DIVERSIFICATION.DEGRADATION = CN()
_C.C1_SOURCE_DIVERSIFICATION.DEGRADATION.ENABLED = True
_C.C1_SOURCE_DIVERSIFICATION.DEGRADATION.BLUR_SIGMA = (0.0, 1.4)
_C.C1_SOURCE_DIVERSIFICATION.DEGRADATION.BLUR_SIGMA_MAX = 1.0
_C.C1_SOURCE_DIVERSIFICATION.DEGRADATION.NOISE_STD = (0.0, 0.035)
_C.C1_SOURCE_DIVERSIFICATION.DEGRADATION.ATTENUATION = (0.0, 0.22)
_C.C1_SOURCE_DIVERSIFICATION.DETAIL = CN()
_C.C1_SOURCE_DIVERSIFICATION.DETAIL.ENABLED = True
_C.C1_SOURCE_DIVERSIFICATION.DETAIL.K = (0.65, 1.25)
_C.C1_SOURCE_DIVERSIFICATION.DETAIL.SIGMA = 1.0
_C.C1_SOURCE_DIVERSIFICATION.LOCAL = CN()
_C.C1_SOURCE_DIVERSIFICATION.LOCAL.ENABLED = True
_C.C1_SOURCE_DIVERSIFICATION.LOCAL.GRID_H = 3
_C.C1_SOURCE_DIVERSIFICATION.LOCAL.GRID_W = 5
_C.C1_SOURCE_DIVERSIFICATION.LOCAL.STRENGTH = (0.08, 0.22)
_C.C1_SOURCE_DIVERSIFICATION.LOCAL.SMOOTH_SIGMA = 5.0
_C.C1_SOURCE_DIVERSIFICATION.SOURCE_ADAPTIVE = CN()
_C.C1_SOURCE_DIVERSIFICATION.SOURCE_ADAPTIVE.ENABLED = True
_C.C1_SOURCE_DIVERSIFICATION.SOURCE_ADAPTIVE.DARK_REFERENCE = 0.35
_C.C1_SOURCE_DIVERSIFICATION.SOURCE_ADAPTIVE.CONTRAST_REFERENCE = 0.18
_C.C1_SOURCE_DIVERSIFICATION.SOURCE_ADAPTIVE.GRADIENT_REFERENCE = 0.08
_C.C1_SOURCE_DIVERSIFICATION.OBJECT_PROTECTION = CN()
_C.C1_SOURCE_DIVERSIFICATION.OBJECT_PROTECTION.ENABLED = True
_C.C1_SOURCE_DIVERSIFICATION.OBJECT_PROTECTION.LAMBDA_OBJECT = 0.35
_C.C1_SOURCE_DIVERSIFICATION.OBJECT_PROTECTION.LAMBDA_SMALL = 0.25
_C.C1_SOURCE_DIVERSIFICATION.OBJECT_PROTECTION.BOX_EXPAND_RATIO = 1.10
_C.C1_SOURCE_DIVERSIFICATION.OBJECT_PROTECTION.SMALL_AREA_RATIO = 0.02
