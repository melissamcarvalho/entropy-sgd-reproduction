from enum import Enum, IntEnum
from typing import Dict, NamedTuple, Tuple


# TODO: Remove it when we eliminate typing requirements
# with ExperimentBaseModel
class DatasetType(Enum):
    CIFAR10 = (1, (3, 32, 32), 10)
    SVHN = (2, (3, 32, 32), 10)

    def __init__(self,
                 id: int,
                 image_shape: Tuple[int, int, int],
                 num_classes: int):
        self.D = image_shape
        self.K = num_classes


class DatasetSubsetType(IntEnum):
    TRAIN = 0
    TEST = 1


class ComplexityType(Enum):
    # Measures from Fantastic Generalization Measures (equation numbers)
    PARAMS = 20
    INVERSE_MARGIN = 22
    LOG_SPEC_INIT_MAIN = 29
    LOG_SPEC_ORIG_MAIN = 30
    LOG_PROD_OF_SPEC_OVER_MARGIN = 31
    LOG_PROD_OF_SPEC = 32
    FRO_OVER_SPEC = 33
    LOG_SUM_OF_SPEC_OVER_MARGIN = 34
    LOG_SUM_OF_SPEC = 35
    LOG_PROD_OF_FRO_OVER_MARGIN = 36
    LOG_PROD_OF_FRO = 37
    LOG_SUM_OF_FRO_OVER_MARGIN = 38
    LOG_SUM_OF_FRO = 39
    FRO_DIST = 40
    DIST_SPEC_INIT = 41
    PARAM_NORM = 42
    PATH_NORM_OVER_MARGIN = 43
    PATH_NORM = 44
    PACBAYES_INIT = 48
    PACBAYES_ORIG = 49
    PACBAYES_FLATNESS = 53
    PACBAYES_MAG_INIT = 56
    PACBAYES_MAG_ORIG = 57
    PACBAYES_MAG_FLATNESS = 61
    # Other Measures
    L2 = 100
    L2_DIST = 101
    # FFT Spectral Measures
    LOG_SPEC_INIT_MAIN_FFT = 129
    LOG_SPEC_ORIG_MAIN_FFT = 130
    LOG_PROD_OF_SPEC_OVER_MARGIN_FFT = 131
    LOG_PROD_OF_SPEC_FFT = 132
    FRO_OVER_SPEC_FFT = 133
    LOG_SUM_OF_SPEC_OVER_MARGIN_FFT = 134
    LOG_SUM_OF_SPEC_FFT = 135
    DIST_SPEC_INIT_FFT = 141


class EvaluationMetrics(NamedTuple):
    acc: float
    avg_loss: float
    num_to_evaluate_on: int
    all_complexities: Dict[ComplexityType, float]
