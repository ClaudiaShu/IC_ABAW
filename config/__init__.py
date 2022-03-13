# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------


from .defaults import _C as config
from .defaults_imagenet import _C as config_imagenet
from .defaults import update_config, merge_configs

from .default_aflw import _C as config_aflw
from .default_cofw import _C as config_cofw
from .default_wflw import _C as config_wflw
from .default_loss import _C as cfg_loss