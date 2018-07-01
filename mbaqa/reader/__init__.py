#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from ..tokenizers import LtpTokenizer
from .. import DATA_DIR


DEFAULTS = {
    'tokenizer': LtpTokenizer,
    'model': os.path.join(DATA_DIR, 'reader/20180308-d004debe.mdl'),
}


def set_default(key, value):
    global DEFAULTS
    DEFAULTS[key] = value


from .model import DocReader
from .predictor import Predictor
from . import config
from . import vector
from . import data
from . import utils
