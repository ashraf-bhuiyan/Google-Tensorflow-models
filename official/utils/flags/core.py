# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from absl import flags

from official.utils.flags import _base
from official.utils.flags import _misc
from official.utils.flags import _performance


def define_base(*args, **kwargs):
  key_flags = _base.define_base(*args, **kwargs)
  [flags.declare_key_flag(fl) for fl in key_flags]


def define_image(*args, **kwargs):
  key_flags = _misc.define_image(*args, **kwargs)
  [flags.declare_key_flag(fl) for fl in key_flags]


def define_performance(*args, **kwargs):
  key_flags = _performance.define_performance(*args, **kwargs)
  [flags.declare_key_flag(fl) for fl in key_flags]

get_tf_dtype = _performance.get_tf_dtype
get_loss_scale = _performance.get_loss_scale
