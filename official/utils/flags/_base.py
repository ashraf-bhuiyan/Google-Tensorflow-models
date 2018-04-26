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

import absl.flags

from official.utils.flags._conventions import help_wrap


def define_base(data_dir=True, model_dir=True, train_epochs=True):
  key_flags = []

  if data_dir:
    absl.flags.DEFINE_string(name="data_dir", short_name="dd", default="/tmp",
                             help=help_wrap("The location of the input data."))
    key_flags.append("data_dir")

  if model_dir:
    absl.flags.DEFINE_string(name="model_dir", short_name="md", default="/tmp",
                             help=help_wrap("The location of the model "
                                            "checkpoint files."))
    key_flags.append("model_dir")

  if train_epochs:
    absl.flags.DEFINE_integer(name="train_epochs", short_name="te", default=1,
                              help=help_wrap("The number of epochs used to "
                                             "train."))
    key_flags.append("train_epochs")

  return key_flags