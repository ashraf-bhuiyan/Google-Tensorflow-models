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

from official.utils.flags._conventions import to_choices_str
from official.utils.flags._conventions import help_wrap


def define_image(data_format=True):
  key_flags = []

  if data_format:
    choices = ["channels_first", "channels_last"]
    flags.DEFINE_string(
        name="data_format", short_name="df", default=None,
        help=help_wrap(
            "A flag to override the data format used in the model. "
            "channels_first provides a performance boost on GPU but is not "
            "always compatible with CPU. If left unspecified, the data format "
            "will be chosen automatically based on whether TensorFlow was "
            "built for CPU or GPU.\n{}".format(to_choices_str(choices))))
    key_flags.append("data_format")

    @flags.validator("data_format")
    def _check_data_format(data_format):
      return data_format in choices or data_format is None

  return key_flags


def define_export(export_dir=True):
  key_flags = []

  if export_dir:
    flags.DEFINE_string(
        name="export_dir", short_name="ed", default=None,
        help=help_wrap("If set, a SavedModel serialization of the model will "
                       "be exported to this directory at the end of training. "
                       "See the README for more details and relevant links.")
    )
    key_flags.append("export_dir")

  return key_flags
