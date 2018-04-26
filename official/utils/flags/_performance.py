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


def define_performance(num_parallel_calls=True, inter_op=True):
  key_flags = []
  if num_parallel_calls:
    absl.flags.DEFINE_integer(
        name="num_parallel_calls", short_name="npc",default=5,
        help=help_wrap("The number of records that are  processed in parallel "
                       "during input processing. This can be optimized per "
                       "data set but for generally homogeneous data sets, "
                       "should be approximately the number of available CPU "
                       "cores."))
    key_flags.append("num_parallel_calls")

  if inter_op:
    absl.flags.DEFINE_integer(
        name="inter_op_parallelism_threads", short_name="inter", default=0,
        help=help_wrap("Number of inter_op_parallelism_threads to use for CPU. "
                       "See TensorFlow config.proto for details.")
    )

  return key_flags
