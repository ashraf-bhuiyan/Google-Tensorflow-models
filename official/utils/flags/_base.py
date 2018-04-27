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

from official.utils.flags._conventions import help_wrap
from official.utils.logs import hooks_helper


DEFAULTS = {
  "data_dir": "/tmp",
  "model_dir": "/tmp",
  "train_epochs": 1,
  "epochs_between_evals": 1,
  "stop_threshold": None,
  "batch_size": 32,
  "hooks": "LoggingTensorHook",
}


def define_base(data_dir=True, model_dir=True, train_epochs=True,
    epochs_between_evals=True, stop_threshold=True, batch_size=True,
    multi_gpu=True, hooks=True):
  key_flags = []

  if data_dir:
    flags.DEFINE_string(
        name="data_dir", short_name="dd", default=DEFAULTS["data_dir"],
        help=help_wrap("The location of the input data."))
    key_flags.append("data_dir")

  if model_dir:
    flags.DEFINE_string(
        name="model_dir", short_name="md", default=DEFAULTS["model_dir"],
        help=help_wrap("The location of the model checkpoint files."))
    key_flags.append("model_dir")

  if train_epochs:
    flags.DEFINE_integer(
        name="train_epochs", short_name="te", default=DEFAULTS["train_epochs"],
        help=help_wrap("The number of epochs used to train."))
    key_flags.append("train_epochs")

  if epochs_between_evals:
    flags.DEFINE_integer(
        name="epochs_between_evals", short_name="ebe",
        default=DEFAULTS["epochs_between_evals"],
        help=help_wrap("The number of training epochs to run between "
                       "evaluations."))
    key_flags.append("epochs_between_evals")

  if stop_threshold:
    flags.DEFINE_float(
        name="stop_threshold", short_name="st",
        default=DEFAULTS["stop_threshold"],
        help=help_wrap("If passed, training will stop at the earlier of "
                       "train_epochs and when the evaluation metric is  "
                       "greater than or equal to stop_threshold."))

  if batch_size:
    flags.DEFINE_integer(
        name="batch_size", short_name="bs", default=DEFAULTS["batch_size"],
        help=help_wrap("Batch size for training and evaluation."))
    key_flags.append("batch_size")

  if multi_gpu:
    flags.DEFINE_bool(
        name="multi_gpu", default=False,
        help=help_wrap("If set, run across all available GPUs."))
    key_flags.append("multi_gpu")

  if hooks:
    # Construct a pretty summary of hooks.
    pad_len = max([len(i) for i in hooks_helper.HOOKS_ALIAS.values()]) + 6
    hook_list_str = (
        "\u180E  {}Abbreviation\n".format("Hook".ljust(pad_len)) + "\n".join(
        ["\u180E    {}({})".format(value.ljust(pad_len), key) for key, value
         in hooks_helper.HOOKS_ALIAS.items()]))
    flags.DEFINE_list(
        name="hooks", short_name="hk", default=DEFAULTS["hooks"],
        help=help_wrap(
            "A comma separated list of (case insensitive) strings to specify "
            "the names of training hooks.\n{}\n\u180E  "
            "Example: `--hooks ProfilerHook,ExamplesPerSecondHook`\n\u180E  "
            "(or)     `-hk p,eps`\nSee official.utils.logs.hooks_helper for "
            "details.".format(hook_list_str))
    )
    key_flags.append("hooks")

  return key_flags
