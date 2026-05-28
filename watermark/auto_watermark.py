# Copyright 2024 THU-BPM MarkLLM.
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

# =========================================================================
# AutoWatermark.py
# =========================================================================

import torch
import importlib
from typing import List
from watermark.auto_config import AutoConfig


# ============================================================
# ALGORITHM → CLASS PATH MAPPING
# ============================================================
WATERMARK_MAPPING_NAMES = {
    'KGW': 'watermark.kgw.KGW',
    'Unigram': 'watermark.unigram.Unigram',
    'SWEET': 'watermark.sweet.SWEET',
    'UPV': 'watermark.upv.UPV',

    # 🔥 NEW: SIR + XSIR supported
    'SIR': 'watermark.sir.SIR',
    'XSIR': 'watermark.xsir.XSIR',

    'Unbiased': 'watermark.unbiased.UnbiasedWatermark',
    'DIP': 'watermark.dip.DIP',
    'EWD': 'watermark.ewd.EWD',
    'EXP': 'watermark.exp.EXP',
    'EXPGumbel': 'watermark.exp_gumbel.EXPGumbel',
    'EXPEdit': 'watermark.exp_edit.EXPEdit',
    'ITSEdit': 'watermark.its_edit.ITSEdit',
    'SynthID': 'watermark.synthid.SynthID',
    'TS': 'watermark.ts.TS',
    'SWEETBLACK': 'watermark.sweetblack.SWEETBLACK',
}


def watermark_name_from_alg_name(name):
    """Get the watermark class name from the algorithm name."""
    if name in WATERMARK_MAPPING_NAMES:
        return WATERMARK_MAPPING_NAMES[name]
    else:
        raise ValueError(f"Invalid algorithm name: {name}")


# ============================================================
# MAIN AUTO-WATERMARK FACTORY
# ============================================================
class AutoWatermark:
    """
        Generic watermark factory.
        Only instantiated via AutoWatermark.load()
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoWatermark must be created using AutoWatermark.load(algorithm_name, algorithm_config, transformers_config)"
        )

    @staticmethod
    def load(algorithm_name, algorithm_config=None, transformers_config=None, *args, **kwargs):
        """
        Load watermark algorithm by name.
        Ensures SIR / XSIR receive BOTH (config, transformers_config)
        """

        # locate class
        watermark_name = watermark_name_from_alg_name(algorithm_name)
        module_name, class_name = watermark_name.rsplit('.', 1)
        module = importlib.import_module(module_name)
        watermark_class = getattr(module, class_name)

        # load config
        watermark_config = AutoConfig.load(
            algorithm_name,
            transformers_config,
            algorithm_config_path=algorithm_config,
            **kwargs
        )

        # =====================================================
        # IMPORTANT FIX:
        # SIR / XSIR REQUIRE (algorithm_config, transformers_config)
        # others accept (watermark_config) only.
        # We detect constructor signature to pass proper args.
        # =====================================================
        try:
            instance = watermark_class(watermark_config, transformers_config)
        except TypeError:
            # fallback for older algorithms (KGW, SWEET, Unigram...)
            instance = watermark_class(watermark_config)

        return instance



# ============================================================
# VLLM SUPPORT
# ============================================================
vllm_supported_methods = ["UPV", "KGW", "Unigram"]


class AutoWatermarkForVLLM:
    def __init__(self, algorithm_name, algorithm_config, transformers_config):
        if algorithm_name not in vllm_supported_methods:
            raise NotImplementedError(
                f"vLLM integration currently supports {vllm_supported_methods}, but got {algorithm_name}"
            )

        self.watermark = AutoWatermark.load(
            algorithm_name=algorithm_name,
            algorithm_config=algorithm_config,
            transformers_config=transformers_config
        )

    def __call__(self, input_ids: List[int], scores: torch.FloatTensor) -> torch.Tensor:

        if len(input_ids) == 0:
            return scores

        input_ids = torch.LongTensor(input_ids).to(self.watermark.config.device)[None, :]
        scores = scores[None, :]

        assert len(input_ids.shape) == 2
        assert len(scores.shape) == 2

        scores = self.watermark.logits_processor(input_ids, scores)
        return scores[0, :]

    def get_data_for_visualization(self, text):
        return self.watermark.get_data_for_visualization(text)

    def detect_watermark(self, text):
        if isinstance(text, list):
            return [self.watermark.detect_watermark(_) for _ in text]
        return self.watermark.detect_watermark(text)
