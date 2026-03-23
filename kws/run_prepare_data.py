#!/usr/bin/env python3
"""
Step 1: Prepare all training data.
Downloads Google Speech Commands and generates custom keyword placeholders.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from config import DataConfig, AudioConfig
from data.download import prepare_all_data

if __name__ == "__main__":
    data_cfg = DataConfig()
    audio_cfg = AudioConfig()
    prepare_all_data(
        data_dir=os.path.dirname(data_cfg.data_dir),
        custom_dir=data_cfg.custom_data_dir,
        sample_rate=audio_cfg.sample_rate,
    )
