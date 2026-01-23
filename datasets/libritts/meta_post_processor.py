"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.11
"""
import os
import sys

# Add parent directory and SPEECHAIN_ROOT to path to enable imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
speechain_root = os.path.dirname(parent_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if speechain_root not in sys.path:
    sys.path.insert(0, speechain_root)

from librispeech.meta_post_processor import LibriSpeechMetaPostProcessor

if __name__ == '__main__':
    LibriSpeechMetaPostProcessor().main()
