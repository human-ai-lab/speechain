"""
    Author: Heli Qi
    Affiliation: NAIST
    Date: 2022.11
"""
import os
import sys

# Add SPEECHAIN_ROOT and the data folder to the path so that the speechain
# package and the sibling dataset scripts can be imported even when the toolkit
# is not installed into the current environment
data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
speechain_root = os.path.dirname(data_dir)
for _path in (speechain_root, data_dir):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from librispeech.meta_post_processor import LibriSpeechMetaPostProcessor

if __name__ == '__main__':
    LibriSpeechMetaPostProcessor().main()
