from setuptools import find_packages, setup

setup(name="speechain",
      version="0.1",
      description="The main folder of the SpeeChain toolkit.",
      author="Heli Qi (original codes are from Andros Tjandra & Sashi Novitasari)",
      author_email='qi.heli.qi9@is.naist.jp',
      platforms=["any"],  # or more specific, e.g. "win32", "cygwin", "osx"
      license="BSD",
      url="",
      # Note: 'speechain.*' is necessary so that all the sub-packages are installed;
      # the top-level 'datasets' folder is NOT a package anymore (its shared code has
      # been moved to 'speechain.datasets'), which also avoids shadowing the
      # HuggingFace 'datasets' package after installation.
      packages=find_packages(include=['speechain', 'speechain.*']),
      install_requires=['numpy',
                        'scipy',
                        'torch',
                        'torchvision',
                        'torchaudio',
                        'pytest',
                        'tabulate',
                        'tqdm',
                        'pathos',
                        'tensorboardX',
                        'tensorboard',
                        'pandas',
                        'tables',
                        'python-speech-features',
                        'soundfile',
                        'psutil',
                        'pyyaml',
                        'ruamel.yaml',
                        'sentencepiece',
                        'g2p-en',
                        'editdistance',
                        'edit-distance',
                        'h5py',
                        'huggingface-hub',
                        'matplotlib',
                        'seaborn',
                        'packaging',
                        'pyworld',
                        'GPUtil'])

