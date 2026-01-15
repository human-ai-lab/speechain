# usage: bash create_env.sh 

# 1. Create the speechain environment if there is no environment named speechain

# Set the python version, if use other than 3.9, modify environment.yaml
PYTHON_VERSION=3.9
echo "Installing speechain env using Python version: $PYTHON_VERSION"

speechain_envir=$(conda env list | grep speechain)
if [ -z "${speechain_envir}" ]; then
  conda env create -f environment.yaml
  envir_path=`conda env list | grep "speechain .*" | awk '{print $2}'`
  echo "Created speechain environment at: $envir_path"
fi

# 2. Get the environment root path by conda
#read -ra speechain_envir <<< "${speechain_envir}"
#envir_path="${speechain_envir[$((${#speechain_envir[*]}-1))]}"

# 3. Get the python compiler path in the environment root
pycompiler_path="${envir_path}/bin/python${PYTHON_VERSION}"

# add the python compiler path to the environmental variables
if ! grep -q "export SPEECHAIN_PYTHON" ~/.bashrc; then
  echo "export SPEECHAIN_PYTHON=${pycompiler_path}" >> ~/.bashrc
fi
export SPEECHAIN_PYTHON=${pycompiler_path}

# 4. Add the current path to the environmental variables as the toolkit root path
if ! grep -q "export SPEECHAIN_ROOT" ~/.bashrc; then
  echo "export SPEECHAIN_ROOT=${PWD}" >> ~/.bashrc
fi
export SPEECHAIN_ROOT=${PWD}

# 5. Install the local development packages to conda environment
conda activate speechain
${SPEECHAIN_PYTHON} -m pip install -e "${SPEECHAIN_ROOT}"
