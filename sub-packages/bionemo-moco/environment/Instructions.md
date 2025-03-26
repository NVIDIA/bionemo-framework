Environment Setup
===============

from the bionemo-moco directory run

 bash environment/setup.sh

This creates the conda environment, installs bionemo-moco and runs the tests.

Local Code Setup
===============
from the bionemo-moco directory run

 bash environment/clone_bionemo_moco.sh

This creates clones only the bionemo subpackage. To install in your local env use pip install -e . inside the bionemo-moco directory.

pip install --no-deps -e . can be used if want to install bionemo-moco over your current torch version. The remaining required jaxtyping and pot dependencies can be manually installed via pip.
