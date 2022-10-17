import os
import inspect

from . import utils
from . import networks
from . import explain

package_dir = os.path.abspath(os.path.join(inspect.getabsfile(inspect.currentframe()), os.pardir))
mni_template = os.path.join(package_dir, "MNI152_T1_2mm_brain_mask.nii.gz")

__all__ = [
    'utils',
    'networks',
    'explain',
]