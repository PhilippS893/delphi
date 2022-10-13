import os
import inspect

package_dir = os.path.abspath(os.path.join(inspect.getabsfile(inspect.currentframe()), os.pardir))
mni_template = os.path.join(package_dir, "MNI152_T1_2mm_brain_mask.nii.gz")
