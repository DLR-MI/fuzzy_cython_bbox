import numpy as np
import pandas as pd
from pathlib import Path
from fuzzy_cython_bbox import disambiguate_ious
pd.set_option('display.precision', 3)

this_dir = Path(__file__).parent

thresh_iou = 0.8
thresh_std = 0.5

ious = np.load(this_dir / 'ious.npy')
ious_std = np.load(this_dir / 'ious_std.npy')
reference = 1 - np.load(this_dir / 'pcost.npy')
scores = np.random.uniform(0, 1, size=ious.shape)

ious_, _, was_ambiguous = disambiguate_ious(
    ious, ious_std, reference, scores, thresh_iou, thresh_std
)
print('Original iou matrix')
print(pd.DataFrame(ious))
print('Disambiguated iou matrix')
print(pd.DataFrame(ious_))
print(f'Was an ambiguous ious matrix? {was_ambiguous}')
print('Notice the permutation of the entries (2,6) and (2,8)')