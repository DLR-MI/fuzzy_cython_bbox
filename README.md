# fuzzy_cython_bbox

cython_bbox is widely used in object detection tasks. It was presumably first implemented in [Faster-RCNN](https://github.com/rbgirshick/py-faster-rcnn). Since then, almost all object detection projects use the source code directly.

In order to use it in standalone code snippets or small projects, [Samson Wang](https://github.com/samson-wang/cython_bbox/tree/master) made it a pypi module. The code was totally borrowed from [Faster-RCNN](https://github.com/rbgirshick/py-faster-rcnn). Thanks [RBG](http://www.rossgirshick.info/) and Samson!

In this repo, the code is extended to compute bbox overlaps together with an estimate of the corresponding uncertainties, provided that the spreads of the bboxes are given.

## Installation

```
pip install git+https://github.com/DLR-MI/fuzzy_cython_bbox.git
```

## Usage


```
from fuzzy_cython_bbox import fuzzy_bbox_ious

ious, ious_std = fuzzy_bbox_ious(
    np.ascontiguousarray(atlbrs, dtype=float),
    np.ascontiguousarray(btlbrs, dtype=float),
    np.ascontiguousarray(axywh_stds, dtype=float),
    np.ascontiguousarray(bxywh_stds, dtype=float)
)

```

For an example of how an `ious` matrix can be disambiguated run

```
python test/disambiguate.py
```
