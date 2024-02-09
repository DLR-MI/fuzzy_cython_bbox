# --------------------------------------------------------
# Licensed under The MIT License [see LICENSE for details]
# Written by Edgardo Solano-Carrillo
# Reusing bbox_overlaps code by Sergey Karayev, 
# which was packaged in cython_bbox by WANG Chenxi
# --------------------------------------------------------

import numpy as np
cimport numpy as np

DTYPE = float
ctypedef np.float_t DTYPE_t
ctypedef np.npy_bool BOOL_t
ctypedef np.long_t INT_t

def fuzzy_bbox_ious(
    np.ndarray[DTYPE_t, ndim=2] boxes,
    np.ndarray[DTYPE_t, ndim=2] query_boxes,
    np.ndarray[DTYPE_t, ndim=2] std_boxes,
    np.ndarray[DTYPE_t, ndim=2] std_query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    std_boxes: (N, 4) ndarray of float (xywh format)
    std_query_boxes: (K, 4) ndarray of float (xywh format)
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    std_overlaps: (N, K) ndarray of corresponding overlap errors 
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] overlaps = np.zeros((N, K), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] std_overlaps = np.zeros((N, K), dtype=DTYPE)
    cdef DTYPE_t iw, ih, box_area
    cdef DTYPE_t ua
    cdef unsigned int k, n

    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = float(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua

                    # Corresponding spreads
                    d_ia  = iw * (abs(std_boxes[n, 1] - std_query_boxes[k, 1]) + 0.5 * (std_boxes[n, 3] + std_query_boxes[k, 3]))
                    d_ia += ih * (abs(std_boxes[n, 0] - std_query_boxes[k, 0]) + 0.5 * (std_boxes[n, 2] + std_query_boxes[k, 2]))
                    std_overlaps[n, k] = d_ia / (iw * ih) if d_ia < iw * ih else 0.0

    return overlaps, std_overlaps


def disambiguate_ious(
    np.ndarray[DTYPE_t, ndim=2] ious,
    np.ndarray[DTYPE_t, ndim=2] ious_std,
    np.ndarray[DTYPE_t, ndim=2] reference,
    np.float_t thresh_iou,
    np.float_t thresh_std = 0.5):
    """
    Parameters
    ----------
    ious: (N, K) ndarray of float
    ious_std: (N, K) ndarray of float
    reference: (N, K) ndarray of float 
    thresh_iou: float
    thresh_iou (optional): float
    Returns
    -------
    ious: (N, K) ndarray of disambiguated ious
    was_ambiguous: bool -> whether or not the ious where ambiguous
    """
    cdef unsigned int N = ious.shape[0]
    cdef unsigned int K = ious.shape[1]
    cdef unsigned int i, j
    cdef unsigned int l, r
    cdef np.ndarray[BOOL_t, ndim=2] unresolved_ious = ious < 0
    cdef np.ndarray[DTYPE_t, ndim=2] new_ious = np.copy(ious)
    cdef bint was_ambiguous = False
    cdef float ll, lr, rl, rr, min_overlap

    # Get unresolved ious according to their overlaps per row
    for i in range(N):
        l, r = 0, K-1
        while l < r:
            if ious[i, l] > thresh_iou and ious[i, r] > thresh_iou: 
                ll = ious[i, l] - ious_std[i, l]
                lr = ious[i, l] + ious_std[i, l]
                rl = ious[i, r] - ious_std[i, r]
                rr = ious[i, r] + ious_std[i, r]
                min_overlap = thresh_std * (ious_std[i, l] + ious_std[i, r])

                # if overlap of ious is greater than min_overlap
                if min(lr, rr) - max(ll, rl) > min_overlap:
                    unresolved_ious[i, l] = True
                    unresolved_ious[i, r] = True
                    was_ambiguous = True
                r -= 1
                l += 1
            else:
                if ious[i, r] < thresh_iou:
                    r -= 1
                if ious[i, l] < thresh_iou:
                    l += 1

    # Rearrange the unresolved ious according to the order in reference
    for i in range(N):
        if unresolved_ious[i].any():
            idx_row_unresolved = np.where(unresolved_ious[i])[0]
            idx_sort_ref_unresolved = np.argsort(reference[i, idx_row_unresolved])
            sort_ious_unresolved = np.sort(ious[i, idx_row_unresolved])

            for j in range(len(idx_row_unresolved)):
                as_ref = idx_sort_ref_unresolved[j]
                new_iou = sort_ious_unresolved[j]
                new_ious[i, idx_row_unresolved[as_ref]] = new_iou

    return new_ious, was_ambiguous
