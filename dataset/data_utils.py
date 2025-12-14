import numpy as np
from loguru import logger
from torch.utils.data import Dataset

class MergedDataset(Dataset):

    """
    Takes two datasets (labelled_dataset, unlabelled_dataset) and merges them
    Allows you to iterate over them in parallel
    """

    def __init__(self, labelled_dataset, unlabelled_dataset):

        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset
        self.target_transform = None

    def __getitem__(self, item):
        if item < len(self.labelled_dataset):
            data = self.labelled_dataset[item]
            tag = 1
        else:
            data = self.unlabelled_dataset[item - len(self.labelled_dataset)]
            tag = 2

        data_new = data.copy()
        data_new['tag'] = np.array([tag])
        return data_new

    def __len__(self):
        return len(self.unlabelled_dataset) + len(self.labelled_dataset)
    
    def get_tags(self):
        tags = [1] * len(self.labelled_dataset) + [2] * len(self.unlabelled_dataset)
        return np.array(tags)
    

    def get_labels(self):
        labelled_labels = self.labelled_dataset.targets
        unlabelled_labels = self.unlabelled_dataset.targets

        if self.labelled_dataset.target_transform is not None:
            labelled_labels = [self.labelled_dataset.target_transform(lbl) for lbl in labelled_labels]
        if self.unlabelled_dataset.target_transform is not None:
            unlabelled_labels = [self.unlabelled_dataset.target_transform(lbl) for lbl in unlabelled_labels]

        return np.array(labelled_labels + unlabelled_labels)



def dataset_stats(all_datasets, known_classes=None, target_xform_dict=None,
                  width=60, train_height=10, min_test_height=4):
    """
    Print an overview of the open-world dataset split.

    Expected keys in `all_datasets`:
        - 'train_labelled'
        - 'train_unlabelled'
        - 'test'
        - optionally 'val'

    The diagram contains two equal-width rectangles:
        TRAIN on top (with a labelled sub-block at top-left)
        TEST on bottom (height proportional / smaller than TRAIN)

    Args:
        all_datasets (dict): output of get_owdfa_datasets().
        known_classes (list[int] or None): ordered list of known class labels.
        target_xform_dict (dict[int, int] or None): global mapping orig_label -> new_label.
        width (int): total width of the outer boxes.
        train_height (int): height of TRAIN box interior (not counting borders).
        min_test_height (int): minimum height of TEST box interior.
    """

    def _get_targets(ds):
        if ds is None:
            return []
        if hasattr(ds, "targets") and ds.targets is not None:
            return list(ds.targets)
        # fallback: infer from samples
        return [s[1] if isinstance(s, (list, tuple)) else s.get("target") for s in ds.samples]

    def _unique_classes(ds):
        t = _get_targets(ds)
        return sorted(set(t)) if t else []

    def _n_c(ds):
        if ds is None:
            return 0, 0
        return len(ds), len(_unique_classes(ds))

    def _hline(w):
        return "+" + "-" * (w - 2) + "+"

    def _empty_row(w):
        return "|" + " " * (w - 2) + "|"

    def _put_text(line, text, col):
        # line is a mutable list of chars
        for i, ch in enumerate(text):
            if 0 <= col + i < len(line):
                line[col + i] = ch

    # ----------------------------
    # Gather datasets and stats
    # ----------------------------
    ds_L = all_datasets.get("train_labelled", None)
    ds_U = all_datasets.get("train_unlabelled", None)
    ds_T = all_datasets.get("test", None)

    nL, cL = _n_c(ds_L)
    nU, cU = _n_c(ds_U)
    nT, cT = _n_c(ds_T)

    nTrain = nL + nU
    # Train class set is union of labelled + unlabelled classes
    train_classes = sorted(set(_unique_classes(ds_L)) | set(_unique_classes(ds_U)))
    cTrain = len(train_classes)

    # ----------------------------
    # Mapping summary
    # ----------------------------
    mapping_info = ""
    if known_classes is not None and target_xform_dict is not None:
        # known range
        known_new = [target_xform_dict[k] for k in known_classes if k in target_xform_dict]
        if known_new:
            kmin, kmax = min(known_new), max(known_new)
        else:
            kmin, kmax = None, None

        # novel range (in train)
        novel_raw = [x for x in train_classes if x not in set(known_classes)]
        novel_new = [target_xform_dict[k] for k in novel_raw if k in target_xform_dict]
        if novel_new:
            nmin, nmax = min(novel_new), max(novel_new)
        else:
            nmin, nmax = None, None

        mapping_info = f" | map: known→[{kmin}..{kmax}], novel→[{nmin}..{nmax}]"
    elif known_classes is not None:
        mapping_info = " | map: known/novel unified (details not provided)"

    # ----------------------------
    # Decide TEST height (smaller, proportional-ish)
    # ----------------------------
    if nTrain > 0:
        ratio = nT / nTrain
    else:
        ratio = 0.0
    test_height = max(min_test_height, min(train_height, int(round(train_height * ratio))))

    # ----------------------------
    # Decide LABELLED sub-box size (top-left)
    # ----------------------------
    inner_w = width - 2
    # labelled width proportional to nL/nTrain, but keep sensible bounds
    if nTrain > 0:
        lw = int(round(inner_w * (nL / nTrain)))
    else:
        lw = int(round(inner_w * 0.2))
    lw = max(18, min(inner_w - 18, lw))  # keep room for unlabelled region

    # labelled height proportional to ratio, bounded
    if nTrain > 0:
        lh = int(round(train_height * (nL / nTrain)))
    else:
        lh = 3
    lh = max(4, min(train_height - 2, lh))  # at least 4 lines, not too tall

    # ----------------------------
    # Build TRAIN box canvas
    # ----------------------------
    train_lines = []
    train_lines.append(_hline(width))
    for _ in range(train_height):
        train_lines.append(_empty_row(width))
    train_lines.append(_hline(width))

    # Draw LABELLED sub-box inside TRAIN (top-left)
    # Coordinates in the canvas (line index, column index)
    # We draw inside borders, so start row=1, col=1
    top = 1
    left = 1
    sub_w = lw
    sub_h = lh

    # top border of sub-box
    row = list(train_lines[top])
    row[left] = "|"
    _put_text(row, "+" + "-" * (sub_w - 2) + "+", left)
    train_lines[top] = "".join(row)

    # middle rows of sub-box
    for r in range(top + 1, top + sub_h - 1):
        row = list(train_lines[r])
        _put_text(row, "|" + " " * (sub_w - 2) + "|", left)
        train_lines[r] = "".join(row)

    # bottom border of sub-box
    row = list(train_lines[top + sub_h - 1])
    _put_text(row, "+" + "-" * (sub_w - 2) + "+", left)
    train_lines[top + sub_h - 1] = "".join(row)

    # Put text inside LABELLED sub-box
    def _write_in_labelled(r_offset, text):
        r = top + 1 + r_offset
        if r >= top + sub_h - 1:
            return
        row = list(train_lines[r])
        _put_text(row, text, left + 2)
        train_lines[r] = "".join(row)

    _write_in_labelled(0, "LABELLED")
    _write_in_labelled(1, f"N={nL}, C={cL}")
    if known_classes is not None and target_xform_dict is not None:
        _write_in_labelled(2, "subset of TRAIN (known)")

    # Put UNLABELLED block info in the remaining area (top, to the right)
    # We don't draw a sub-box for it; we annotate the region.
    ul_text1 = "UNLABELLED"
    ul_text2 = f"N={nU}, C={cU}"
    ul_text3 = "(known remainder + novel)"
    # place them on the first two interior lines of TRAIN, to the right of labelled box
    ul_col = 1 + sub_w + 2

    ul_row1 = 2
    ul_row2 = 3
    ul_row3 = 4

    if ul_col < width - 2:
        row = list(train_lines[ul_row1])
        _put_text(row, ul_text1, ul_col)
        train_lines[ul_row1] = "".join(row)

        row = list(train_lines[ul_row2])
        _put_text(row, ul_text2, ul_col)
        train_lines[ul_row2] = "".join(row)

        row = list(train_lines[ul_row3])
        _put_text(row, ul_text3, ul_col)
        train_lines[ul_row3] = "".join(row)


    # ----------------------------
    # Build TEST box canvas
    # ----------------------------
    test_lines = []
    test_lines.append(_hline(width))
    for _ in range(test_height):
        test_lines.append(_empty_row(width))
    test_lines.append(_hline(width))

    # ----------------------------
    # Print
    # ----------------------------
    logger.info(f"TRAIN (N={nTrain}, C={cTrain}){mapping_info}")
    for ln in train_lines:
        logger.info(ln)

    logger.info(f"TEST  (N={nT}, C={cT})")
    for ln in test_lines:
        logger.info(ln)
