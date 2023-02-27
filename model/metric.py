from typing import Dict
import numpy as np
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
import torch
from utils.dist_utils import all_reduce_tensor
import pandas as pd
import os

__all__ = ["UnSegMetrics"]


class UnSegMetrics(nn.Module):

    def __init__(self,
                 num_classes: int,
                 extra_classes: int,
                 compute_hungarian: bool,
                 device: torch.device
                 ) -> None:
        super().__init__()

        self.num_classes = num_classes

        if (not compute_hungarian) and (extra_classes != 0):
            raise ValueError("No hungarian means that all classes are in order, so extra classes should be 0.")

        self.compute_hungarian = compute_hungarian
        self.extra_classes = extra_classes
        self.device = device

        self.register_buffer("confusion_matrix",
                             torch.zeros(num_classes + extra_classes, num_classes, dtype=torch.long, device=device))
        # self.confusion_matrix = torch.zeros(num_classes + extra_classes, num_classes, dtype=torch.long)

        self.assignments = None  # placeholder
        self.histogram = None  # placeholder

    def reset(self):
        self.confusion_matrix.fill_(0)
        self.assignments = None
        self.histogram = None

    @torch.no_grad()
    def update(self, preds: torch.Tensor, label: torch.Tensor):
        """Accumulate confusion matrix."""
        preds = preds.view(-1)
        label = label.view(-1)
        mask = (label >= 0) & (label < self.num_classes) & (preds >= 0) & (preds < self.num_classes)
        preds = preds[mask]
        label = label[mask]

        confusion = torch.bincount(
            label * (self.num_classes + self.extra_classes) + preds,  # row: label, colum: pred,
            minlength=self.num_classes * (self.num_classes + self.extra_classes)
        )
        confusion = confusion.reshape(self.num_classes, self.num_classes + self.extra_classes).t()
        self.confusion_matrix += confusion

    @torch.no_grad()
    def compute(self, prefix: str = None) -> Dict[str, torch.Tensor]:
        """Measure mIoU and accuracy."""
        self.confusion_matrix = all_reduce_tensor(self.confusion_matrix, op="sum")

        if self.compute_hungarian:  # cluster
            self.assignments = linear_sum_assignment(self.confusion_matrix.detach().cpu(), maximize=True)
            # the output of 'linear_sum_assignment':
            # (row_indices,)  ex) [0, 3, 1, 2, 4]
            # (column_indices)  ex) [3, 4, 1, 0, 2]
            # indicates which row index <--> column index  ex) 0<->3, 3<->4, ... 4<->2

            if self.extra_classes == 0:
                self.histogram = self.confusion_matrix[np.argsort(self.assignments[1]), :]
            else:
                assignments_t = linear_sum_assignment(self.confusion_matrix.detach().cpu().t(), maximize=True)
                histogram = self.confusion_matrix[assignments_t[1], :]
                missing = list(set(range(self.num_classes + self.extra_classes)) - set(self.assignments[0]))

                new_row = self.confusion_matrix[missing, :].sum(0, keepdim=True)
                histogram = torch.cat([histogram, new_row], dim=0)
                new_col = torch.zeros(self.num_classes + 1, 1, device=histogram.device)
                self.histogram = torch.cat([histogram, new_col], dim=1)
        else:  # linear
            self.assignments = (torch.arange(self.num_classes).unsqueeze(1),
                                torch.arange(self.num_classes).unsqueeze(1))
            self.histogram = self.confusion_matrix

        tp = torch.diag(self.histogram)
        fp = torch.sum(self.histogram, dim=0) - tp
        fn = torch.sum(self.histogram, dim=1) - tp

        iou = tp / (tp + fp + fn)
        iou = iou[~torch.isnan(iou)].mean()
        precision = tp / (tp + fn)
        accuracy = torch.sum(tp) / torch.sum(self.histogram)

        output = dict(iou=100 * iou, accuracy=100 * accuracy)

        # TODO class_matrix acc
        # os.makedirs(f'./class_matrix/', exist_ok=True)
        #
        # # tp : (27, 1)
        # precision = precision * 100
        # precision = precision.unsqueeze(-1)
        # tmp = torch.cat([self.histogram, precision], dim=1)
        # matrix_np = tmp.cpu().numpy()
        # matrix_df = pd.DataFrame(matrix_np)
        # matrix_df.to_csv(f'./class_matrix/STEGO/{prefix}/{prefix}_crf.csv')

        return output

    @torch.no_grad()
    def map_clusters(self, clusters):
        if self.extra_classes == 0:
            return torch.tensor(self.assignments[1])[clusters]
        else:
            missing = sorted(list(set(range(self.num_classes + self.extra_classes)) - set(self.assignments[0])))
            cluster_to_class = self.assignments[1]
            for missing_entry in missing:
                if missing_entry == cluster_to_class.shape[0]:
                    cluster_to_class = np.append(cluster_to_class, -1)
                else:
                    cluster_to_class = np.insert(cluster_to_class, missing_entry + 1, -1)
            cluster_to_class = torch.tensor(cluster_to_class)
            return cluster_to_class[clusters]
