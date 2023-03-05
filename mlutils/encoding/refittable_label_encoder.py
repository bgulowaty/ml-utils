from sklearn.preprocessing import LabelEncoder
import numpy as np


class RefitableLabelEncoder(LabelEncoder):

    def __init__(self) -> None:
        super().__init__()

    def fit(self, y):
        if hasattr(self, 'classes_'):
            self._add_new_labels_to_encoder(y)

            return self

        return super().fit(y)

    def _add_new_labels_to_encoder(self, y):
        new_labels = [l for l in y if l not in self.classes_]
        if new_labels:
            self.classes_ = np.append(self.classes_, new_labels)

    def fit_transform(self, y):
        if hasattr(self, 'classes_'):
            self._add_new_labels_to_encoder(y)
            return super().transform(y)

        return super().fit_transform(y)

