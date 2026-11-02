import pandas as pd
from sklearn.utils import Bunch


def make_dataset(data, target, metadata, description="", feature_pipeline=None):
    target_frame = target.to_frame() if isinstance(target, pd.Series) else target
    return Bunch(
        data=data,
        target=target,
        X=data,
        y=target,
        frame=pd.concat([data, target_frame], axis=1),
        feature_names=list(data.columns),
        target_names=list(target_frame.columns),
        metadata=metadata,
        DESCR=description,
        feature_pipeline=feature_pipeline,
    )
