from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class LakeSegDataset(BaseSegDataset):
    
    METAINFO = {
        'classes':['background','lake'],
        'palette':[[0,0,0],[1,1,1]]
    }
    
    
    def __init__(self,
                 seg_map_suffix='.png',   
                 reduce_zero_label=False,  # This should be false since 0 is our background
                 **kwargs) -> None:
        super().__init__(
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)