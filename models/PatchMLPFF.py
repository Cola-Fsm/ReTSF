"""Original PatchMLP followed by utility-guided CAF refinement."""
from models.ReTSF_FF import Model as RefinementModel


class Model(RefinementModel):
    def __init__(self, configs):
        super().__init__(configs, backbone_name='PatchMLP')
