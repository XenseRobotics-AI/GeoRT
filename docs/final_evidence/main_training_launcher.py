import json,random
from pathlib import Path
import numpy as np
import torch
from geort.trainer import GeoRTTrainer
from geort.utils.config_utils import get_config
from geort import trainer
root=Path('/home/xense-wufan/GeoRT');report=root/'reports/main_geort_manus_seed42'
assert Path(trainer.__file__).resolve()==report/'source/geort/trainer.py'
protocol=json.loads((report/'protocol.json').read_text())
random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(4)
print('MAIN TRAINER',trainer.__file__,flush=True)
print('TRAIN DATA',protocol['train_path'],flush=True)
GeoRTTrainer(get_config('wuji_hand2_beta1_right')).train(Path(protocol['train_path']),tag='main_manus_train01_seed42',log_dir=str(report/'curves'),epoch=200,save_every=50,seed=42,no_update_last=True,paper_loss=True,paired_sampling=True,w_chamfer=80.,w_curvature=1.,w_collision=0.,w_pinch=1000.)
print('TRAINING COMPLETE',flush=True)
