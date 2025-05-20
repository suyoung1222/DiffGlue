from pathlib import Path

root = Path(__file__).parent.parent  # top-level directory
# DATA_PATH = root / "data/"  # datasets and pretrained weights
# DATA_PATH = root / "/project/pi_hzhang2_umass_edu/suyoungkang_umass_edu/diffglue_data/"  # datasets and pretrained weights
DATA_PATH = Path("/project/pi_hzhang2_umass_edu/suyoungkang_umass_edu/diffglue_data/data/")
TRAINING_PATH = Path("/project/pi_hzhang2_umass_edu/suyoungkang_umass_edu/diffglue_data/outputs/training/")  # training checkpoints
EVAL_PATH = Path("/project/pi_hzhang2_umass_edu/suyoungkang_umass_edu/diffglue_data/outputs/results/")  # evaluation results
