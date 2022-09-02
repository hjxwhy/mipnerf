SCENE=lego
EXPERIMENT=debug
TRAIN_DIR=/home/hjx/Videos/block_exp/mip
DATA_DIR=/media/hjx/DataDisk/waymo/WaymoDataset
export XLA_PYTHON_CLIENT_MEM_FRACTION=.7
rm $TRAIN_DIR/*
python -m train \
  --data_dir=$DATA_DIR \
  --train_dir=$TRAIN_DIR \
  --gin_file=configs/waymo.gin \
  --logtostderr