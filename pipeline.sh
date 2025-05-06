set -e
CONFIG1=./configs/lego-projection.txt
CONFIG2=./configs/lego-projection-5.txt


python run_nerf.py --config $CONFIG1 --ft_path './logs/lego-proj/200000.tar' --step1_initialize
python run_nerf.py --config $CONFIG2 --ft_path './logs/lego-proj/210000.tar' --step2_freeze
python run_nerf.py --config $CONFIG2 --ft_path './logs/lego-proj/220000.tar' --step3_unfreeze