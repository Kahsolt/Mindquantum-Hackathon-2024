:DU_LM_SB
python train_LM_SB.py -M DU_LM_SB --lr 0.01
python train_LM_SB.py -M DU_LM_SB --lr 0.0001 --steps 100000

:pReg_LM_SB
python train_LM_SB.py -M pReg_LM_SB --lr 0.01
python train_LM_SB.py -M pReg_LM_SB --lr 0.0001 --steps 100000

python train_LM_SB.py -M pReg_LM_SB --load log\DU-LM-SB_T=10_lr=0.01.pth --lr 0.005
python train_LM_SB.py -M pReg_LM_SB --load log\DU-LM-SB_T=10_lr=0.0001.pth --lr 0.0001 --steps 50000

:ppReg_LM_SB
python train_LM_SB.py -M ppReg_LM_SB --lr 0.01
python train_LM_SB.py -M ppReg_LM_SB --lr 0.0001 --steps 100000

python train_LM_SB.py -M ppReg_LM_SB --load log\DU-LM-SB_T=10_lr=0.01.pth --lr 0.005
python train_LM_SB.py -M ppReg_LM_SB --load log\DU-LM-SB_T=10_lr=0.0001.pth --lr 0.0001 --steps 50000

:pppReg_LM_SB
python train_LM_SB.py -M pppReg_LM_SB --lr 0.01
