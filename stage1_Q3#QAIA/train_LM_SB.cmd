:DU_LM_SB
python train_LM_SB.py -M DU_LM_SB
python train_LM_SB.py -M DU_LM_SB --overfit

:pReg_LM_SB
REM python train_LM_SB.py -M pReg_LM_SB
python train_LM_SB.py -M pReg_LM_SB --load log\DU-LM-SB_T=10_lr=0.01.pth --lr 0.005
REM python train_LM_SB.py -M pReg_LM_SB --overfit
python train_LM_SB.py -M pReg_LM_SB --overfit --load log\DU-LM-SB_T=10_lr=0.01_overfit.pth --lr 0.005

:ppReg_LM_SB
python train_LM_SB.py -M ppReg_LM_SB
REM python train_LM_SB.py -M ppReg_LM_SB --load log\DU-LM-SB_T=10_lr=0.01.pth --lr 0.005
python train_LM_SB.py -M ppReg_LM_SB --overfit
REM python train_LM_SB.py -M ppReg_LM_SB --overfit --load log\DU-LM-SB_T=10_lr=0.01_overfit.pth --lr 0.005
