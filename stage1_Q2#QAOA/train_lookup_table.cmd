:ft_ada_decay
python train_lookup_table.py --name ft-ada-decay --dx_decay 0.98

:ft_ada_decay_moment_fast
python train_lookup_table.py --name ft-ada-decay-moment-fast --dx_decay 0.98 --momentum 0.6 --steps 40
REM resume from fad-9400, finetune 1k step
python train_lookup_table.py --name ft-ada-decay-moment-fast_ft --dx_decay 0.98 --momentum 0.6 --steps 40 --load log\ft-ada-decay\lookup_table-iter=9400.json --iters 10400
