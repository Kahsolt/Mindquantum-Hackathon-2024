@ECHO OFF
CHCP 65001

SET SFX=_蒋松儒_13052375730

GOTO pack_%1


:pack_1
SET SRC=stage1_Q1#QSim
SET TGT=hackathon_qsim%SFX%
IF EXIST %TGT%.zip DEL %TGT%.zip

7z u %TGT%.zip ^
  %SRC%\solution.py

7z rn %TGT%.zip ^
  %SRC% %TGT%

GOTO END


:pack_2
SET SRC=stage1_Q2#QAOA
SET TGT=hackathon_qaoa%SFX%
IF EXIST %TGT%.zip DEL %TGT%.zip

7z u %TGT%.zip ^
  %SRC%\data\* ^
  %SRC%\utils\* ^
  %SRC%\score.py ^
  %SRC%\README.md ^
  %SRC%\main.py

7z rn %TGT%.zip ^
  %SRC% %TGT%

GOTO END


:pack_3
SET SRC=stage1_Q3#QAIA
SET TGT=hackathon_qaia%SFX%
IF EXIST %TGT%.zip DEL %TGT%.zip

7z u %TGT%.zip ^
  %SRC%\graphs\* ^
  %SRC%\MLD_data\* ^
  %SRC%\qaia\* ^
  %SRC%\judger.py ^
  %SRC%\readme.md ^
  %SRC%\main.py

7z rn %TGT%.zip ^
  %SRC% %TGT%

GOTO END


:pack_
ECHO ^>^> Nothing to do, should specify number in {1, 2, 3}.

:END
