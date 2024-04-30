# NOTE: 答题时只能修改此文件，整合一切代码，因为评测机只会上传这单个脚本！！
# 一点也不能改 solution() 函数的签名，例如不能给 molecule 加类型注解，因为评测机做了硬编码的校验！！

import os, sys
sys.path.append(os.path.abspath(__file__))

from typing import *

from simulator import HKSSimulator


def solution(molecule, Simulator: HKSSimulator) -> float:
    molecule: List[Tuple[str, List[float]]]     # i.e.: geometry
