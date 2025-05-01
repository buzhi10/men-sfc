import random
import math
import numpy as np
import sys
import os

vnf_type = 15
random.seed(42)
np.random.seed(42)



def create_SFC(num_sfc=5, min_length=2, max_length=6):
    SFCs = []
    for _ in range(num_sfc):
        sfc_length = np.random.randint(min_length, max_length + 1)

        used_vnf_ids = set()
        sfc = []
        for _ in range(sfc_length):
            while True:
                vnf = np.array([
                    np.random.randint(100, 301),
                    np.random.randint(5, 10),
                    np.random.randint(0, vnf_type)
                ])
                if vnf[2] not in used_vnf_ids:
                    used_vnf_ids.add(vnf[2])
                    sfc.append(vnf)
                    break

        while len(sfc) < max_length:
            sfc.append(np.array([0, 0, 0]))
        SFCs.append(sfc)

    SFCs = np.array(SFCs, dtype=np.int32)
    return SFCs


SFCs = create_SFC()
config_dir = 'config'
os.makedirs(config_dir, exist_ok=True)

sfc_file_path = os.path.join(config_dir, 'SFCs_5.npy')
np.save(sfc_file_path, SFCs)