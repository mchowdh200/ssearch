import sys

import nvitop

ngpus = sys.argv[1]
ngpus = 4
nparts = 2

mig_devices = [
    nvitop.MigDevice((i, j)).uuid()
    for i in range(ngpus)
    for j in range(nparts)
    if len(nvitop.MigDevice((i, j)).processes().keys()) == 0
][:ngpus]

for d in mig_devices:
    print(d)
