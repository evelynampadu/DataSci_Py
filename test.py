
import numpy as np


vac_nums = [0,0,0,0,0,
            1,1,1,1,1,1,1,1,
            2,2,2,2,
            3,3,3
            ]

mean = np.mean(vac_nums)
standard_deviation = np.std(vac_nums)
variance = np.var(vac_nums)
result = 0
for i in vac_nums:
    if i >= mean-standard_deviation and i <= mean+standard_deviation:
        result += 1
print(result)