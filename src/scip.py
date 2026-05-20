from demo_configs import INPUTS
from pyscipopt import Model
import pandas as pd
import numpy as np

time_limits=[5,10]
data = []

for time in time_limits:
    for _ in range(5):
        model = Model()
        model.readProblem(INPUTS[0]) #can also use LP file
        model.setParam('limits/time', time)
        model.hideOutput()
        model.optimize()

        energy = model.getPrimalbound()
        status = model.getStatus()

        data.append({
            'time_limit': time,
            'solver': 'SCIP',
            'energy': energy,
            'status': status
        })

        df = pd.DataFrame(data)
        df.to_csv('scip_rcpsp.csv', index=False)
