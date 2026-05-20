from demo_configs import INPUTS
import highspy as hs
import pandas as pd

time_limits = [5,10]
data = []

for time in time_limits:
    for _ in range(5):
        h = hs.Highs()
        h.readModel(INPUTS[0])
        h.setOptionValue('time_limit', time)
        h.setOptionValue('output_flag', False)
        status = h.run()
        info = h.getInfo()
        energy = info.objective_function_value
        status_str = h.getModelStatus()

        data.append({
            'solver': 'HiGHS',
            'time': time,
            'energy': energy,
            'status': status_str
        })

        df = pd.DataFrame(data)
        df.to_csv('highs_rcpsp.csv', index=False)
