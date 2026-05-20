from demo_configs import INPUTS
import pulp
import numpy as np
import pandas as pd

from dwave.optimization import Model, put
import dwave.optimization
from dwave.optimization import Model, put, symbols
from dwave.optimization.mathematical import concatenate, argsort
from dwave.system import LeapHybridNLSampler

def create_runtime_use_matrices():

    variables, problem = pulp.LpProblem.fromMPS(INPUTS[0])

    data = problem.to_dict()

    rm = {(j,m,t): [] for j in range(1,31) for m in range(1,4) for t in range(213)}
    rt = {(j,m,t): [] for j in range(1,31) for m in range(1,4) for t in range(213)}
    rm_use = {(j,m,t): 0 for j in range(1,31) for m in range(1,4) for t in range(213)}
    rt_use = {(j,m,t): 0 for j in range(1,31) for m in range(1,4) for t in range(213)}
    for const in data['constraints']:
        name = const['name']
        if name.startswith('rescap_Mechaniker'):
            const_time = int(name.split('_')[2])
            for coeff in const['coefficients']:
                coeff_name = coeff['name']
                if coeff_name.startswith('x'):
                    job = int(coeff_name.split('_')[1])
                    mode = int(coeff_name.split('_')[2])
                    time = int(coeff_name.split('_')[3])
                    rm[(job, mode, const_time)].append(time)
                    value = int(coeff['value'])
                    rm_use[(job, mode, const_time)] = value

        elif name.startswith('rescap_Techniker'):
            const_time = int(name.split('_')[2])
            for coeff in const['coefficients']:
                coeff_name = coeff['name']
                if coeff_name.startswith('x'):
                    job = int(coeff_name.split('_')[1])
                    mode = int(coeff_name.split('_')[2])
                    time = int(coeff_name.split('_')[3])
                    rt[(job, mode, const_time)].append(time)
                    value = int(coeff['value'])
                    rt_use[(job, mode, const_time)] = value

    rm = {k: set(v) for k,v in rm.items()}
    rt = {k: set(v) for k,v in rt.items()}

    # runtimes per job and mode and resource type
    rm_times = {(job, mode): 0 for job in range(1,31) for mode in range(1,4)}
    rt_times = {(job, mode): 0 for job in range(1,31) for mode in range(1,4)}

    for (j,m,t), v in rm.items():
        rm_times[(j,m)] = max(rm_times[(j,m)], len(v))

    for (j,m,t), v in rt.items():
        rt_times[(j,m)] = max(rt_times[(j,m)], len(v))

    # resource use per job, mode, and resource type

    rm_use_total = {(job, mode): 0 for job in range(1,31) for mode in range(1,4)}
    rt_use_total = {(job, mode): 0 for job in range(1,31) for mode in range(1,4)}

    for (j,m,t), v in rm_use.items():
        rm_use_total[(j,m)] = max(v, rm_use_total[(j,m)])

    for (j,m,t), v in rt_use.items():
        rt_use_total[(j,m)] = max(v, rt_use_total[(j,m)])

    runtimes = {}
    for k,v in rm_times.items():
        runtimes[k] = max(v, rt_times[k])

    runtimes_matrix = np.zeros((30, 3))

    for (j,m),v in runtimes.items():
        runtimes_matrix[j-1,m-1] = v

    rm_use_matrix = np.zeros((30,3))
    rt_use_matrix = np.zeros((30,3))

    for (j,m),v in rm_use_total.items():
        rm_use_matrix[j-1,m-1] = v

    for (j,m),v in rt_use_total.items():
        rt_use_matrix[j-1,m-1] = v

    return runtimes_matrix, rm_use_matrix, rt_use_matrix

def create_precedence_pairs():
    variables, problem = pulp.LpProblem.fromMPS(INPUTS[0])
    data = problem.to_dict()

    precedence_pairs = []

    for const in data['constraints']:
        if const['name'].startswith('prec'):
            for var in const['coefficients']:
                if var['name'].startswith('C'):
                    v1 = int(var['name'].split('_')[1])
                elif var['name'].startswith('S'):
                    v2 = int(var['name'].split('_')[1])
            precedence_pairs.append((v1,v2))

    return precedence_pairs

def create_model(lower_bounds, upper_bounds, runtimes_matrix, rm_use_matrix, rt_use_matrix,
                 precedence_pairs):
    # accumulate zip formulation
    model = Model()

    num_jobs = 30
    time_horizon = 213
    upper_bounds_modes = [2,2,2,2,2,2,2,2,2,1,2,2,2,2,2,1,2,2,2,2,2,1,2,2,2,2,2,2,1,2]

    starts = model.integer(num_jobs, lower_bound=lower_bounds, upper_bound=upper_bounds)

    modes = model.integer(num_jobs, lower_bound=0, upper_bound=upper_bounds_modes)

    runtimes_mat_mod = runtimes_matrix - 1

    runtimes_matrix_c = model.constant(runtimes_matrix)
    runtimes_mat_mod_c = model.constant(runtimes_mat_mod)
    rm_use_matrix_c = model.constant(rm_use_matrix)
    rt_use_matrix_c = model.constant(rt_use_matrix)

    for (j1,j2) in precedence_pairs:
        model.add_constraint(starts[model.constant(j1-1)]
                            + runtimes_matrix_c[model.constant(j1-1),modes[model.constant(j1)]]
                            - starts[model.constant(j2-1)] <= 0)

    # upper bounds from lp file
    r_mechaniker = model.integer(1, lower_bound=0, upper_bound=40)
    r_techniker = model.integer(1, lower_bound=0, upper_bound=30)

    # get runtime for each job (depends on which mode it is in)
    runtimes = model.constant(np.zeros(num_jobs))
    for i in range(num_jobs):
        runtimes = put(runtimes, model.constant(i).reshape((1)), runtimes_mat_mod_c[i, modes[i]].reshape((1)))

    end_times = starts + runtimes

    # concatenate the consumption with the negative consumption for start and end times

    # get total consumption (mechaniker and techniker)
    #  note that each job uses only one resource and not the other, so we can just sum over all jobs

    m_consumption_pos = model.constant(np.zeros(num_jobs))
    t_consumption_pos = model.constant(np.zeros(num_jobs))

    for i in range(num_jobs):
        m_consumption_pos = put(m_consumption_pos, indices = model.constant(i).reshape((1)),
            values = rm_use_matrix_c[i,modes[i]].reshape((1)))
        t_consumption_pos = put(t_consumption_pos, indices = model.constant(i).reshape((1)),
            values = rt_use_matrix_c[i, modes[i]].reshape((1)))

    m_consumption = concatenate((m_consumption_pos, -1*m_consumption_pos))
    t_consumption = concatenate((t_consumption_pos, -1*t_consumption_pos))

    events = concatenate((starts, end_times))
    order = argsort(events)
    order_events_consumption_m = m_consumption[order]
    order_events_consumption_t = t_consumption[order]
    # track the cumulative consumption, adding consumption when a job starts
    #  and subtracting when it ends

    @dwave.optimization.expression
    def add(x, y):
        return x + y

    cumulative_consumption_m = symbols.AccumulateZip(
        add, (order_events_consumption_m, ), initial=model.constant(0)
        )
    model.add_constraint((cumulative_consumption_m <= r_mechaniker).all())

    cumulative_consumption_t = symbols.AccumulateZip(
        add, (order_events_consumption_t, ), initial=model.constant(0)
        )
    model.add_constraint((cumulative_consumption_t <= r_techniker).all())

    model.objective = 100*r_mechaniker + 51*r_techniker

    return model


lower_bounds = [0,0,6,79,5,83,35,62,67,76,29,35,39,79,103,10,44,89,13,107,19,53,93,56,
                113,85,103,62,113,116]
upper_bounds = [90,87,93,166,100,175,127,154,159,163,116,186,131,188,190,102,136,181,
                105,194,111,140,185,143,209,200,207,182,205,208]

runtimes_matrix, rm_use_matrix, rt_use_matrix = create_runtime_use_matrices()
precedence_pairs = create_precedence_pairs()


data = []
time_limits = [5,10]

for time in time_limits:
    for _ in range(1):
        model = create_model(lower_bounds, upper_bounds, runtimes_matrix, rm_use_matrix,
                            rt_use_matrix, precedence_pairs)
        model.lock()
        solver = LeapHybridNLSampler()
        solver.sample(model, time_limit = time)
        energy = model.objective.state()
        feas = all(sym.state() for sym in model.iter_constraints())

        data.append({
            'solver': 'Stride solver',
            'time': time,
            'feasibility': feas,
            'energy': energy
        })
        df = pd.DataFrame(data)
        df.to_csv('nl_rcpsp.csv', index=False)
