import numpy as np

# save previous measurement in track

# state = [pos, velocity]
position = [0, 1, 0, 0, 0, 0]
velocity = [2, 0, 0, 0, 0, 0]
acceleration = [0.1, 0, 0, 0, 0, 0]
dt = 4
position_transition = np.concatenate([np.diag(np.ones(6)), dt*np.diag(np.ones(6)), (0.5*dt**2)*np.diag(np.ones(6))], axis = 1)
velocity_transition = np.concatenate([np.zeros((6,6)), np.diag(np.ones(6)), dt*np.diag(np.ones(6))], axis = 1)
acceleration_transition = np.concatenate([np.zeros((6,6)), np.zeros((6,6)), np.diag(np.ones(6))], axis = 1)
state = np.concatenate([position, velocity, acceleration])
state_transition = np.concatenate([position_transition, velocity_transition, acceleration_transition])
observation_matrix = np.diag(np.ones(18))
control_matrix = np.zeros([18])
print(state_transition.shape)
print(state.shape)
print((state_transition @ state.T).T.shape)
print(control_matrix.shape)

measurement_position_variance = 10e-6
measurement_velocity_variance = 10e-1
measurement_acceleration_variance = 10e-1

measurement_var_position = np.concatenate([measurement_position_variance*np.diag(np.ones(6)), np.zeros((6,6)), np.zeros((6,6))], axis = 1)
measurement_var_velocity = np.concatenate([np.zeros((6,6)), measurement_velocity_variance*np.diag(np.ones(6)), np.zeros((6,6))], axis = 1)
measurement_var_acceleration = np.concatenate([np.zeros((6,6)), np.zeros((6,6)), measurement_acceleration_variance*np.diag(np.ones(6))], axis = 1)

measurement_variance = np.concatenate([measurement_var_position, measurement_var_velocity, measurement_var_acceleration])

state_position_variance = 10e-6
state_velocity_variance = 10e-1
state_acceleration_variance = 10e-1

state_var_position = np.concatenate([state_position_variance*np.diag(np.ones(6)), np.zeros((6,6)), np.zeros((6,6))], axis = 1)
state_var_velocity = np.concatenate([np.zeros((6,6)), state_velocity_variance*np.diag(np.ones(6)), np.zeros((6,6))], axis = 1)
state_var_acceleration = np.concatenate([np.zeros((6,6)), np.zeros((6,6)), state_acceleration_variance*np.diag(np.ones(6))], axis = 1)

process_variance = np.concatenate([state_var_position, state_var_velocity, state_var_acceleration])

print(measurement_variance.shape)
print(process_variance.shape)