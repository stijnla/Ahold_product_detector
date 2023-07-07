import numpy as np

# Implementation of a Kalman Filter estimating 6 Degrees of Freedom
# The state is defined as [x, y, z, theta, phi, psi]

class KalmanFilter:
	
	def __init__(self, init_state, frequency, initial_state_variance=1, measurement_variance=10, method="Velocity"):
		
		self.method = method # Model acceleration or not
		

		# Variances used in prediction
		self.initialStateVariance = initial_state_variance
		self.U = np.ones((1,1)) if method == "Acceleration" else np.zeros((1,1))
		
		dt = 1/frequency
		
		# State transition matrix, a kinematic model of state transition here
		position_transition = np.concatenate([np.diag(np.ones(6)), dt*np.diag(np.ones(6))], axis = 1)
		velocity_transition = np.concatenate([np.zeros((6,6)), np.diag(np.ones(6))], axis = 1)

		self.A = np.concatenate([position_transition, velocity_transition])
		print(self.A)
		# Control input matrix, models influence of control input on state
		self.B = np.zeros([12]).reshape(1, 12)
		
		# Observation model matrix,
		self.H = np.diag(np.ones(12))

		# Measurement Variance
		position_variance = 10e-12
		velocity_variance = 10e-3
		
		var_position = np.concatenate([position_variance*np.diag(np.ones(6)), np.zeros((6,6))], axis = 1)
		var_velocity = np.concatenate([np.zeros((6,6)), velocity_variance*np.diag(np.ones(6))], axis = 1)

		self.measurementVariance = np.concatenate([var_position, var_velocity])

		# State transition variance
		var = 10e30
		process_variance = np.diag(np.ones((1, 1))) * var


		# Observation noise covariance matrix (Measured with one object with 2500 measurements)
		self.R = self.measurementVariance

		# Process noise covariance matrix
		self.Q = self.B * process_variance * self.B.T

		# Estimated covariance matrix
		self.P = self.initialStateVariance*np.identity(self.A.shape[0])
		self.err_cov = self.P
		self.pred_err_cov = self.P

		# Ensure proper shape of state for calculation
		self.state = init_state.reshape(init_state.shape[0], 1)
		self.pred_state = init_state.reshape(init_state.shape[0], 1)




	def predict(self):
		# Predict next state and the next covariance matrix

		self.pred_state = (self.A.T @ self.state) + (self.B * self.U).T
		self.pred_err_cov = self.A@self.err_cov@self.A.T + self.Q



	def update(self, measurement):
		# Check whether sensor is giving trusty readouts, otherwise neglect by having a large observation variance matrix
		# Use measurement to refine prediction
		kalman_gain = self.pred_err_cov@self.H.T@np.linalg.pinv(self.H@self.pred_err_cov@self.H.T+self.R)
	
		self.state = (self.pred_state + kalman_gain @ (measurement - (self.H @ self.pred_state)))
		self.err_cov = (np.identity(self.err_cov.shape[0]) - kalman_gain@self.H)@self.pred_err_cov



	def update_matrices(self, frequency):
		# For certain applications, the matrices should be updated to in order to
		# incorporate the processing time necessary for these (and potentially other)
		# steps.

		dt = 1/frequency
		
		# State transition matrix, a kinematic model of state transition here
		position_transition = np.concatenate([np.diag(np.ones(6)), dt*np.diag(np.ones(6))], axis = 1)
		velocity_transition = np.concatenate([np.zeros((6,6)), np.diag(np.ones(6))], axis = 1)

		self.A = np.concatenate([position_transition, velocity_transition])

		