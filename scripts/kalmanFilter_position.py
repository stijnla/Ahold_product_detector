import numpy as np

# Implementation of a Kalman Filter estimating 6 Degrees of Freedom
# The state is defined as [x, y, z, theta, phi, psi]

class KalmanFilter:
	
	def __init__(self, init_state, frequency, initial_state_variance=1, measurement_variance=10, method="Velocity"):
		
		self.method = method # Model acceleration or not
		

		# Variances used in prediction
		self.initialStateVariance = initial_state_variance
		self.processVariance = 1000
		self.U = 1 if method == "Acceleration" else 0
		
		dt = 1/frequency
		
		# State transition matrix, a kinematic model of state transition here
		self.A = np.matrix([
			[1, 0, 0, 0, 0, 0],
			[0, 1, 0, 0, 0, 0],
			[0, 0, 1, 0, 0, 0],
			[0, 0, 0, 1, 0, 0],
			[0, 0, 0, 0, 1, 0],
			[0, 0, 0, 0, 0, 1],
		])
		# Control input matrix, models influence of control input on state
		self.B = np.matrix([
			[dt],
			[dt],
			[dt],
			[dt],
			[dt],
			[dt],
		])
		
		# Observation model matrix,
		self.H = np.matrix([
			[1, 0, 0, 0, 0, 0],
			[0, 1, 0, 0, 0, 0],
			[0, 0, 1, 0, 0, 0],
			[0, 0, 0, 1, 0, 0],
			[0, 0, 0, 0, 1, 0],
			[0, 0, 0, 0, 0, 1],
			
		])
		self.measurementVariance = (10**-9) * np.array([[8.6,   -5.7,  -24.4,   -0.7,     15.9,   0],
														[-5.7,  106.,  48.8,    36.3,   -2.4,    0],
														[-24.4, 48.8,  217.9,   18.6,   -211.3,  0],
														[-0.7,  36.3,  18.6,    1433.8,  731.7,  0],
														[15.9, -2.4,  -211.3,   731.7,   4148.6, 0],
														[0,     0,     0,       0,       0,      0 ]])
		# Observation noise covariance matrix (Measured with one object with 2500 measurements)
		self.R = self.measurementVariance

		# Process noise covariance matrix
		self.Q = self.B * np.diag(np.ones((1, 1))) * self.processVariance * self.B.T

		# Estimated covariance matrix
		self.P = np.matrix(self.initialStateVariance*np.identity(self.A.shape[0]))
		self.err_cov = self.P
		self.pred_err_cov = self.P

		# Ensure proper shape of state for calculation
		self.state = init_state.reshape(6, 1)
		self.pred_state = init_state.reshape(6, 1)



	def predict(self):
		# Predict next state and the next covariance matrix
		self.pred_state = self.A*self.state + self.B*self.U
		self.pred_err_cov = self.A*self.err_cov*self.A.T + self.Q



	def update(self, measurement):
		# Check whether sensor is giving trusty readouts, otherwise neglect by having a large observation variance matrix
		self.check_measurement_trustworthiness(measurement)

		# Use measurement to refine prediction
		kalman_gain = self.pred_err_cov*self.H.T*np.linalg.pinv(self.H*self.pred_err_cov*self.H.T+self.R)

		self.state = self.pred_state + kalman_gain*(measurement - (self.H*self.pred_state))
		self.err_cov = (np.identity(self.err_cov.shape[0]) - kalman_gain*self.H)*self.pred_err_cov



	def update_matrices(self, frequency):
		# For certain applications, the matrices should be updated to in order to
		# incorporate the processing time necessary for these (and potentially other)
		# steps.

		dt = 1/frequency
		
		# State transition matrix, a kinematic model of state transition here
		self.A = np.matrix([
			[1, 0, 0, 0, 0, 0],
			[0, 1, 0, 0, 0, 0],
			[0, 0, 1, 0, 0, 0],
			[0, 0, 0, 1, 0, 0],
			[0, 0, 0, 0, 1, 0],
			[0, 0, 0, 0, 0, 1],
		])
		# Control input matrix, models influence of control input on state
		self.B = np.matrix([
			[dt],
			[dt],
			[dt],
			[dt],
			[dt],
			[dt],
		])

		# Update observation matrix
		self.Q = self.B * self.B.T

		


	def check_measurement_trustworthiness(self, measurement):
		edge_value = 10**-6

		if measurement[0] < edge_value and measurement[1] < edge_value and measurement[2] < edge_value:
			# object too close to the camera, measurments cannot be trusted
			very_large = 10**9
			self.R = np.diag(very_large*np.ones((6)))
		else:
			self.R = self.measurementVariance

