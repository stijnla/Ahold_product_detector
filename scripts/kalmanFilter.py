import numpy as np

# Implementation of a Kalman Filter estimating 6 Degrees of Freedom
# The state is defined as [x, x_dot, y, y_dot, z, z_dot, theta, theta_dot, phi, phi_dot, psi, psi_dot]

class KalmanFilter:
	
	def __init__(self, init_state, frequency, state_variance=100, measurement_variance=0.001, method="Velocity"):
		
		self.method = method # Model acceleration or not
        
		process_variance = 100 # Comes from control input U (so velocity for example)

		# Variances used in prediction
		self.stateVariance = state_variance
		self.measurementVariance = measurement_variance
		self.processVariance = process_variance
		self.U = 1 if method == "Acceleration" else 0
		
		dt = 1/frequency
		
		# State transition matrix, a kinematic model of state transition here
		self.A = np.matrix([
			[1, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 1, dt, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 1, dt, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 1, dt, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 1, dt, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, dt],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            
		])

		# Control input matrix, models influence of control input on state
		self.B = np.matrix([
			[dt**2/2],
			[dt],
			[dt**2/2],
			[dt],
			[dt**2/2],
			[dt],
			[dt**2/2],
			[dt],
			[dt**2/2],
			[dt],
			[dt**2/2],
			[dt]
		])
		
		# Observation model matrix,
		self.H = np.matrix([
			[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
			
		])

		# Observation noise covariance matrix
		self.R = np.matrix(self.measurementVariance*np.identity(self.H.shape[0]))

		# Process noise covariance matrix
		self.Q = self.B * np.diag(np.ones((1, 1)))* self.processVariance * self.B.T

		#TODO: self.R and self.Q do not change, make model!

		# Estimated covariance matrix
		self.P = np.matrix(self.stateVariance*np.identity(self.A.shape[0]))
		self.err_cov = self.P
		self.pred_err_cov = self.P

		# Ensure proper shape of state for calculation
		self.state = init_state.reshape(12, 1)
		self.pred_state = init_state.reshape(12, 1)



	def predict(self):
		# Predict next state and the next covariance matrix
		
		self.pred_state = self.A*self.state + self.B*self.U
		self.pred_err_cov = self.A*self.err_cov*self.A.T + self.Q



	def update(self, measurement):
		# Use measurement to refine prediction
		kalman_gain = self.pred_err_cov*self.H.T*np.linalg.pinv(self.H*self.pred_err_cov*self.H.T+self.R)

		self.state = self.pred_state + kalman_gain*(measurement - (self.H*self.pred_state))
		self.err_cov = (np.identity(self.P.shape[0]) - kalman_gain*self.H)*self.pred_err_cov

		print("measured state of x =   " + str(measurement[0])) 
		print("predicted state of x = " + str(self.pred_state[0]))
		print("updated state of x =   " + str(self.state[0]))
		print()



	def update_matrices(self, frequency):
		# For certain applications, the matrices should be updated to in order to
		# incorporate the processing time necessary for these (and potentially other)
		# steps.

		dt = 1/frequency

		# Update state transition matrix
		self.A = np.matrix([
			[1, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 1, dt, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 1, dt, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 1, dt, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 1, dt, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, dt],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            
		])

		# Update control input matrix
		self.B = np.matrix([
			[dt**2/2],
			[dt],
			[dt**2/2],
			[dt],
			[dt**2/2],
			[dt],
			[dt**2/2],
			[dt],
			[dt**2/2],
			[dt],
			[dt**2/2],
			[dt]
		])

		# Update observation matrix
		self.Q = self.B * self.B.T