import numpy as np

class HMMTrain():
	"""
	private: 
		self.transition: A matrix
		self.observation: O matrix
		self.state: state vector
		self.words: words vector
	public:
		forward-Backwards: E step
		Mstep:
		train_unsupervised:

	"""

	def __init__(self, A, O, states, words):
		self.transition = A
		self.observation = O
		self.state = states
		self.words = words

#obs = x ; transfer = transition, emit = observation
	def Forward(self, x, start_prob):
		V = np.zeros((len(self.transition),len(x)))
		path = [str(0) for i in range(len(self.transition))]
		newpath = [str(0) for i in range(len(self.transition))]
    
		#initialize
		for y in range(len(self.transition)):
			V[y][0] = start_prob[y] * self.observation[y][x[0]]
 
		#Viterbi
		for t in range(1, len(x)):
			for y in range(len(self.transition)):
				prob = sum(V[y0][t-1]*self.observation[y][x[t]]*self.transition[y0][y] for y0 in range(len(self.transition)))
				V[y][t] = prob
			Alpha = V[:,V.shape[1]-1]    	
		return Alpha


	def Backward(self,x):
		V = np.zeros((len(self.transition),len(x)))
		path = [str(0) for i in range(len(self.transition))]
		newpath = [str(0) for i in range(len(self.transition))]

		#initialize
		for y in range(len(self.transition)):
			V[y][len(x)-1] =  1

		#Viterbi
		for t in range(len(x)-2,-1,-1):
			for y in range(len(self.transition)):
				prob = sum(V[y0][t+1]*self.observation[y0][x[t+1]]*self.transition[y][y0] for y0 in range(len(self.transition)))
				V[y][t] = prob
			Beta = V[:,0]
		return Beta




