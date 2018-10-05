import numpy as np
import networkx as nx

import gzip
import json
from datetime import datetime

from .errors import *

class AdaptiveVoter(object):
	"""
	Simulation class for the Adaptive Voter Model. 
	"""
	
	__version__ = 0.1


	def __init__(self, prebuilt=False, **kwargs):		
		""" 
		Input:
			kap (int/float): Node preferred degree. Take values in [0,infty)
			r (float): Probability of node update versus edge update. Takes values in [0,1].
			h (float): Homophily. Takes values in [-1,1].
			f (float): Neighbourhood agreement threshold. Takes values in [0,1].
			T (float): Temperature of the system. Takes values in [0,infty)
		
		To load a previously saved simulation use AdaptiveVoter.from_saved_simulation(filepath).
		"""

		if prebuilt:
			self.init_from_file(**kwargs)
		else:
			self.init_from_parameters(**kwargs)
		
		self._idx = 0			

	def init_from_parameters(self, **kwargs):
		"""
		Set up the simulation from parameter values.
		"""

		valid_parameters = ['kap', 'r', 'h', 'f', 'T']
		for key, val in kwargs.items():
			if key in valid_parameters:
				setattr(self, key, val)

		for parameter in valid_parameters:	
			if not hasattr(self,parameter):
				raise MissingInputError("Please specify parameter: {}".format(parameter))

		self.t = 0
		self.pointer = 0

		# Lists to store state (will be converted to arrays for efficiency)
		self.events = []

	def init_from_file(self, **kwargs):
		"""
		Set up the simulation using a prebuilt simulation.
		"""

		filepath = kwargs.get('filepath', None)

		if filepath.endswith('.gz'):
			with gzip.open(filepath, 'rb') as f:
				payload = json.loads(f.read().decode())
		else:
			with open(filepath, 'r') as f:
				payload = json.load(f)

		for key, val in payload['parameters'].items():
			setattr(self, key, val)
		self.events = payload['events']
		
		self.A = np.zeros(shape=(self.N, self.N))
		self.S = np.zeros(self.N)
		self.pointer = -1
		self.t = self.events[-1]['body'][-1]

		self.K = self.A.sum(axis=1)
		self.X = -self.S * (self.A @ self.S)

	def set_initial_network(self, G):
		"""
		Set the network for the simulation to run on.

		Input:
			G (nx.Graph or np.ndarray):

		Returns:
			None
		"""
		if isinstance(G, nx.Graph):
			A = nx.to_numpy_matrix(G)
			self.A = np.asarray(A)
		elif isinstance(G, np.ndarray) and G.shape[0] == G.shape[1]:
			self.A = G
		else:
			raise InvalidNetworkError("Enter a valid NetworkX Graph or an adjacency matrix.")

		self.N = self.A.shape[0]
		self.K = self.A.sum(axis=1)

	def set_inital_condition(self, S=None):
		"""
		Set the inital condition of node states.
		Node states should be either -1 or +1.
		This method must be called AFTER set_initial_network.

		Input: 
			S (np.array):

		Returns: 
			None
		"""

		if S is None:
			self.S = 2*np.random.randint(2, size=self.N)-1
		elif len(S) != self.N:
			raise InvalidInitialConditionError("Dimension mismatch with initial condition.")
		else:
			self.S = S

		self.X = - self.S * (self.A @ self.S)

		self._save_initial_state()
	
	def _save_initial_state(self):
		"""
		Saves the inital state of the simulation as a sequence of events.
		"""

		for a,b in zip(*self.A.nonzero()):
				if a < b:
					self.events.append({'type':'edge',
										'body':(a,b,0)})
		for ix, sign in enumerate(self.S):
				if sign == -1:
					self.events.append({'type':'node',
										'body':(ix,0)})

	@classmethod
	def with_erdos_renyi_network(cls, N, p, random_initial=False, **kwargs):
		"""
		Construct using an Erdos Renyi network.

		Input:
			N (int):
			p (float):
			random_initial (bool):
			**kwargs: Parameter values for specifying the simulation.
		"""

		G = nx.fast_gnp_random_graph(N=N, p=p)

		adaptive = cls(**kwargs)
		adaptive.set_initial_network(G)

		if random_initial:
			adaptive.set_inital_condition()

		return adaptive

	@classmethod
	def from_saved_simulation(cls, filepath):
		"""
		Load a previously run instance of the simulation.
		"""

		return cls(prebuilt=True, filepath=filepath)

	def build(self, t=0):
		"""
		Reconstruct the simulation at a given point.

		Input:

		Returns:
			None
		"""

		changed = False
		
		# Cannot build past what is simulated, need to run new iterations.
		if t > self.t or t=='max':
			t = self.t

		# Forwards building.
		if self.pointer <= t:
			end_ix = -1

			for ix, event in enumerate(self.events[self._idx:]):
				if event['body'][-1] < self.pointer:
					continue
				if event['body'][-1] > t:
					break
				
				changed = True
				end_ix = ix

				if event['type'] == 'node':
					self.S[event['body'][0]] *= -1
				else:
					u,v = event['body'][:2]
					self.A[u,v] = 1 - self.A[u,v]
					self.A[v,u] = 1 - self.A[v,u]
			self._idx = self._idx + end_ix + 1

		# Backwards build.
		else:
			end_ix = 0

			for ix, event in enumerate(reversed(self.events[:self._idx])):
				if event['body'][-1] > self.pointer:
					continue
				if event['body'][-1] < t:
					break

				changed=True
				end_ix = ix

				if event['type'] == 'node':
					self.S[event['body'][0]] *= -1
				else:
					u,v = event['body'][:2]
					self.A[u,v] = 1 - self.A[u,v]
					self.A[v,u] = 1 - self.A[v,u]

			self._idx = self._idx - (end_ix + 1)

		self.pointer = t
		self.K = self.A.sum(axis=1)
		self.X = -self.S * (self.A @ self.S)

		return changed

	def run_iteration(self):
		""" 
		Run a single iteration of the model. 

		Input: None

		Returns: None
		"""
		
		if self.pointer < self.t:
			raise Exception("Current view of simulation (pointer={}) is behind final state (t={}). Move pointer forward to proceed.".format(self.pointer, self.t))

		# Move time onwards
		self.t += 1
		self.pointer += 1
		
		# Generate random numbers
		a1,a2,a3,a4 = np.random.random(4)
		
		# 1. Choose a random node.
		focus = int(self.N*a1)
		
		# 2a. Node update.
		if a2 < self.r:
			if self.T == 0: 
				if self.X[focus] > self.f * self.K[focus]:
					self.S[focus] *= -1
					self.events.append({'type':'node', 'body':(focus, self.t)})
			elif a3 < (1 + np.exp((self.f*self.K[focus] - self.X[focus])/self.T))**(-1)  :
				self.S[focus] *= -1
				self.events.append({'type':'node', 'body':(focus, self.t)})
		
		# 2b. Link update. 
		else:
			if self.K[focus] > self.kap: # Cut link
				neighbours = tuple(set(np.nonzero(self.A[focus,:])[0]) - set({focus})) # Ugly code
				link = np.random.choice(neighbours)
				if a4 < 0.5*(1-self.h*self.S[focus]*self.S[link]):
					self.A[focus, link] = 0
					self.A[link, focus] = 0
					self.events.append({'type':'edge','body':(focus, link, self.t)})
				
			else: # Add link
				# An error appears here if kappa is below the network size.
				antineighbours = tuple(set(np.nonzero(1-self.A[focus,:])[0]) - set({focus})) # Ugly code
				link = np.random.choice(antineighbours)
				if a4 < 0.5*(1+self.h*self.S[focus]*self.S[link]):
					self.A[focus, link] = 1
					self.A[link, focus] = 1
					self.events.append({'type':'edge','body':(focus, link, self.t)})
					
		# For now, we'll do a full update, although we only need to update affected nodes.
		self.K = self.A.sum(axis=1)
		self.X = - self.S * (self.A @ self.S)
		self._idx = len(self.events)

	def run(self, iterations):
		"""
		Run the simulation for a set number of iterations.
	
		Input: 
			iterations (int): The number of iterations to run.
		Returns:
			None
		"""
		for _ in range(iterations):
			self.run_iteration()

	def save_to_file(self, filename, compressed=False):
		"""
		Save the simulation history and parameters to file (currently JSON format).

		Input:
			filename (str):
			compressed (bool):
		"""

		parameters = {key:getattr(self,key) for key in ['kap', 'r', 'h', 'f', 'T', 'N']}
		payload = {'parameters':parameters,
				   'events':self.events,
				   'time': datetime.now(),
				   'version':self.__class__.__version__}

		payload = json.dumps(payload,
							 sort_keys=True,
	                  		 indent=2,
							 cls=JSONEncoder)
		if compressed:
			with gzip.open(filename+'.gz', 'wb') as f:
				f.write(payload.encode())
		else:
			with open(filename, 'w') as f:
				f.write(payload)

	def to_graph(self):
		"""
		Create a NetworkX graph of the current simulation state.
		Node opinions are saved as 'opinion' in the node dictionary.
		"""

		G = nx.from_numpy_array(self.A)
		nx.set_node_attributes(G, {key:val for key,val in enumerate(self.S)}, 'opinion')
		return G

class JSONEncoder(json.JSONEncoder):
	"""Encoder to assist with converting types for safe saving."""

	def default(self, obj):
		if isinstance(obj, np.int64):
			return int(obj)
		if isinstance(obj, datetime):
			return obj.strftime('%D %T')
		return json.JSONEncoder.default(self, obj)