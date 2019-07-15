from MDP import MDP
from math import inf

class BellmanDPSolver(object):

	def __init__(self, discountRate=1.0):
		self.MDP = MDP()
		self.discountRate = discountRate

		self.V = None
		self.Policy = None

	def initVs(self):

		# initialize state values to zero
		self.V = {s:0 for s in self.MDP.S}

		 # initialize to uniform policy (so all actions are 'optimal')
		self.Policy = {s: self.MDP.A for s in self.MDP.S}

	def BellmanUpdate(self):

		V_new = {s:0 for s in self.MDP.S}  # make updates in a new value function (i.e. not in place)

		for s in self.V.keys():  # loop through all the states

			max_value= -inf

			for a in self.MDP.A:  # loop through all actions

				estimated_value = 0
				next_states = self.MDP.probNextStates(s, a)  # get next state probabilities

				for next_state in next_states:
					probability = next_states[next_state]
					reward = self.MDP.getRewards(s, a, next_state)  # next state reward
					estimated_value += probability * (reward + self.discountRate * self.V[next_state])

				# if s == (3,2):
					# print("Value of {} from (3,2): {}".format(a, estimated_value))

				if estimated_value > max_value:
					# we've found a new best optimal action
					self.Policy[s] = [a]  # update policy
					max_value = estimated_value
				elif estimated_value == max_value:
					# we've found another optimal action with same value
					self.Policy[s].append(a)  # update policy

			# print()
			V_new[s] = max_value  # update value function estimate

		self.V = V_new
		return (self.V, self.Policy)

	def pretty_print(self):

          'Pretty print in grid with (0,0) as the top left corner'

          states = [(x,y) for x in range(5) for y in range(5)]
          print("\nState Values")
          for counter, (y, x) in enumerate(states):
              print("{:+.4f}  ".format(self.V[(x,y)]), end = '')

              if((counter+1)%5==0 and counter!=0):
                  print("")
          print("\n State Policies")
          for counter, (y,x) in enumerate(states):
              print("{:25} ".format(', '.join(self.Policy[(x,y)])),end='')
              if((counter+1)%5==0 and counter!=0):
                  print("")

if __name__ == '__main__':
	solution = BellmanDPSolver()
	solution.initVs()
	for i in range(10000):
	# for i in range(1000):
		values, policy = solution.BellmanUpdate()
	print("Values : ", values)
	print("Policy : ", policy)

	print()
	solution.pretty_print()
