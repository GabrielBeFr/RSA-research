import numpy as np
import itertools
from collections import defaultdict
from tqdm import tqdm

class InfoJigsaw:
    def __init__(self, grid_size, objects, goal):
        self.grid_size = grid_size
        self.objects = objects
        self.goal = goal
        self.S = [set(), set()]
        self.generate_private_states()
        self.A = [set(), set()]
        self.generate_actions()
        self.pStateInitial = [{}, {}] # Initial state probabilities (uniform)
        self.initialize_initial_state_probs()

    def initialize_initial_state_probs(self):
        for agent in range(2):
            for state in self.S[agent]:
                self.pStateInitial[agent][state] = 1.0 / len(self.S[agent])

    def generate_private_states(self):
        letters = set([obj[3] for obj in self.objects])
        digits = set([obj[4] for obj in self.objects])
        for letter_assignment in itertools.product(letters, repeat=len(self.objects)):
             self.S[0].add(tuple(letter_assignment))
        for digit_assignment in itertools.product(digits, repeat=len(self.objects)):
             self.S[1].add(tuple(digit_assignment))

    def generate_actions(self):
        properties = set([obj[i] for obj in self.objects for i in range(3)])
        for prop in properties:
            for p in range(2): self.A[p].add(f"say_{prop}")
        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                for p in range(2): self.A[p].add(f"click_{row}_{col}")
        for p in range(2): self.A[p].add("end")

    def is_terminal(self, context):
        return any("click" in a for a in context)

    def compute_utility(self, context, agent, state_agent, state_other, correct_goal_util=100, wrong_goal_util=-100, action_cost=-50):
        utility = 0
        for _ in context:
             utility += action_cost 

        if self.is_terminal(context):
            click_action = [a for a in context if "click" in a][-1]
            row, col = map(int, click_action.split("_")[1:])
            clicked_object = next((obj for obj in self.objects if obj[2] == (row,col)), None)

            if clicked_object:  
                is_goal = (clicked_object[3], clicked_object[4]) == self.goal
                utility += correct_goal_util if is_goal else wrong_goal_util
        return utility


    def literal_states(self, agent, utterance):
        """Returns the set of states consistent with a literal interpretation of the utterance."""
        #  Implement the literal semantic mapping here.  This is crucial!
        # This is a placeholder – you MUST define the actual mapping for your game.
        # For example:
        if utterance.startswith("say_"):
            property_value = utterance.split("_")[1]
            consistent_states = set()
            for state in self.S[agent]:
                for i, obj in enumerate(self.objects):
                    if agent == 0 and obj[0] == property_value or obj[1] == property_value: # check with the letter view if the property exists
                        consistent_states.add(state)
                        break
                    if agent == 1 and str(obj[2][0]) == property_value or str(obj[2][1]) == property_value or str(obj[4]) == property_value : # check with the digit view if the property exists
                        consistent_states.add(state)
                        break
            return consistent_states
        else:  # For click or end actions, all states are consistent (doesn't affect meaning).
             return set(self.S[agent]) 
            



    def makeContextNice(self, context):
        return str([self.A[i%2].get(context.get(i)).getName() for i in range(context.size())])



class PIP:
    def __init__(self, game, max_k, alpha, lookahead, history_length, smoothing=0.01): 
        self.game = game
        self.max_k = max_k
        self.alpha = alpha
        self.lookahead = lookahead
        self.history_length = history_length
        self.cache = {}
        self.smoothing = smoothing

    def infer(self, agent, context, state_agent, depth, k):
        if (agent, tuple(context), state_agent) in self.cache:
            return self.cache[(agent, tuple(context), state_agent)]

        if k == 0:  # Level-0 inference
            probs = {}
            consistent_states = self.game.literal_states(1 - agent, context[-1] if context else "")
            for state in self.game.S[1 - agent]:
                probs[state] = 1.0 / len(consistent_states) if state in consistent_states else 0.0
        else:  # Level-k inference (recursive)
            probs = defaultdict(float)
            #print("infer > Infering...")
            print(len(self.game.S[1 - agent]))
            for state_other in tqdm(self.game.S[1 - agent],desc=f"Infering; Agent: {agent}, Depth: {depth}, Context: {context}, k: {k}"):
                if context:
                    #print("infer > Planning...")
                    action_probs_other = self.plan(1 - agent, context[:-1], state_other, depth, k=k-1)
                    #print("infer > Planning finished.")
                    last_action = context[-1]
                    if last_action in action_probs_other: # Handle cases where the observed action is outside the other agent's vocabulary/actions.
                      probs[state_other] = action_probs_other[last_action]
                    else:
                      probs[state_other] = 0.0 #give probability 0 if not a valid option
            #print("infer > Infering finished.")


        self.cache[(agent, tuple(context), state_agent)] = probs
        return self.normalize_probs(probs)


    def plan(self, agent, context, state_agent, depth=None, k=None):
        if depth is None: 
            depth = self.lookahead
            do_tqdm = True
        if depth == 0 : return {action: 0.0 for action in self.game.A[agent]}

        if k is None: 
            k = self.max_k
            do_tqdm = True
        else: do_tqdm = False

        action_utilities = defaultdict(float)
        #print("plan > Infering...")
        beliefs_other = self.infer(agent, context, state_agent, depth, k)
        #print("plan > Infering finished.")
        expected_utility = 0

        #print(f"self.game.A[agent]: {self.game.A[agent]}")
        for action in self.game.A[agent]:
            next_context = context + [action]
            #print(f"Agent: {agent}, Depth: {depth}, Context: {context}, Action: {action}")
            for state_other, prob in beliefs_other.items():
                if self.game.is_terminal(next_context):
                    #print("Action is terminal")
                    u = self.game.compute_utility(next_context, agent, state_agent, state_other)
                else:
                    #print("plan > Partner is planning...")
                    partner_action_probs = self.plan(agent=1-agent, context=next_context, state_agent=state_other, depth=depth-1, k=k)
                    #print("plan > Partner planning finished.")
                    u = sum(p*self.game.compute_utility(next_context + [a], agent, state_agent, state_other)  
                                                for a,p in partner_action_probs.items()) # Simplified utility for now
                expected_utility += prob * u
            action_utilities[action] = expected_utility

        return self.softmax(action_utilities)

    def act(self, agent, context, state_agent):
        print("********************")
        print("Agent acting: ", agent)
        action_probs = self.plan(agent, context, state_agent)
        print("Choosing an action...")
        actions, probs = zip(*action_probs.items())
        chosen_action = np.random.choice(actions, p=probs)
        print("Chosen action: ", chosen_action)
        print("********************")
        return chosen_action

    def softmax(self, action_utilities):
        # Softmax with smoothing to ensure all actions have some probability
        utilities = np.array(list(action_utilities.values()))
        utilities = np.exp(self.alpha * utilities)
        utilities = (utilities + self.smoothing) / (np.sum(utilities) + self.smoothing* len(utilities))

        return dict(zip(action_utilities.keys(), utilities))
    

    def normalize_probs(self, probs):
        total_prob = sum(probs.values())
        if total_prob > 0:
            return {state: prob / total_prob for state, prob in probs.items()}
        else:  # Handle cases where all probabilities are 0 (e.g., inconsistent context)
            return {state: 1/len(probs) for state in probs} # Uniform if all zero
        

# # Example usage:

# grid_size = (2, 3)
# objects = [
#     ("square", "blue", (0, 0), "A", 1),
#     ("circle", "yellow", (0, 1), "B", 2),
#     ("diamond", "green", (0, 2), "C", 3),
#     ("square", "yellow", (1, 0), "B", 1),
#     ("circle", "blue", (1, 1), "A", 2),
#     ("diamond", "green", (1, 2), "C", 1)
# ]
# goal = ("B", 2)

# game = InfoJigsaw(grid_size, objects, goal)
# pip_agent = PIP(game, k=1, alpha=10, lookahead=2, history_length=float('inf'))



# context = []
# agent = 0
# state_agent = next(iter(game.S[0])) # An example initial private state – replace as needed.

# action = pip_agent.act(agent, context, state_agent)

# #... (Game loop)