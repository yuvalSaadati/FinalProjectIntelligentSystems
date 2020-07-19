# name : Yuval Saadati
# id: 205956634
import os
import random
import json
import pickle
from collections import defaultdict, deque
from valid_actions import ValidActions
from valid_actions import PythonValidActions
amount_actions_return = 0
list_visit = {}
first_state = None
pickl_first = True
hash_dict = {}
# save the last action
last_return_action = ""
# save the last state
last_state = None
# save the Q-table in dictionary
q_table = None
# save the best policy in dictionary
policy_table = {}
last_goals_amount = 0
pickle_index = 0
pickle_list = list()
goals_files_name = list()
punish_state_repeated = False
graphBFS = None


from collections import defaultdict
class Graph():
    # Using dijsktra algorithm to find the shortest path in graph
    # From benalexkeen.com
    def __init__(self):
        """
        self.edges is a dict of all possible next nodes
        e.g. {'X': ['A', 'B', 'C', 'E'], ...}
        self.weights has all the weights between two nodes,
        with the two nodes as a tuple as the key
        e.g. {('X', 'A'): 7, ('X', 'B'): 2, ...}
        """
        self.edges = defaultdict(list)
        self.weights = {}
        self.actions = {}

    def add_edge(self, from_node, to_node, weight, action):
        # Note: assumes edges are bi-directional
        self.edges[from_node].append(to_node)
        self.weights[(from_node, to_node)] = weight
        self.actions[(from_node, to_node)] = action
graph = Graph()

class Executor(object):
    def __init__(self, arg):
        super(Executor, self).__init__()
        # -L is for learning and -E is for execute the optimal policy
        self.arg = arg
    def initialize(self, services):
        global first_state
        self.services = services
        self.valid_actions_options = ValidActions(self.services.parser, self.services.pddl, self.services.perception)
        self.python_valid_actions_options = PythonValidActions(self.services.parser, self.services.perception)
        self.graph_file_E = self.services.parser.problem_name + self.services.parser.domain_name

    def createBFSgraph(self):
        # create graph to run on it BFS search
        global graph
        global graphBFS
        graphBFS = {}
        for key, value in graph.edges.items():
            graphBFS[key] = []
            for vertex in value:
                if vertex not in graphBFS[key]:
                    graphBFS[key].append(vertex)
        return graphBFS

    def dijsktra(self, graph, initial, end):
        # shortest paths is a dict of nodes
        # whose value is a tuple of (previous node, weight)
        shortest_paths = {initial: (None, 0)}
        current_node = initial
        visited = set()

        while current_node != end:
            visited.add(current_node)
            destinations = graph.edges[current_node]
            weight_to_current_node = shortest_paths[current_node][1]

            for next_node in destinations:
                weight = graph.weights[(current_node, next_node)] + weight_to_current_node
                if next_node not in shortest_paths:
                    shortest_paths[next_node] = (current_node, weight)
                else:
                    current_shortest_weight = shortest_paths[next_node][1]
                    if current_shortest_weight > weight:
                        shortest_paths[next_node] = (current_node, weight)

            next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
            if not next_destinations:
                return "Route Not Possible"
            # next node is the destination with the lowest weight
            current_node = min(next_destinations, key=lambda k: next_destinations[k][1])

        # Work back through destinations in shortest path
        path = []
        while current_node is not None:
            path.append(current_node)
            next_node = shortest_paths[current_node][0]
            current_node = next_node
        # Reverse path
        path = path[::-1]
        return path

    def action_made_impact(self):
        # check if the last action did not take place
        global last_return_action
        state = self.services.perception.get_state()
        actions = self.valid_actions_options.get(state)
        for action in actions:
            if action == last_return_action:
                # if the action is valid so the agent in the same state
                return False
        return True

    def max_action_Qvalue(self, state):
        # return the highest value in a particular state from Q-table
        global q_table
        actions = self.valid_actions_options.get(state)
        max_value = 0
        flag = True
        if self.string_state(state) not in q_table.keys():
            q_table[self.string_state(state)] = {}
            return 0
        actions_dic = q_table[self.string_state(state)]
        for action in actions:
            for key, value in actions_dic.iteritems():
                if flag:
                    flag = False
                    max_value = value
                if value >= max_value:
                    if key == action:
                        max_value = value
        return max_value

    def build_table(self):
        # create Q-table for learning
        global q_table
        state = self.services.perception.get_state()
        actions = self.valid_actions_options.get(state)
        state_name = self.string_state(state)
        if q_table is None:
            # initial the table
            q_table = {"epsilon": 0.1, "alpha": 0.1, "gamma": 0.6 }
            q_table[state_name] = {}
            for action in actions:
                q_table[state_name][action] = 0
        else:
            if state_name not in q_table.keys():
                q_table[state_name] = {}
                for action in actions:
                    # initial all Q-values with 0
                    q_table[state_name][action] = 0
            else:
                actions_dic = q_table[state_name]
                for action in actions:
                    if action not in actions_dic.keys():
                        # initial all Q-values with 0
                        q_table[state_name][action] = 0

    def max_action_Qname(self, state):
        # returns the action name with the highest value
        global q_table
        actions = self.valid_actions_options.get(state)
        action_name = ""
        max_value = 0
        flag = True
        if self.string_state(state) not in q_table.keys():
            return random.choice(actions)
        actions_dic = q_table[self.string_state(state)]
        for key, value in actions_dic.iteritems():
            if flag:
                flag = False
                max_value = value
                action_name = random.choice(actions)
            if value >= max_value:
                if key in actions:
                    max_value = value
                    action_name = key
        return action_name

    def string_state(self, state):
        # return the state name
        global hash_dict, q_table, pickle_index, pickle_list, punish_state_repeated
        for pic in pickle_list:
            data = pickle.load(open(pic, "rb"))
            if data == state:
                # this state exists
                punish_state_repeated = True
                return pic
        pickle_index += 1
        pickle_list.append(str(pickle_index))
        pickle.dump(state, open(str(pickle_index), "wb"))
        return str(pickle_index)

    def reward_function(self):
        # reward the agent according to last action
        global last_goals_amount, pickle_index, punish_state_repeated
        # list of all sub goals
        goals = list()
        count_goals = 0
        for goal in self.services.goal_tracking.uncompleted_goals:
            goals.extend([part.test(self.services.perception.get_state()) for part in goal.parts])
        for sub_goals in goals:
            if sub_goals is True:
                # amount of current goals
                count_goals += 1
        if last_goals_amount > count_goals:
            # last action make the agent decrease sub-goal
            last_goals_amount = count_goals
            if punish_state_repeated:
                # the agent was already in the current state
                punish_state_repeated = False
                return -20
            return -10
        elif last_goals_amount < count_goals:
            # last action make the agent increase sub-goal
            last_goals_amount = count_goals
            goals_files_name.append(self.string_state(last_state))
            return 100
        else:
            last_goals_amount = count_goals
            if punish_state_repeated:
                # the agent was already in the current state
                punish_state_repeated = False
                return -5
            return -0.1

    def find_shortest_pathBFS(self, graphBFS, start, end):
        # bfs search on graph
        dist = {start: [start]}
        q = deque(start)
        while len(q):
            at = q.popleft()
            if at == "g":
                break
            if at in graphBFS.keys():
                for next in graphBFS[at]:
                    if next not in dist:
                        if at in dist.keys():
                            dist[next] = [dist[at], next]
                            q.append(next)
        new_str = ""
        test_str = str(dist.get(end)).replace("[", "")
        for i in range(len(test_str)):
            if test_str[i] != ']' and test_str[i] != '[' and test_str[i] != "'" and test_str[i] != ","  :
                new_str = new_str + test_str[i]
        return new_str

    def next_action(self):
        # Return the next action to apply
        global graph, pickle_index, last_return_action, last_state, q_table
        global graphBFS, list_visit, amount_actions_return, policy_table, goals_files_name
        amount_actions_return += 1
        q_table_file = self.services.parser.domain_name + ".json"
        # current state
        state = self.services.perception.get_state()
        # all possible actions
        actions = self.valid_actions_options.get(state)
        # the name of the state file
        state_name = self.string_state(state)
        # the name of the last state file
        last_state_name = self.string_state(last_state)
        # create q table file named by domain name
        if self.arg == "-L":
            if self.services.goal_tracking.reached_all_goals():
                # agent reached all goals
                # save the state where all goals reached
                if "files index" in q_table.keys():
                    if pickle_index > q_table["files index"] :
                        q_table["files index"] = pickle_index
                else:
                    q_table["files index"] = pickle_index
                # save q-table in a file in order to use the table in the next running
                qFile = open(q_table_file, "w+")
                qFile.write(json.dumps(q_table))
                qFile.close()
                if last_state_name != state_name:
                    prob = q_table[last_state_name][last_return_action]
                    if prob < 0:
                        prob *= -1
                    else:
                        prob *= 3
                    # adding new edg to graph
                    edg = (last_state_name, "g", prob, last_return_action)
                    graph.add_edge(*edg)
                # saving graph in a file in order to use the graph in execution
                pickle.dump(graph, open(self.graph_file_E, "wb"))
                return None

            if last_state is None:
                try:
                    # this is the first iteration of the agent but the Q-table created before
                    with open(q_table_file, 'rb') as file:
                        if self.services.parser.domain_name in self.services.parser.problem_name:
                            q_table = json.load(file)
                        else:
                            # file not exists, create new q table file
                            self.build_table()
                            # update Q-table
                            with open(q_table_file, 'w+') as file3:
                                file3.write(json.dumps(q_table))
                        random_action = random.choice(actions)
                        last_return_action = random_action
                        last_state = state
                        # create the first edge in graph
                        edg = ("0", "1", 1, random_action)
                        graph.add_edge(*edg)
                        self.build_table()
                        list_visit[random_action] = 1
                        return random_action
                except IOError:
                    # the is the first iteration of learning
                    first = True
                if first:
                    q_table = {"epsilon": 0.1, "alpha": 0.1, "gamma": 0.6, "files index":0}
                    q_table[state_name] = {}
                    for action in actions:
                        q_table[state_name][action] = 0
                    # build for the first time Q-table
                    self.build_table()
                    random_action = random.choice(actions)
                    # update Q-table
                    with open(q_table_file, 'w+') as file3:
                        file3.write(json.dumps(q_table))
                    last_return_action = random_action
                    last_state = state
                    # create the first edge in graph
                    edg = ("0", "1", 1, random_action)
                    graph.add_edge(*edg)
                    list_visit[random_action] = 1
                    return random_action

            else:
                # not the first iteration so Q-table exists
                with open(q_table_file) as f:
                    q_table = json.load(f)
                    self.build_table()
                f.close()
            if random.uniform(0, 1) < q_table["epsilon"]:
                self.build_table()
                # choose random action
                random_action = random.choice(actions)
                # get reward by the random action
                reward = self.reward_function()
                old_value = q_table[last_state_name][last_return_action]
                next_max = self.max_action_Qvalue(state)
                #if amount_actions_return > 30:
                #    q_table["alpha"] += 0.1
                #    if q_table["alpha"] > 0.9:
                #        # maximum value of alpha will be 0.9
                #        q_table["alpha"] = 0.9
                #    amount_actions_return = 0
                alpha = q_table["alpha"]
                gamma = q_table["gamma"]
                new_value = (1 - alpha) * old_value + (gamma * next_max)
                q_table[state_name][random_action] = new_value
                q_table[last_state_name][last_return_action] += reward * alpha
                if random_action in list_visit.keys():
                    # punish the agent for taking the same action more the 2 times
                    if list_visit[random_action] > 2 :
                        q_table[last_state_name][last_return_action] += -10 * alpha
                    else:
                        list_visit[random_action] += 1
                else:
                    list_visit[random_action] = 1
                if not self.action_made_impact():
                    # punish agent because the last action did not make an impact
                    q_table[last_state_name][last_return_action] += alpha * -0.5

                jsonFile = open(q_table_file, "w+")
                jsonFile.write(json.dumps(q_table))
                jsonFile.close()
                self.build_table()
                # no self loop in graph
                if last_state_name != state_name:
                    # the edge wight is the Q-value
                    prob = q_table[last_state_name][last_return_action]
                    if prob < 0:
                        prob *= -1
                    else:
                        prob *= 3
                    # adding new edg to graph
                    edg = (last_state_name, state_name, prob, last_return_action)
                    if (last_state_name, state_name) in graph.weights.keys():
                        if prob < graph.weights[(last_state_name, state_name)] :
                            # taking the edge with the lowest wight
                            graph.add_edge(*edg)
                    else:
                        graph.add_edge(*edg)
                # save the last state
                last_state = state
                # save the last action
                last_return_action = random_action
                return random_action
            else:
                self.build_table()
                # get the action with the highest value
                action = self.max_action_Qname(state)
                reward = self.reward_function()
                old_value = q_table[last_state_name][last_return_action]
                next_max = self.max_action_Qvalue(state)
                #if amount_actions_return > 30:
                #    q_table["alpha"] += 0.1
                #    if q_table["alpha"] > 0.9:
                #        # maximum value of alpha will be 0.9
                #        q_table["alpha"] = 0.9
                #    amount_actions_return = 0
                alpha = q_table["alpha"]
                gamma = q_table["gamma"]
                new_value = (1 - alpha) * old_value + alpha * ( gamma * next_max)
                q_table[state_name][action] = new_value
                q_table[last_state_name][last_return_action] += reward * alpha
                if action in list_visit.keys() :
                    if list_visit[action] > 2 :
                        # punish the agent for taking the same action more the 2 times
                        q_table[last_state_name][last_return_action] += -10 * alpha
                    else:
                        list_visit[action] +=1
                else:
                    list_visit[action] = 1
                if not self.action_made_impact():
                    q_table[last_state_name][last_return_action] += alpha * -0.5
                # saving q-table into file
                jsonFile = open(q_table_file, "w+")
                jsonFile.write(json.dumps(q_table))
                jsonFile.close()
                if last_state_name != state_name:
                    # no self loop in graph
                    prob = q_table[last_state_name][last_return_action]
                    if prob < 0:
                        prob *= -1
                    else:
                        prob *= 3
                        # adding new edg to graph
                    edg = (last_state_name, state_name, prob, last_return_action)
                    if (last_state_name, state_name) in graph.weights.keys():
                        if prob < graph.weights[(last_state_name, state_name)]:
                            # taking the edge with the lowest wight
                            graph.add_edge(*edg)
                    else:
                        graph.add_edge(*edg)

                last_state = state
                last_return_action = action
                return action
        elif self.arg == "-E":
            # load q-table from file
            with open(q_table_file) as f:
                q_table = json.load(f)
            if self.services.goal_tracking.reached_all_goals():
                for i in range(1, q_table["files index"]):
                    os.remove(str(i))
                return None
            # load graph from file
            graph = pickle.load(open(self.graph_file_E, "rb"))

            if graphBFS is None:
                graphBFS = self.createBFSgraph()
            path = self.find_shortest_pathBFS(graphBFS, state_name, "g")
            path = path.split()
            flag_firsr_state = True
            actions_string = ""
            last_p = ""
            for p in path:
                if flag_firsr_state:
                    last_p = p
                    flag_firsr_state = False
                    continue
                if not flag_firsr_state:
                    if (last_p, p) in graph.actions.keys():
                        actions_string += graph.actions[(last_p, p)]
                        last_p = p
            print actions_string
            first_v = ""
            second_v = ""
            flag_second = False
            for i in range(len(path)):
                p = path[i]
                if not flag_second:
                    first_v = path[i]
                    flag_second = True
                    continue
                if flag_second:
                    second_v = path[i]
                    break
            if (first_v, second_v) in graph.weights.keys():
                if graph.actions[(first_v, second_v)] in actions:
                    return graph.actions[(path[0], path[1])]
                else:
                    return random.choice(actions)
            else:
                return random.choice(actions)
            """

            path = self.dijsktra(graph, state_name, "g" )
            if path != "Route Not Possible" and graph.actions[(path[0], path[1])] in actions:
                return graph.actions[(path[0], path[1])]
            else:
                return random.choice(actions)
 """
        else:
            # failed to choose action
            if self.services.goal_tracking.reached_all_goals():
                return None
            state = self.services.perception.get_state()
            actions = self.valid_actions_options.get(state)
            return random.choice(actions)
