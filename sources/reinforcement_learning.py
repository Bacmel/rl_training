import numpy as np
from toolbox import softmax
from maze import build_maze

from dynamic_programming import policy_iteration_q, get_policy_from_q
from toolbox import egreedy_loc

# -------------------------------------------------------------------------------------#
# Given state and action spaces and a policy, computes the state value of this policy


def temporal_difference(mdp, pol, nb_episodes=50, alpha=0.2, timeout=25, render=True):
    # alpha: learning rate
    # timeout: timeout of an episode (maximum number of timesteps)
    v = np.zeros(mdp.nb_states)  # initial state value v
    mdp.timeout = timeout

    if render:
        mdp.new_render()
    
    for _ in range(nb_episodes):  # for each episode
        
        # Draw an initial state randomly (if uniform is set to False, the state is drawn according to the P0 
        #                                 distribution)
        x = mdp.reset(uniform=True) 
        done = mdp.done()
        while not done:  # update episode at each timestep
            # Show agent
            if render:
                mdp.render(v, pol)
            
            # Step forward following the MDP: x=current state, 
            #                                 pol[i]=agent's action according to policy pol, 
            #                                 r=reward gained after taking action pol[i], 
            #                                 done=tells whether the episode ended, 
            #                                 and info gives some info about the process
            [y, r, done, _] = mdp.step(egreedy_loc(pol[x], mdp.action_space.size, epsilon=0.2))
            
            # Update the state value of x
            if x in mdp.terminal_states:
                v[x] = # TODO : fill this
            else:
                delta = # TODO : fill this
                v[x] = # TODO : fill this
            
            # Update agent's position (state)
            x = y
    
    if render:
        # Show the final policy
        mdp.current_state = 0
        mdp.render(v, pol)
    return v


# --------------------------- Q-Learning -------------------------------#

# Given a temperature "tau", the QLearning function computes the state action-value function
# based on a softmax policy
# alpha is the learning rate

def q_learning(mdp, tau, nb_episodes=20, timeout=50, alpha=0.5, render=True):
    # Initialize the state-action value function
    # alpha is the learning rate
    q = np.zeros((mdp.nb_states, mdp.action_space.size))

    # Run learning cycle
    mdp.timeout = timeout  # episode length

    if render:
        mdp.new_render()

    for _ in range(nb_episodes):
        # Draw the first state of episode i using a uniform distribution over all the states
        x = mdp.reset(uniform=True)
        done = mdp.done()
        while not done:
            if render:
                # Show the agent in the maze
                mdp.render(q, q.argmax(axis=1))

            # Draw an action using a soft-max policy
            u = mdp.action_space.sample(prob_list=softmax(q, x, tau))

            # Perform a step of the MDP
            [y, r, done, _] = mdp.step(u)

            # Update the state-action value function with q-Learning
            if x in mdp.terminal_states:
                q[x, u] = # TODO : fill this
            else:
                delta = # TODO : fill this
                q[x, u] = # TODO : fill this

            # Update the agent position
            x = y

    if render:
        # Show the final policy
        mdp.current_state = 0
        mdp.render(q, get_policy_from_q(q))
    return q


# --------------------------- Q-Learning -------------------------------#

# Given a temperature "tau", the QLearning function computes the state action-value function
# based on a softmax policy
# alpha is the learning rate

def sarsa(mdp, tau, nb_episodes=20, timeout=50, alpha=0.5, render=True):
    # Initialize the state-action value function
    # alpha is the learning rate
    q = np.zeros((mdp.nb_states, mdp.action_space.size))

    # Run learning cycle
    mdp.timeout = timeout  # episode length

    if render:
        mdp.new_render()

    for _ in range(nb_episodes):
        # Draw the first state of episode i using a uniform distribution over all the states
        x = mdp.reset(uniform=True)
        done = mdp.done()

        # Draw an action using a soft-max policy
        u = mdp.action_space.sample(prob_list=softmax(q, x, tau))
        while not done:
            if render:
                # Show the agent in the maze
                mdp.render(q, q.argmax(axis=1))

            # Perform a step of the MDP
            [y, r, done, _] = mdp.step(u)

            # Update the state-action value function with q-Learning
            if x in mdp.terminal_states:
                q[x, u] = # TODO : fill this
            else:
                # Draw an action using a soft-max policy
                u2 = mdp.action_space.sample(prob_list=softmax(q, y, tau))
                delta = # TODO : fill this
                q[x, u] = # TODO : fill this
                u = u2

            # Update the agent position
            x = y

    if render:
        # Show the final policy
        mdp.current_state = 0
        mdp.render(q, get_policy_from_q(q))
    return q


# --------------------------- run it -------------------------------#

def run_rl():
    walls = [5, 6, 13]
    height = 4
    width = 5
    m = build_maze(width, height, walls, hit=True)

    q = policy_iteration_q(m, render=False)
    pol = get_policy_from_q(q)
    print("TD-learning")
    temporal_difference(m, pol, render=True)
    print("Q-learning")
    q_learning(m, tau=6)
    print("Sarsa")
    sarsa(m, tau=6)
    input("press enter")


if __name__ == '__main__':
    run_rl()
