# EECS-738_Proj04


# EECS-738_Proj03

In this project, I implemented a Treasure Hunters game to find the optimal path to get treasure while avoiding obstacles and opponents by using reinforcement learning. The enviornment map includes rocks, monsters, treasure, and blank road.

# Project Instruction
We do the treasure hunting and monster fighting for you  
1. Set up a new git repository in your GitHub account   
2. Think up a map-like environment with treasure, obstacles and opponents   
3. Choose a programming language (Python, C/C++, Java)   
4. Formulate ideas on how reinforcement learning can be used to find treasure efficiently while avoiding obstacles and opponents   
5. Build one or more reinforcement policies to model situational assessments, actions and rewards programmatically   
6. Document your process and results    
7. Commit your source code, documentation and other supporting files to the git repository in GitHub   

# Reinforcement Learning - Q-Learning

I used Q-learning startegy for my reinforcement learning algorithm

## Reward table

In the reward table, each step the hunters move will have a cost except treasure cell, the rock cell and monster cell will have more cost.

Blank road = -1   
Rock = -5  
Monster = -10  
Treasure = +10

## Action

The hunter has four actions at each state: UP, DOWN, LEFT, RIGHT  
If the hunter at the edge of the map and trying to move out of the map, then hunter will stay in current state, but will still have cost.

## Q Table

My Q-table is a n by p matrix based on map size, each element in matrix is a state of hunter and contains 4 Q vlaues for 4 action

# Results

After implementing the reinforcement learning, the model is able to find the best path to treasure and avoid rocks and monsters. And by the testing, the model work for different entry point and different map as well 


# References
https://medium.com/@curiousily/solving-an-mdp-with-q-learning-from-scratch-deep-reinforcement-learning-for-hackers-part-1-45d1d360c120  
https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/  
