# Q-Learning with Dyna-Q

**Author**: Souptik Banerjee

## 1 Overview

A very simple implementation of the Q-Learning and Dyna-Q solutions to the reinforcement learning problem.
This implementation is tested on a 2D grid robot world. 



### 1.2 Test Environment
The navigation task takes place in a 10 x 10 grid world. The particular environment is expressed in a CSV file of integers, where the value in each position is interpreted as follows:
0: blank space.
1: an obstacle.
2: the starting location for the robot.
3: the goal location.
5: quicksand.