# README

## Algorithm

Following the provided hint, we chose to use CMA-ES to optimize the parameters of a neural network controller. CMA-ES is a derivative-free evolutionary algorithm well-suited for non-linear control problems. It is also well adapted to small networks like the one used in our case. At each generation, a population of candidate solutions is sampled from a Gaussian distribution. Individuals are evaluated based on their fitness, and the distribution is updated to favor high-performing regions of the search space. The algorithm progressively learns which directions are promising and which combinations of parameters work well.

## Environment Design

The environment is based on a MuJoCo simulation of an Ant robot on flat terrain. The observation consists of joint positions (excluding global x,y) and joint velocities, making the task translation-invariant. The basic reward function includes three components: forward velocity (to encourage locomotion), a survival reward (to promote stability), and a control cost (to penalize excessive actuator usage). After some experiments, we decided to also penalize lateral velocity to encourage straight movement. The simulation terminates when the torso height is outside a predefined range.

## Controller

The controller is a feedforward neural network with one hidden layer. The network maps observations to actuator commands using tanh activations, ensuring outputs remain in the range [-1, 1]. The parameters of the network (weights) are directly optimized by CMA-ES. This simple architecture allows the emergence of coordinated locomotion behaviors without manual feature engineering. Using a kind of CPG strategy with oscillators. The training takes part in multiple sessions. The first one with easier rules and criteria in the reward function and a second and a third part trying to make our ant going forward by penalizing the y-velocity and the rotation around the yaw axis.