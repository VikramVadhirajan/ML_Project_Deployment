# Reinforcement Learning

Reinforcement Learning (RL) is a machine learning paradigm where an **agent learns to make decisions by interacting with an environment** and receiving feedback in the form of **rewards or penalties**.

The objective of the agent is to **learn a policy that maximizes cumulative reward over time**.

Reinforcement learning differs from:

- [[Supervised Learning]] → learning from labeled examples
    
- [[Unsupervised Learning]] → discovering patterns in unlabeled data
    

In RL, learning happens through **trial and error interactions**.

---

# Core Components

Reinforcement learning systems consist of several key elements.

---

## Agent

The **decision maker** that interacts with the environment.

Examples:

- robot
    
- self-driving car
    
- game-playing AI
    

---

## Environment

The system or world with which the agent interacts.

The environment provides:

- new states
    
- rewards
    

Examples:

- a video game
    
- a robotics simulation
    
- a financial market
    

---

## State (S)

The state represents the **current situation of the environment**.

Example:

In chess, the state is the **current board configuration**.

---

## Action (A)

An action is a **decision taken by the agent**.

Example:

In a game:

- move left
    
- move right
    
- jump
    

---

## Reward (R)

A reward is the **feedback signal** returned by the environment after an action.

Examples:

+10 → successful action  
-5 → undesirable action

The agent tries to **maximize cumulative rewards**.

---

## Policy (π)

A policy defines **how the agent selects actions given a state**.

π(a | s)

Meaning:

Probability of taking action **a** in state **s**.

The goal of reinforcement learning is to **learn an optimal policy**.

---

# Interaction Cycle

The reinforcement learning process follows a loop:

1. Agent observes current state
    
2. Agent chooses an action
    
3. Environment transitions to a new state
    
4. Environment returns reward
    
5. Agent updates its strategy
    

This cycle repeats many times.

---

# Return (Cumulative Reward)

The objective is not just immediate reward but **long-term cumulative reward**.

Return is defined as:

G = r₁ + γr₂ + γ²r₃ + ...

Where:

γ = discount factor (0–1)

The discount factor controls the importance of **future rewards**.

---

# Exploration vs Exploitation

A central challenge in RL is balancing:

**Exploration**

Trying new actions to discover better rewards.

**Exploitation**

Using known actions that produce high rewards.

Effective agents must balance both strategies.

---

# Value Functions

Value functions estimate **how good a state or action is**.

---

## State Value Function

V(s)

Expected return starting from state **s**.

---

## Action Value Function (Q-function)

Q(s, a)

Expected return after taking action **a** in state **s**.

These functions guide the agent toward better decisions.

---

# Common Reinforcement Learning Algorithms

## Q-Learning

A value-based algorithm that learns the **optimal action-value function**.

See: [[Q Learning]]

---

## Deep Q Networks (DQN)

Uses [[Neural Networks]] to approximate the Q-function.

Used in complex environments such as video games.

---

## Policy Gradient Methods

Directly optimize the policy instead of value functions.

---

## Actor-Critic Methods

Combine:

- policy-based methods
    
- value-based methods
    

---

# Applications

Reinforcement learning is widely used in:

- robotics
    
- autonomous driving
    
- game AI (AlphaGo, AlphaZero)
    
- recommendation systems
    
- resource management
    
- finance
    

---

# Advantages

- learns through interaction with environment
    
- suitable for sequential decision problems
    
- capable of solving complex control tasks
    

---

# Limitations

- requires large amounts of training data
    
- training can be unstable
    
- reward design can be difficult
    

---

# Related Concepts

[[Machine Learning]]  
[[Supervised Learning]]  
[[Unsupervised Learning]]  
[[Markov Decision Process]]  
[[Q Learning]]  
[[Deep Reinforcement Learning]]  
[[Neural Networks]]