# CodenamesAI

This repository contains the implementation of AI agents designed to play the game "Codenames." The project explores various AI strategies, including word vector models, multi-agent systems, and communication across different semantic spaces. 

## Project Overview

Codenames is a popular board game that involves two teams competing to identify words related to clues given by their team's spymaster. This project tackles the challenge of developing AI agents that can effectively play as spymasters and guessers, considering the semantic nuances required for providing and interpreting clues.

## Methodology

### Word to Vector Models
The AI agents utilize word vector models like Word2Vec and GloVe to map words into high-dimensional spaces, capturing semantic relationships between words. These models enable the spymaster to generate clues that are semantically close to the target words while avoiding dangerous words like the assassin.

### Multi-Agent System: Committees and Auctions
To simulate real-world dynamics, the AI implements a multi-agent system where agents make decisions based on committee voting or auction-based mechanisms. This approach ensures a robust decision-making process by aggregating multiple perspectives from different agents, mimicking the collective decision-making found in human gameplay.

### Communication Across Semantic Spaces
One unique aspect of the project is training a neural network to translate between different word vector models, like GloVe and Google News Word2Vec, to facilitate better communication between agents using different semantic spaces. This approach aims to bridge gaps in understanding, enhancing the overall coordination between agents.

## Key Components

- **Spymaster Agent**: Generates clues based on vector distances and optimized search strategies, including Simulated Annealing, to find the optimal clue and number combination.
  
- **Guesser Agents**: Utilize various algorithms, including dialect matrix adjustments and random selection within a constrained feature space, to interpret the clues provided by the spymaster effectively.

- **Multi-Agent Strategies**: Implements committee and auction-based methods for collective decision-making among multiple agents. This includes Borda Count, Proportional Voting, and Sealed Bid Auctions, each evaluated against set criteria for effectiveness and fairness.

## Results

The AI agents demonstrated varying success rates across different configurations. Key findings include:
- The GloVe-based agents often aligned better with human-like semantic understanding but struggled with specific in-game dynamics.
- Auction-based multi-agent approaches provided superior results in decision-making scenarios, highlighting the potential of competitive strategies in AI coordination.

## Future Work

The project opens several avenues for further research, including:
- Enhancing the robustness of AI agents by exploring additional word vector models and training techniques.
- Developing more sophisticated multi-agent algorithms that can dynamically adjust strategies based on game state and opponent behavior.
- Investigating the application of game-theoretic approaches and reinforcement learning to improve AI performance in Codenames and similar games.

## Getting Started


### Installation
Clone the repository and install the necessary dependencies:
```bash
git clone https://github.com/ItamarBaer/CodenamesAI.git
cd CodenamesAI
pip install -r requirements.txt
