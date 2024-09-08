# CodenamesAI

This repository contains the implementation of AI agents designed to play the game "Codenames." The project explores various AI strategies, including word vector models, multi-agent systems, and communication across different semantic spaces. This repository uses code from CodenamesAICompetition: *https://github.com/CodenamesAICompetition/Game*

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

- **Multi-Agent Strategies**: Implements committee and auction-based methods for collective decision-making among multiple agents. This includes Plurality Voting, Borda Count, Proportional Voting, Sealed Bid with limited\unlimited budget and Vickrey Auctions, each evaluated against set criteria for effectiveness and fairness.


## Running the game from terminal instructions

**run_game.py handles system arguments.**

Optionally if certain word vectors are needed, the directory to which should be specified in the arguments here.
argument parsers have been provided:
* --two_teams *True/False*
* --guesser1 *path/to/red_teams_guesser*
* --guesser2 *path/to/blue_teams_guesser* (optional - only relevant when two_teams=True)
* --codemaster1 *path/to/red_teams_codemaster*
* --codemaster2 *path/to/blue_teams_codemaster* (optional - only relevant when two_teams=True)
  
* --w2v *path/to/google_vectors*
  * (to be loaded by gensim)
* --glove *path/to/glove_vectors*
  *  (in stanford nlp format)


An optional seed argument can be used for the purpose of consistency against the random library.
* --seed *Integer value* or "time"
  * ("time" uses Time.time() as the seed)


**Other optional arguments include:**
* --no_log
  * raise flag for suppressing logging
* --no_print
  * raise flag for suppressing printing to std out
* --game_name *String*
  * game_name in logfile
* --num_games *Int*
  * run consecutive games

    

## Installation
Clone the repository and install the necessary dependencies:
```bash
git clone https://github.com/ItamarBaer/CodenamesAI.git
cd CodenamesAI
pip install -r requirements.txt
```


## Example Usage:

 A game between consisting of two human players:
```python
python run_game.py
```

A game consisting of the Annealer codemaster and the baseline Google guesser:
```python
python run_game.py --codemaster1  players.codemaster_annealing.AICodemaster --guesser1 players.guesser_google_baseline.AIGuesser --w2v filtered_GoogleNews-vectors-negative300.bin --glove filtered_glove.6B.300d.txt
```

A game consisting of the Annealer codemaster and Plurality Voting:
```python
python run_game.py --codemaster1  players.codemaster_annealing.AICodemaster --guesser1 players.guesser_multi_agent_plurality_voting.MetaGuesser --w2v filtered_GoogleNews-vectors-negative300.bin --glove filtered_glove.6B.300d.txt
```



#### The Vector files contanied in this repository are filtered. The full files can be found here:

* [Glove Vectors](https://nlp.stanford.edu/data/glove.6B.zip) (~2.25 GB)
* [Google News Vectors](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) (~3.5 GB)
