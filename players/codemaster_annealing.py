import math

# only anneal the word choice, not the num choice.

# choose num according to how many words you think match this word before a bad word will appear, and make the score
# consider it

import numpy as np
import scipy
from simanneal import Annealer
import random
import json

from players.codemaster import Codemaster


class MyProblem(Annealer):
    def __init__(self, state, pre_processed_ds, lm, good_words, bad_words, assassin, previous_clues):
        self.state = state
        self.pre_processed_ds = pre_processed_ds  # a dictionary where each word has it's top 20 words
        self.lm = lm  # a dictionary where each word is a vector
        self.good_words = good_words
        self.bad_words = bad_words
        self.assassin = assassin
        self.num = 1
        self.previous_clues = previous_clues
        super(MyProblem, self).__init__(state)
        self.good_factor = 7.5
        self.assassin_factor = 3
        self.bad_factor = 1
        self.num_factor = 1

    def move(self):
        legal_words = [item for item in self.pre_processed_ds[self.state.lower()][:10] if item.lower() not in self.good_words and item.lower() not in self.bad_words]
        weights = [math.sqrt(len(legal_words) - i) for i in range(len(legal_words))]

        # Choose a word randomly with weights
        chosen_word = random.choices(legal_words, weights=weights, k=1)[0]
        # chosen_word = random.choices(legal_words, k=1)[0]
        while chosen_word in self.good_words or chosen_word in self.bad_words:
            chosen_word = random.choices(legal_words, weights=weights, k=1)[0]
            # chosen_word = random.choices(legal_words, k=1)[0]
        self.state = chosen_word
    #
    def update(self, *args, **kwargs):
        # keep this if I want to avoid printing.
        pass

    def energy(self):

        def word_score(word):
            dist = scipy.spatial.distance.cosine(self.lm[word.lower()], self.lm[self.state.lower()])
            if dist > 1.99:
                return 2.1 - dist

            return dist

        available_words = []
        for word in self.good_words:
            available_words.append(word)
        for word in self.bad_words:
            available_words.append(word)

        available_words.sort(key=word_score)

        num = 0
        for w in available_words:
            if w in self.good_words and word_score(w) < 0.8:
                num += 1
            else:
                break
        if num == 0:
            return 1000
        good_dist = np.mean([word_score(w) for w in available_words[:num]], axis=0)
        bad_dist = min([word_score(word) for word in self.bad_words])
        assassin_dist = word_score(self.assassin.lower())
        score = (self.good_factor*good_dist) - (self.bad_factor * bad_dist) - (self.assassin_factor*assassin_dist) - (self.num_factor * num)

        if self.state.lower() in self.previous_clues:
            score += 10

        if self.best_energy is not None and self.best_energy >= score:
            self.num = max(1, num)
        return score


class AICodemaster(Codemaster):

    def __init__(self, glove_vecs = None, word_vectors = None):
        if glove_vecs:
            self.lm = glove_vecs
        elif word_vectors:
            self.lm = word_vectors
        else:
            raise RuntimeError("No vectors were provided")
        super().__init__()
        self.pre_processed_ds = {}
        with open('closest_combined_words_within_filtered_glove.json', 'r') as f:

            self.pre_processed_ds = json.load(f)
        self.previous_clues = set()

    def set_game_state(self, words_in_play, map_in_play):
        self.words = words_in_play
        self.maps = map_in_play

    def get_clue(self, bad_color = "Blue", good_color="Red"):
        red_words = []
        bad_words = []
        assassin = ''

        for i in range(25):
            if self.words[i][0] == '*':
                continue
            elif self.maps[i] == "Assassin" or self.maps[i] == bad_color or self.maps[i] == "Civilian":
                bad_words.append(self.words[i].lower())
                if self.maps[i] == "Assassin":
                    assassin = self.words[i]
            else:
                red_words.append(self.words[i].lower())

        initial_state = self.get_init_state(red_words, bad_words)

        problem = MyProblem(initial_state, self.pre_processed_ds, self.lm, red_words, bad_words, assassin,
                            self.previous_clues)
        problem.steps = 5000  # Number of iterations
        problem.Tmax = 1.0  # Initial temperature
        problem.Tmin = 0.01  # Final temperature
        problem.schedule = 'linear'
        best_state, best_energy = problem.anneal()
        self.previous_clues.add(best_state.lower())

        return best_state.lower(), max(1, problem.num)

    def get_init_state(self, red_words, bad_words):
        avg_red_word_score = np.mean([self.lm[w] for w in red_words], axis=0)
        init_index = np.argmin([np.linalg.norm(self.lm[self.pre_processed_ds[w][0]] - avg_red_word_score)
                                for w in red_words])
        state = self.pre_processed_ds[red_words[init_index]][0]
        while state in red_words or state in bad_words:
            state = random.choice(self.pre_processed_ds[random.choice(red_words)])
        return state

