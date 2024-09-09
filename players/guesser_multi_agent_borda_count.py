from collections import Counter
from players.guesser import Guesser
import players.guesser_random_dialect


class MetaGuesser:
    """
    Each player ranks their guesses between 1-number of guesses,
    and the guesses receive points based on their rank.
    The guess with the highest total points across all players is the
    most acceptable candidate."""
    def __init__(self, glove_vecs=None, word_vectors=None):
        self.players = [players.guesser_random_dialect.AIGuesser(glove_vecs, word_vectors, -2) for i in
                        range(5)]
        self.unique_guess_counts = []
        self.certainty_of_chosen_guess = []

    def set_board(self, words):
        for player in self.players:
            player.set_board(words)

    def set_clue(self, clue, num):
        for player in self.players:
            player.set_clue(clue, num)

    def keep_guessing(self):
        return all(player.keep_guessing() for player in self.players)

    def get_answer(self):
        """ Simple weights version: the first is the most important, and so on"""
        all_guesses = []
        for player in self.players:
            guesses = player.get_answer(3)  # Assume this method returns [(certainty, guess), ...]
            all_guesses.append(guesses)

        answer_counts = Counter()
        for guesses in all_guesses:
            for i in range(3):
                answer_counts[guesses[i][1]] += 3 - i

        for (el, count) in answer_counts.items():
            print(f"ELEMENT: {el}, COUNT: {count}")

        final_answer, final_count = answer_counts.most_common(1)[0]

        total_score = sum(answer_counts.values())
        print(f'Meta-player final guess: {final_answer}')
        return final_answer