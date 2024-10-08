from collections import Counter
from players.guesser import Guesser
import players.guesser_random_dialect


class MetaGuesser:
    """
    This committee works as follows: each guesser returns 3 guesses along with their certainty levels.
    Each guesser has 1 point to distribute among their guesses, and they will do so using normalization.
    """

    def __init__(self, glove_vecs=None, word_vectors=None, num_players=5):
        self.players = [players.guesser_random_dialect.AIGuesser(glove_vecs, word_vectors) for i in
                        range(num_players)]


    def set_board(self, words):
        for player in self.players:
            player.set_board(words)

    def set_clue(self, clue, num):
        for player in self.players:
            player.set_clue(clue, num)

    def keep_guessing(self):
        return all(player.keep_guessing() for player in self.players)

    def get_answer(self):
        """
        Nornalize the guesses as the closest the word is, the higher score it gets.
        Every player has 1 point to distribute among their guesses
        """
        all_guesses = []
        for player in self.players:
            guesses = player.get_answer(3)  # Assume this method returns [(certainty, guess), ...]
            all_guesses.append(guesses)

        answer_counts = Counter()

        for guesses in all_guesses:
            weights = [2 - guess[0] for guess in guesses]
            total_weight = sum(weights)
            normalized_weights = [weight / total_weight for weight in weights]

            for i in range(3):
                answer_counts[guesses[i][1]] += normalized_weights[i]

        for (el, count) in answer_counts.items():
            print(f"ELEMENT: {el}, COUNT: {count}")

        final_answer, final_count = answer_counts.most_common(1)[0]
        return final_answer
