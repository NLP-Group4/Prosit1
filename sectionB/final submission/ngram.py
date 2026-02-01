"""
Ngram model for text generation in twi
"""

import math
import random
from preprocessor import Text as tx

class Ngram:
    def __init__(self, corpus, n=2, smoothing='IP', eval_set=None):
        self.n = n
        self.tokens = corpus if isinstance(corpus, list) else tx(corpus).word_tokens
        self.eval_set = tx(eval_set).word_tokens if eval_set else self.tokens
        self.ngram_counts = self.build_ngram_counts()
        self.vocabs = set(self.tokens)
        self.V = len(self.vocabs)
        self.N = len(self.tokens)

        self.smoothing = smoothing
        # only generate lambdas if we're using interpolation
        self.weights = self.__generate_lambdas(n) if smoothing == 'IP' else None

        # build lower order models for interpolation or kneser-ney - only if needed
        if smoothing == 'IP':
            self.lower_models = {gram: Ngram(self.tokens, n=gram, smoothing='None', eval_set=None)
                                 for gram in range(1, n)}
        elif smoothing == 'KN':
            # for kneser-ney, build lower models with KN smoothing too
            self.lower_models = {gram: Ngram(self.tokens, n=gram, smoothing='KN' if gram > 1 else 'None', eval_set=None)
                                 for gram in range(1, n)}
            # precompute continuation counts for efficiency
            self.continuation_counts = self.__build_continuation_counts()
        else:
            self.lower_models = None
            self.continuation_counts = None



    # loosely the training function
    def build_ngram_counts(self) -> dict[tuple[str, str], dict[str, int]]:
        # build a table(dict) of {(n-1_tokens):{w1: #, w2: #, ....}}
        table = dict()

        for i in range(len(self.tokens) - self.n + 1):
            context = self.tokens[i:i+self.n-1]
            word = self.tokens[i+self.n-1]

            if tuple(context) not in table:
                table[tuple(context)] = dict()

            table[tuple(context)][word] = table[tuple(context)].get(word, 0) + 1

        # for unigrams, context is empty tuple, so just return the word counts dict
        table = table[tuple([])] if self.n == 1 else table

        return table


    def P(self, word, context) -> float:
        # ensure context is always a tuple for consistency
        context = tuple(context) if not isinstance(context, tuple) else context

        if self.smoothing == 'IP':
            return self.__prob_interpolation(word, context)
        elif self.smoothing == 'LP':
            return self.__prob_laplace(word, context)
        elif self.smoothing == 'KN':
            return self.__prob_kneser_ney(word, context)
        else:
            return self._prob_no_smoothing(word, context)


    def _prob_no_smoothing(self, word, context):
        # raw MLE probability without any smoothing
        if self.n == 1:
            return self.ngram_counts.get(word, 0) / len(self.tokens)

        sequence_count = self.ngram_counts.get(context, {}).get(word, 0)
        context_count = sum(self.ngram_counts.get(context, {}).values())

        return sequence_count / context_count if context_count > 0 else 0.0



    def __prob_laplace(self, word, context, alpha=1):
        # add-alpha smoothing, default alpha=1 is add-one/laplace smoothing
        if self.n == 1:
            return (self.ngram_counts.get(word, 0) + alpha) / (self.N + alpha * self.V)

        context_bucket = self.ngram_counts.get(context, {})

        sequence_count = context_bucket.get(word, 0) + alpha
        context_count = sum(context_bucket.values()) + alpha * self.V

        return sequence_count / context_count


    def __prob_interpolation(self, word, context):
        prob = 0.0
        sub_context = context

        for gram in range(self.n, 0, -1):
            if gram == self.n:
                # use this model for the highest order
                model_prob = self._prob_no_smoothing(word, sub_context)
            else:
                # use pre-built lower order models
                model = self.lower_models[gram]
                model_prob = model._prob_no_smoothing(word, sub_context)

            # weight and accumulate
            weight = self.weights[self.n - gram]
            temp = weight * model_prob

            print(f'{weight} * P_{gram}({word}|{sub_context}) = {temp}')
            prob += temp

            # shrink context for next lower order model
            sub_context = sub_context[1:] if len(sub_context) > 0 else sub_context

        return prob



    def __build_continuation_counts(self):
        # precompute continuation counts for kn
        # continuation count: unique contexts a word appears in
        continuation = {}

        for context in self.ngram_counts:
            for word in self.ngram_counts[context]:
                continuation[word] = continuation.get(word, 0) + 1

        return continuation


    def __prob_kneser_ney(self, word, context, d=0.75):
        # unigram for base case using continuation probability
        if self.n == 1:
            if not self.continuation_counts:
                # fallback to uniform if no continuation counts
                return 1.0 / self.V if self.V > 0 else 0.0

            total_bigram_types = sum(self.continuation_counts.values())
            continuation_count = self.continuation_counts.get(word, 0)

            return continuation_count / total_bigram_types if total_bigram_types > 0 else 0.0

        context_bucket = self.ngram_counts.get(context, {})

        # discounted count
        raw_count = context_bucket.get(word, 0)
        discounted_count = max(raw_count - d, 0)

        context_total = sum(context_bucket.values())

        # compute lower order probability using pre-built lower model
        lower_context = context[1:] if len(context) > 0 else tuple()

        if self.n - 1 in self.lower_models:
            # use pre-built lower order model
            lower_model = self.lower_models[self.n - 1]
            lower_prob = lower_model.P(word, lower_context)
        else:
            # shouldn't happen but fallback to uniform
            lower_prob = 1.0 / self.V if self.V > 0 else 0.0

        if context_total == 0:
            return lower_prob

        # compute lambda (normalization factor for backoff)
        num_unique_continuations = len(context_bucket)
        lambda_weight = (d / context_total) * num_unique_continuations

        # combine discounted probability with backoff
        prob = (discounted_count / context_total) + (lambda_weight * lower_prob)

        return prob


    def perplexity(self) -> float:
        # perplexity = 2^(-1/M * sum(log2(P(w_i)))), on eval set
        log_prob_sum = 0.0
        M = 0  # number of predicted tokens actually scored

        for i in range(len(self.eval_set) - self.n + 1):
            context = tuple(self.eval_set[i:i+self.n-1])
            word = self.eval_set[i+self.n-1]

            prob = self.P(word, context)
            print('Probability of', word, 'given', context, 'is', prob)

            if prob > 0:
                log_prob_sum += math.log2(prob)
                M += 1
            # skip zero probability cases -> pp will be inf

        # guard against division by zero
        if M == 0:
            return float('inf')

        avg_log_prob = log_prob_sum / M
        perplexity = 2 ** (-avg_log_prob)
        return perplexity


    @staticmethod
    def apply_temperature(probs, temperature=0.8):
        if temperature <= 0:
            raise ValueError("Temperature must be greater than 0")

        adjusted_probs = [p ** (1 / temperature) for p in probs]
        total = sum(adjusted_probs)

        # normalize to sum to 1
        return [p / total for p in adjusted_probs] if total > 0 else probs


    def generate_random_sentence(self, max_words=20, temperature=0.8, start_token='<s>', end_token='</s>', style='w_sampling'):
        # generate a sentence by sampling from the model
        # style options:
        #   'w_sampling': weighted sampling based on probabilities with temperature
        #   'greedy': always pick the most likely next word
        #   'random': uniformly random choice (ignores probabilities)

        sentence = [start_token]

        # initialize context with start tokens for higher order models
        current_context = tuple([start_token] * (self.n - 1))

        for _ in range(max_words):
            # get all possible next words
            possible_words = list(self.vocabs)

            if style == 'random':
                # random selection, ignore probabilities
                next_word = random.choice(possible_words)

            elif style == 'greedy':
                # pick the word with highest probability
                probs = [self.P(word, current_context) for word in possible_words]

                # edge case (all probs are zero), just choose randomly
                if max(probs) == 0:
                    next_word = random.choice(possible_words)
                else:
                    # get index of max probability
                    max_idx = probs.index(max(probs))
                    next_word = possible_words[max_idx]

            elif style == 'w_sampling':
                # weighted sampling with temperature
                probs = [self.P(word, current_context) for word in possible_words]

                if sum(probs) == 0:
                    probs = [1.0] * len(possible_words)

                # apply temperature to adjust randomness
                adjusted_probs = self.apply_temperature(probs, temperature)

                # compute CDF and sample uniformly
                next_word = random.choices(possible_words, weights=adjusted_probs, k=1)[0]

            else:
                raise ValueError(f"Invalid style: {style}. Choose 'w_sampling', 'greedy', or 'random'")

            sentence.append(next_word)

            # stop if we hit the end token
            if next_word == end_token:
                break

            # update context for next prediction
            # take last n-1 words as context
            current_context = tuple(sentence[-(self.n-1):])

        return ' '.join(sentence)


    def get_top_k_predictions(self, context, k=5):
        context = tuple(context) if not isinstance(context, tuple) else context

        word_probs = [(word, self.P(word, context)) for word in self.vocabs]
        word_probs.sort(key=lambda x: x[1], reverse=True)

        return word_probs[:k]


    @staticmethod
    def __generate_lambdas(n: int, r: float = 0.35) -> list[float]:
        # generate interpolation weights using geometric decay
        # lambdas ordered as [l_n, l_n-1, ..., l_1] so that l_n >= l_n-1 >= ... >= l_1

        if n <= 0:
            raise ValueError("n must be >= 1")

        if n == 1:
            return [1.0]  # unigram only, weight is 1

        if not (0.0 < r < 1.0):
            raise ValueError("r must be in (0, 1)")

        # generate geometric sequence: [1, r, r^2, ..., r^(n-1)]
        raw = [r ** k for k in range(n)]

        # normalize so they sum to 1
        total = sum(raw)
        lambdas = [x / total for x in raw]

        return lambdas


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]

    bigram = Ngram(args[0], n=int(args[1]), smoothing=args[2], eval_set=args[3])
    print(f'Perplexity: {bigram.perplexity()}')
