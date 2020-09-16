
import random


class Mutation:
    @staticmethod
    def reverse_gene(gene, start, end):
        gene[start:end] = reversed(gene[start:end])

    @staticmethod
    def randomly_swap_gene(gene):
        indices = range(len(gene))
        a, b = random.sample(indices, 2)
        gene[a], gene[b] = gene[b], gene[a]

    @staticmethod
    def shift_gene(gene, n):
        n = -n
        return gene[n:] + gene[:n]
