#BigPhoney
from big_phoney import BigPhoney

#User defined libraries
from generate_word_pairs import filter_alphabets, generate_pairs
from generate_nearest_neighbours import *
from generate_homophones import generate_homophones

if __name__ == '__main__':

	phoney = BigPhoney()

	generate_pairs(phoney)

	generate_homophones()

	generate_nn(phoney)