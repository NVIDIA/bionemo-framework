from enum import Enum

class TaskTypes(str, Enum):
    MUTATION_PREDICTION = 'mutation_prediction'
    MASKED_LANGUAGE_MODELING = 'masked_language_modeling'
    FITNESS_PREDICTION = 'fitness_prediction'
    EMBEDDING_PREDICTION = 'embedding_prediction'
    SYNONYMOUS_MUTATION_PREDICTION = 'synonymous_mutation_prediction'
    DOWNSTREAM_PREDICTION = 'downstream_prediction'
    NEXT_CODON_PREDICTION = 'next_codon_prediction'
    SEQUENCE_GENERATION = 'sequence_generation'
    MISSENSE_PREDICTION = 'missense_prediction'
