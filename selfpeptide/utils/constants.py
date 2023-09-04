MIN_PEPTIDE_LEN=8
MAX_PEPTIDE_LEN=12



PADDING_TOKEN = "*"
ALIGNMENT_TOKEN = '-'
amino_acids = sorted(['A',
 'C',
 'D',
 'E',
 'F',
 'G',
 'H',
 'I',
 'K',
 'L',
 'M',
 'N',
 'P',
 'Q',
 'R',
 'S',
 'T',
 'V',
 'W',
 'X',
 'Y'])

sorted_vocabulary = ['A',
 'C',
 'D',
 'E',
 'F',
 'G',
 'H',
 'I',
 'K',
 'L',
 'M',
 'N',
 'P',
 'Q',
 'R',
 'S',
 'T',
 'V',
 'W',
 'X',
 'Y',
 '-',
 '*']


beta_priors = {
    "uniform": {
        "Negative": [1, 1],
        "Positive": [1, 1],
        "Positive-Intermediate": [1, 1],
        "Positive-Low": [1, 1],
        "Positive-High": [1, 1],
        },
    "weak_priors": {
        "Negative": [1, 3],
        "Positive": [8, 2],
        "Positive-Intermediate": [8, 2],
        "Positive-Low": [3, 2],
        "Positive-High": [18, 2],
    },
    "deepimmuno": {
        "Negative": [3, 3],
        "Positive-Low": [28, 1],
        "Positive": [30, 1],
        "Positive-Intermediate": [30, 1],
        "Positive-High": [32, 1],
    },
    "intermediate_priors": {
        "Negative": [2, 8],
        "Positive": [16, 4],
        "Positive-Intermediate": [16, 4],
        "Positive-Low": [12, 4],
        "Positive-High": [20, 2],
    },
    "strong_priors": {
        "Negative": [2, 18],
        "Positive": [18, 2],
        "Positive-Intermediate": [18, 2],
        "Positive-Low": [16, 4],
        "Positive-High": [24, 2],
    },
    "strong_priors_lower_var": {
                    "Negative": [3, 27],
                    "Positive": [27, 3],
                    "Positive-Intermediate": [27, 3],
                    "Positive-Low": [24, 6],
                    "Positive-High": [36, 3],
                  },
    
    "strong_priors_lowest_var": {
                    "Negative": [4, 36],
                    "Positive": [36, 4],
                    "Positive-Intermediate": [36, 4],
                    "Positive-Low": [32, 8],
                    "Positive-High": [48, 4],
                  },
    
    "strongest_priors": {
                    "Negative": [2, 36],
                    "Positive": [36, 2],
                    "Positive-Intermediate": [36, 2],
                    "Positive-Low": [32, 4],
                    "Positive-High": [48, 2],
                  },
    "strong_priors_lowest_var_strong_negatives": {
                    "Negative": [1, 36],
                    "Positive": [36, 4],
                    "Positive-Intermediate": [36, 4],
                    "Positive-Low": [32, 8],
                    "Positive-High": [48, 4],
                  },
    
    "p1": {
        "Negative": [3, 3],
        "Positive": [5, 1],
        "Positive-Intermediate": [5, 1],
        "Positive-Low": [3, 1],
        "Positive-High": [6, 1]
    },
    "avg_posterior_uniform": {
        "Positive-Low": [4.24, 6.88],
        "Positive": [4.76, 7.61],
        "Negative": [1.00, 4.57],
        "Positive-Intermediate": [6.75, 7.69],
        "Positive-High": [11.10, 5.09]
    },
    "full_random_search_mapping": {'Negative': [1, 1],
        'Positive': [15, 7],
        'Positive-High': [24, 7],
        'Positive-Intermediate': [11, 37],
        'Positive-Low': [21, 5]},
    
    "constrained_random_search_mapping": {'Negative': [1, 1],
        'Positive-Low': [21, 16],
        'Positive': [36, 27],
        'Positive-Intermediate': [36, 27],
        'Positive-High': [30, 22]}
    
}
