import pandas as pd
from scipy.stats import pearsonr
from pathlib import Path

def calculate_correlation(gold_standard_path, VAD_lexicon_path):
    """
    computes correlation (Pearson's r) between gold standard lexicon and predicted VAD lexicon.
    :param gold_standard_path: str, path to gold standard lexicon
    :param VAD_lexicon_path: str, path to induced VAD lexicon
    :return: list, [r (Valence dimension), p value (Valence dimension), r (Arousal dimension), p value (Arousal dimension),
    r (Dominance dimension), p value (Dominance dimension), r (across all three dimensions)]
    """
    GOLD_words = []
    GOLD_V = []
    GOLD_A = []
    GOLD_D = []
    with open(gold_standard_path) as f:
        for l in f:
            l = l.split('\t')
            GOLD_words.append(l[0])
            GOLD_V.append(float(l[1]))
            GOLD_A.append(float(l[2]))
            GOLD_D.append(float(l[3]))

    Predicted_V = []
    Predicted_A = []
    Predicted_D = []

    with open(VAD_lexicon_path) as file:
        for line in file:
            line = line.split('\t')
            Predicted_V.append(float(line[1]))
            Predicted_A.append(float(line[2]))
            Predicted_D.append(float(line[3]))

    r_V, p_value_V = pearsonr(Predicted_V, GOLD_V)
    r_A, p_value_A = pearsonr(Predicted_A, GOLD_A)
    r_D, p_value_D = pearsonr(Predicted_D, GOLD_D)

    single_mean = (r_V + r_A + r_D) / 3
    return [round(r_V, 4), round(p_value_V, 4), round(r_A, 4), round(p_value_A,4), round(r_D, 4), round(p_value_D, 4), round(single_mean, 4)]

def compare_lexica(directory_path, gold_standard_path):
    """
    computes correlation (Pearson's r) for a whole folder of predicted VAD lexica. Saves information as csv file
    (correlation values for every individual lexicon + mean of all lexica in the last line)
    :param directory_path: str, directory path to predicted VAD lexica.
    :param gold_standard_path: str, path to gold standard lexica.
    :return: list, mean of all lexica correlations
    """
    result = []
    row_names = []
    col_names = ['r_V', 'p_value_V', 'r_A', 'p_value_A', 'r_D', 'p_value_D', 'single_mean']

    lexica_paths_to_compare = []
    lexica_paths_to_compare.extend(sorted(list(directory_path.glob('*'))))

    for entry in lexica_paths_to_compare:
        full_path = entry
        row_names.append(entry)
        result.append(calculate_correlation(gold_standard_path, full_path))

    df =  pd.DataFrame(result, columns= col_names, index=row_names)
    means = df.mean()
    df.loc['Mean'] = means

    save_dir = Path('HistoricalVAD/Correlations/')
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{directory_path.parts[1]}_{directory_path.parts[2]}_Correlation.csv"

    df.to_csv(save_path)
    return list(df.loc['Mean'])




