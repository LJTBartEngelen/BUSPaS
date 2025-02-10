import pandas as pd


def mcts_pval_result_to_latex(results_with_p_values):
    index = []
    descriptions = []
    qualities = []
    coverages = []
    pvalues = []

    for result_idx in range(len(results_with_p_values)):
        number = result_idx + 1
        index.append(number)
        description = " Ʌ ".join(
            ["(" + condition.replace(" == ", "=").strip() + ")" for condition in results_with_p_values[result_idx][3]])
        descriptions.append(description)
        quality = round(results_with_p_values[result_idx][0], 3)
        qualities.append(quality)
        coverage = round(results_with_p_values[result_idx][1], 3)
        coverages.append(coverage)
        pvalue = round(results_with_p_values[result_idx][4], 3)
        pvalues.append(pvalue)

    results_structured = pd.DataFrame(
        {'Description': descriptions, 'Quality': qualities, 'Coverage': coverages, 'p-Value': pvalues}, index=index)

    return results_structured.round(3).astype(str), results_structured.round(3).astype(str).to_latex()


def bus_pval_result_to_latex(results_with_p_values):
    index = []
    descriptions = []
    qualities = []
    coverages = []
    pvalues = []

    for result_idx in range(len(results_with_p_values)):
        number = result_idx + 1
        index.append(number)
        description = " Ʌ ".join(
            ["(" + condition.replace(" == ", "=").strip() + ")" for condition in results_with_p_values[result_idx][2]])
        descriptions.append(description)
        quality = round(results_with_p_values[result_idx][0], 3)
        qualities.append(quality)
        coverage = round(results_with_p_values[result_idx][1], 3)
        coverages.append(coverage)
        pvalue = round(results_with_p_values[result_idx][3], 3)
        pvalues.append(pvalue)

    results_structured = pd.DataFrame(
        {'Description': descriptions, 'Quality': qualities, 'Coverage': coverages, 'p-Value': pvalues}, index=index)

    return results_structured.round(3).astype(str), results_structured.round(3).astype(str).to_latex()


def bs_pval_result_to_latex(results_with_p_values):
    index = []
    descriptions = []
    qualities = []
    coverages = []
    pvalues = []

    for result_idx in range(len(results_with_p_values)):
        number = result_idx + 1
        index.append(number)
        description = " Ʌ ".join(
            ["(" + condition.replace(" == ", "=").strip() + ")" for condition in results_with_p_values[result_idx][3]])
        descriptions.append(description)
        quality = round(results_with_p_values[result_idx][0], 3)
        qualities.append(quality)
        coverage = round(results_with_p_values[result_idx][1], 3)
        coverages.append(coverage)
        pvalue = round(results_with_p_values[result_idx][5], 3)
        pvalues.append(pvalue)

    results_structured = pd.DataFrame(
        {'Description': descriptions, 'Quality': qualities, 'Coverage': coverages, 'p-Value': pvalues}, index=index)

    return results_structured.round(3).astype(str), results_structured.round(3).astype(str).to_latex()
