#!/usr/bin/env python3

from os import path
import sys
import numpy as np
import csv
from typing import List, Tuple, Dict
from sklearn.metrics import cohen_kappa_score
import statsmodels.api as sm
import krippendorff
import seaborn as sns
import pingouin as pg
import matplotlib.patches as mpatches
from scipy.stats import kstest
import scipy
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ttest_ind
import math
from scipy.interpolate import splev, splrep
import datetime as dt
from scipy.special import rel_entr
from fractions import Fraction
import pandas as pd
from pathlib import Path
from textwrap import wrap
from matplotlib.gridspec import GridSpec
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
import re
from scipy.stats import ttest_ind

weights = {'a': 0.2, 'b': 0.5, 'c': 1}

quan_low = 0.5
quan_medium = 0.75

# For the dataframe, list columns don't need to convert: Image ID, Model, Prompt, Pic Num, Special Note and Annotator
exclusive_columns = ['Image Id', 'Model', 'Prompt', 'Pic Num', 'Annotator']
# Error type and body part mappings
error_type_mapping = {
        'Missing Error': ['1. Missing Error-1.1 Torso', '1. Missing Error-1.2 Limbs', '1. Missing Error-1.3 Feet', '1. Missing Error-1.4 Hands', '1. Missing Error-1.5 Face'],
        'Extra Error': ['2. Extra Error-2.1 Torso', '2. Extra Error-2.2 Limbs', '2. Extra Error-2.3 Feet', '2. Extra Error-2.4 Hands', '2. Extra Error-2.5 Face'],
        'Configuration Error': ['3. Configuration Error-3.1 Torso', '3. Configuration Error-3.2 Limbs', '3. Configuration Error-3.3 Feet', '3. Configuration Error-3.4 Hands', '3. Configuration Error-3.5 Face'],
        'Orientation Error': ['4. Orientation Error-4.1 Torso', '4. Orientation Error-4.2 Limbs', '4. Orientation Error-4.3 Feet', '4. Orientation Error-4.4 Hands', '4. Orientation Error-4.5 Face'],
        'Proportion Error': ['5. Proportion Error-5.1 Torso', '5. Proportion Error-5.2 Limbs', '5. Proportion Error-5.3 Feet', '5. Proportion Error-5.4 Hands', '5. Proportion Error-5.5 Face']
}
body_part_mapping = {
        'Torso': ['1. Missing Error-1.1 Torso', '2. Extra Error-2.1 Torso', '3. Configuration Error-3.1 Torso', '4. Orientation Error-4.1 Torso', '5. Proportion Error-5.1 Torso'],
        'Limbs': ['1. Missing Error-1.2 Limbs', '2. Extra Error-2.2 Limbs', '3. Configuration Error-3.2 Limbs', '4. Orientation Error-4.2 Limbs', '5. Proportion Error-5.2 Limbs'],
        'Feet': ['1. Missing Error-1.3 Feet', '2. Extra Error-2.3 Feet', '3. Configuration Error-3.3 Feet', '4. Orientation Error-4.3 Feet', '5. Proportion Error-5.3 Feet'],
        'Hands': ['1. Missing Error-1.4 Hands', '2. Extra Error-2.4 Hands', '3. Configuration Error-3.4 Hands', '4. Orientation Error-4.4 Hands', '5. Proportion Error-5.4 Hands'],
        'Face': ['1. Missing Error-1.5 Face', '2. Extra Error-2.5 Face', '3. Configuration Error-3.5 Face', '4. Orientation Error-4.5 Face', '5. Proportion Error-5.5 Face']
}

# Define the prompts and categories
individual_prompts = ['person jogging', 'athlete performing salto']
pair_prompts = ['couple hugging', 'mother or father holding baby', 'physician examining patient', 'old couple in sauna', 'wrestling in arena']
group_prompts = ['people eating pizza','five people sunbathing on beach','five people playing volleyball']

columns_all_krip = ['Pair', 'Type', 'Value']

# Define custom colors for each model
colour_models = ['Blues', 'Greens', 'Reds']
palette_models = {"dall-e3": plt.get_cmap(colour_models[0])(0.6), "sdxl": plt.get_cmap(colour_models[1])(0.6), "stablecascade": plt.get_cmap(colour_models[2])(0.6)}

annotator_colours = {'anno1': '#F6C06E','anno3':'#ffd670', 'anno4':'#8F5A00','anno2':'#d76b48'}
colours_krip = {'anno1-anno2': '#e7965b', 'anno1-anno4': '#c38d37', 'anno3-anno4': '#c79838', 'anno2-anno3': '#eba15c', 'anno2-anno4': '#b36324', 'anno1-anno3': '#fbcb6f'}


error_level_colours = {"a": "darkgreen", "b": "gold", "c": "darkred"}
error_sev_colours = {"low": "darkgreen", "medium": "gold", "high": "darkred"}
# Define custom colors for each error type
error_type_colors = {
    'Missing Error': 'Blues',
    'Extra Error': 'Oranges',
    'Configuration Error': 'Greens',
    'Orientation Error': 'Greys',
    'Proportion Error': 'Purples'  # Adjust this according to your actual error types
}
error_type_palette = {
    'Missing Error': plt.get_cmap(error_type_colors['Missing Error'])(0.6),
    'Extra Error': plt.get_cmap(error_type_colors['Extra Error'])(0.6),
    'Configuration Error': plt.get_cmap(error_type_colors['Configuration Error'])(0.6),
    'Orientation Error': plt.get_cmap(error_type_colors['Orientation Error'])(0.6),
    'Proportion Error': plt.get_cmap(error_type_colors['Proportion Error'])(0.6) 
}


# Define custom colors for each body part
body_part_colors = {
    'Torso': 'Blues',
    'Limbs': 'Greens',
    'Feet': 'Reds',
    'Hands': 'Purples',
    'Face': 'Grays' 
}

body_part_palette = {
    'Torso': plt.get_cmap(body_part_colors['Torso'])(0.6),
    'Limbs': plt.get_cmap(body_part_colors['Limbs'])(0.6),
    'Feet': plt.get_cmap(body_part_colors['Feet'])(0.6),
    'Hands': plt.get_cmap(body_part_colors['Hands'])(0.6),
    'Face': plt.get_cmap(body_part_colors['Face'])(0.6),
}

def main():

    df, df2, combined_df = get_data()
    # get krippendorff stats
    krippendorff_results = inter_rater_agreement(df)
    dist_overall_sev = get_overall_sev_dist(df)
    error_count_df = get_data_error_level(df2)

    # cat1 = df[df['Model']=='sdxl']
    # cat2 = df[df['Model']=='stablecascade']
    # print(ttest_ind(cat1['Total Score'], cat2['Total Score']))

    # plot everything
    plot_on_grid(df, krippendorff_results, dist_overall_sev, error_count_df, combined_df)

#################
# prepare data #
################
def get_data():

    # read in data
    file_anno = "../data/annotations.csv"
    df = pd.read_csv(file_anno)
    df2 = pd.read_csv(file_anno)

    # Calculat the cumulative error score for each row
    weights = {'a': 0.2, 'b': 0.5, 'c': 1}

    # Function to convert annotations to numerical scores
    def annotation_to_score(annotation, weights):
        if pd.isna(annotation):
            return 0
        total_score = 0
        parts = annotation.split(',')
        for part in parts:
            match = re.search(r"(\d+)/(\d+)\s*([abc])", part.strip())
            if match:
                numerator, denominator, letter = match.groups()
                score = (weights[letter] * int(numerator)) / int(denominator)
                total_score += score
        return total_score


    # Apply the conversion function to all error columns
    for col in df.columns:
        if re.search(r'\d\.\d', col):
            df[col] = df[col].apply(lambda x: annotation_to_score(x, weights))

    # Calculate the total score for each image
    df['Total Score'] = df[[col for col in df.columns if re.search(r'\d\.\d', col)]].sum(axis=1)

    # Melt the DataFrame for visualization of violin plots for prompts
    prompt_melted_df = df.melt(id_vars=['Prompt'], value_vars=['Total Score'], value_name='Scores')

    # Define the prompts and categories
    individual_prompts = [
        'person jogging',
        'athlete performing salto'
    ]

    pair_prompts = [
        'couple hugging',
        'mother or father holding baby',
        'physician examining patient',
        'old couple in sauna',
        'wrestling in arena'
    ]

    group_prompts = [
        'people eating pizza',
        'five people sunbathing on beach',
        'five people playing volleyball'
    ]

    # Add a category column based on the prompts
    df['Category'] = df['Prompt'].apply(lambda x: 'Individual' if x in individual_prompts else 'Pair' if x in pair_prompts else 'Group')

    # Initialize an empty list to store the error counts in the desired format
    combined_counts = []

    # Iterate over each row in the DataFrame
    for i, row in df2.iterrows():
        # Aggregate counts by body part mapping
        for body_part, cols in body_part_mapping.items():
            for col in cols:
                if col in df2.columns:
                    counts = annotation_to_count(row[col])
                    # Create separate entries for error levels 'a', 'b', and 'c'
                    combined_counts.append({'Model': row['Model'], 'Prompt': row['Prompt'], 'Body Part': body_part, 'Error Type': None, 'Error Level': 'a', 'Counts': counts['a']})
                    combined_counts.append({'Model': row['Model'], 'Prompt': row['Prompt'], 'Body Part': body_part, 'Error Type': None, 'Error Level': 'b', 'Counts': counts['b']})
                    combined_counts.append({'Model': row['Model'], 'Prompt': row['Prompt'], 'Body Part': body_part, 'Error Type': None, 'Error Level': 'c', 'Counts': counts['c']})
        
        # Aggregate counts by error type mapping
        for error_type, cols in error_type_mapping.items():
            for col in cols:
                if col in df2.columns:
                    counts = annotation_to_count(row[col])
                    # Create separate entries for error levels 'a', 'b', and 'c'
                    combined_counts.append({'Model': row['Model'], 'Prompt': row['Prompt'], 'Body Part': None, 'Error Type': error_type, 'Error Level': 'a', 'Counts': counts['a']})
                    combined_counts.append({'Model': row['Model'], 'Prompt': row['Prompt'], 'Body Part': None, 'Error Type': error_type, 'Error Level': 'b', 'Counts': counts['b']})
                    combined_counts.append({'Model': row['Model'], 'Prompt': row['Prompt'], 'Body Part': None, 'Error Type': error_type, 'Error Level': 'c', 'Counts': counts['c']})

    # Convert the list to a DataFrame
    combined_df = pd.DataFrame(combined_counts)


    return df, df2, combined_df

def get_overall_sev_dist(data):

    scores_q = []
    i = 0

    just_3 = data[data['Annotator'] == 'anno3']
    quantiles = just_3.quantile(q=[0.25,0.5,0.75], method="table", interpolation="nearest", numeric_only=True)['Total Score']
    for _, rows in just_3.iterrows():
        if rows["Total Score"] <= quantiles[0.5]:
            scores_q.append([i,'low',rows['Annotator'],rows['Model']])
        elif rows["Total Score"] <= quantiles[0.75]:
            scores_q.append([i,'medium',rows['Annotator'],rows['Model']])
        else:
            scores_q.append([i,'high', rows['Annotator'], rows['Model']])
        i += 1
    just_1 = data[data['Annotator'] == 'anno1']
    quantiles = just_1.quantile(q=[0.25,0.5,0.75], method="table", interpolation="nearest", numeric_only=True)['Total Score']
    for _, rows in just_1.iterrows():
        if rows["Total Score"] <= quantiles[0.5]:
            scores_q.append([i,'low',rows['Annotator'],rows['Model']])
        elif rows["Total Score"] <= quantiles[0.75]:
            scores_q.append([i,'medium',rows['Annotator'],rows['Model']])
        else:
            scores_q.append([i,'high',rows['Annotator'],rows['Model']])
        i += 1
    just_2 = data[data['Annotator'] == 'anno2']
    quantiles = just_2.quantile(q=[0.25,0.5,0.75], method="table", interpolation="nearest", numeric_only=True)['Total Score']
    for _, rows in just_2.iterrows():
        if rows["Total Score"] <= quantiles[0.5]:
            scores_q.append([i,'low',rows['Annotator'],rows['Model']])
        elif rows["Total Score"] <= quantiles[0.75]:
            scores_q.append([i,'medium',rows['Annotator'],rows['Model']])
        else:
            scores_q.append([i,'high',rows['Annotator'],rows['Model']])
        i += 1
    just_4 = data[data['Annotator'] == 'anno4']
    quantiles = just_4.quantile(q=[0.25,0.5,0.75], method="table", interpolation="nearest", numeric_only=True)['Total Score']
    for _, rows in just_4.iterrows():
        if rows["Total Score"] <= quantiles[0.5]:
            scores_q.append([i,'low',rows['Annotator'],rows['Model']])
        elif rows["Total Score"] <= quantiles[0.75]:
            scores_q.append([i,'medium',rows['Annotator'],rows['Model']])
        else:
            scores_q.append([i,'high',rows['Annotator'],rows['Model']])
        i += 1

    return scores_q

# Function to convert annotations to numerical scores without weights
def annotation_to_count(annotation):
    if pd.isna(annotation):
        return {'a': 0, 'b': 0, 'c': 0}
    counts = {'a': 0, 'b': 0, 'c': 0}
    parts = annotation.split(',')
    for part in parts:
        match = re.search(r"(\d+)/(\d+)\s*([abc])", part.strip())
        if match:
            numerator, denominator, letter = match.groups()
            count = int(numerator)/ int(denominator)  # get the count without weights
            counts[letter] += count
    return counts

# Apply the conversion function to all error columns and calculate totals
def get_data_error_level(df):

    error_counts = []
    for i, row in df.iterrows():
        row_counts = {'Model': row['Model'], 'Prompt': row['Prompt']}
        total_a = total_b = total_c = 0
        for col in df.columns:
            if re.search(r'\d\.\d', col):
                counts = annotation_to_count(row[col])
                total_a += counts['a']
                total_b += counts['b']
                total_c += counts['c']
        row_counts['a'] = total_a
        row_counts['b'] = total_b
        row_counts['c'] = total_c
        error_counts.append(row_counts)

    error_count_df = pd.DataFrame(error_counts)
    
    return error_count_df

######################################
# inter-rater agreement calculations #
######################################
def inter_rater_agreement(df):

    scores: Dict[Tuple[str, str], Tuple[List[float], List[float]]] = {}
    categories_sum: Dict[Tuple[str, str], Tuple[List[int], List[int]]] = {}

    # just_score = df[['Model','Prompt','Pic Num.', 'Annotator', 'Total Score']].copy()
    df_filtered = df.groupby(["Model", "Prompt", "Pic Num."]).filter(lambda rows: rows['Annotator'].nunique() > 1).sort_values(["Model", "Prompt", "Pic Num."])

    for _, rows in df_filtered.groupby(["Model", "Prompt", "Pic Num."]):

        rows = rows.sort_values("Annotator")
        key = tuple(rows["Annotator"])

        if key not in scores:
            scores[key] = [], []
        if key not in categories_sum:
            categories_sum[key] = [], []

        scores[key][0].append(rows["Total Score"].values[0])
        scores[key][1].append(rows["Total Score"].values[1])

        for item in error_type_mapping:
            for i in error_type_mapping[item]:
                value_0_sum = 0
                value_1_sum = 0
                if rows[i].values[0] > 0.0:
                    value_0_sum = 1
                if rows[i].values[1] > 0.0:
                    value_1_sum = 1
            categories_sum[key][0].append(value_0_sum)
            categories_sum[key][1].append(value_1_sum)
    # krippendorffs alpha for cum score, overall error severity, binary categories
    return inter_rater_stats(scores, categories_sum)


def inter_rater_stats(df, df2):

    krippen_cum_score_dict = {}
    krippen_overall_sev_dict = {}
    krippen_bin_cat_dict = {}

    for key, value in df.items():
            
        low = np.quantile(np.array(value[0]), quan_low)
        medium = np.quantile(np.array(value[0]), quan_medium)
        low2 = np.quantile(np.array(value[1]), quan_low)
        medium2 = np.quantile(np.array(value[1]), quan_medium)

        scores_q = []
        scores_q2 = []
            
        for n in value[0]:
            if n <= low:
                scores_q.append('low')
            elif n <= medium:
                scores_q.append('medium')
            else:
                scores_q.append('high')

        for n in value[1]:
            if n <= low2:
                scores_q2.append('low')
            elif n <= medium2:
                scores_q2.append('medium')
            else:
                scores_q2.append('high')
        # calculate Krippendorff's alpha for overall seversity
        krippen_overall_sev_dict[key[0] + '-' + key[1]] = round(krippendorff.alpha([scores_q,scores_q2], level_of_measurement="ordinal", value_domain=["low", "medium", "high"]),2)
        # calculate Krippendorff's alpha for cum score
        krippen_cum_score_dict[key[0] + '-' + key[1]] = round(krippendorff.alpha([value[0],value[1]]),2)
   
    for key, value in df2.items():
        # calculate Krippendorffs alpha for binary categories
        krippen_bin_cat_dict[key[0] + '-' + key[1]] = round(krippendorff.alpha([value[0],value[1]]),2)


    all_krip = []
    for key, row in krippen_cum_score_dict.items():
        all_krip.append([key, 'Cumulative score\n agreement' ,row])
        all_krip.append([key, 'Overall image error\n severity agreement' ,krippen_overall_sev_dict[key]])
        all_krip.append([key, 'Per category\n agreement' ,krippen_bin_cat_dict[key]])

    return all_krip


#########
# plots #
#########

def plot_on_grid(df, krippendorff_results, dist_overall_sev, error_count_df, combined_df):

######### Figure 1
    fig = plt.figure(layout="constrained", figsize=(10, 5))

    gs = GridSpec(1, 2, figure=fig)
    ax11 = fig.add_subplot(gs[0])
    ax12 = fig.add_subplot(gs[1])

    plot_cum_score_dist(df, ax11)
    plot_dist_overall_severity(dist_overall_sev, ax12) 
    fig.text(0.01, 0.96, 'A', fontsize=14)
    fig.text(0.51, 0.96, 'B', fontsize=14)

    format_axes(fig)
    # plt.tight_layout()
    plt.savefig('../Figure_1.png', dpi=300)
    
    plt.cla()

    # stats on results
    print('dall-e3 mean ', df[df['Model'].isin(['dall-e3'])].loc[:, 'Total Score'].mean(numeric_only=True), ', variance ', df[df['Model'].isin(['dall-e3'])].loc[:, 'Total Score'].var(numeric_only=True))

    print('sdxl mean ',df[df['Model'].isin(['sdxl'])].loc[:, 'Total Score'].mean(numeric_only=True), ', variance ',df[df['Model'].isin(['sdxl'])].loc[:, 'Total Score'].var(numeric_only=True))

    print('stablecascade mean ', df[df['Model'].isin(['stablecascade'])].loc[:, 'Total Score'].mean(numeric_only=True), ', variance ', df[df['Model'].isin(['stablecascade'])].loc[:, 'Total Score'].var(numeric_only=True))


    print('Welch t-test dall-e3 vs sdxl ', scipy.stats.ttest_ind(df[df['Model'].isin(['dall-e3'])].loc[:, 'Total Score'], df[df['Model'].isin(['sdxl'])].loc[:, 'Total Score'], equal_var=False))
    print('Students t-test dall-e3 vs sdxl ', ttest_ind(df[df['Model'].isin(['dall-e3'])].loc[:, 'Total Score'], df[df['Model'].isin(['sdxl'])].loc[:, 'Total Score']))
    print('Welch t-test dall-e3 vs stablecascade ', scipy.stats.ttest_ind(df[df['Model'].isin(['dall-e3'])].loc[:, 'Total Score'], df[df['Model'].isin(['stablecascade'])].loc[:, 'Total Score'], equal_var=False))
    print('Students t-test dall-e3 vs stablecascade ', ttest_ind(df[df['Model'].isin(['dall-e3'])].loc[:, 'Total Score'], df[df['Model'].isin(['stablecascade'])].loc[:, 'Total Score']))
    print('Welch t-test stablecascade vs sdxl ', scipy.stats.ttest_ind(df[df['Model'].isin(['stablecascade'])].loc[:, 'Total Score'], df[df['Model'].isin(['sdxl'])].loc[:, 'Total Score'], equal_var=False))
    print('Students t-test stablecascade vs sdxl ', ttest_ind(df[df['Model'].isin(['stablecascade'])].loc[:, 'Total Score'], df[df['Model'].isin(['sdxl'])].loc[:, 'Total Score']))


######### Figure 2
    fig2 = plt.figure(layout="constrained", figsize=(10, 10))

    gs = GridSpec(2, 2, figure=fig2)
    ax11 = fig2.add_subplot(gs[0, 0])
    ax12 = fig2.add_subplot(gs[0, 1])
    ax21 = fig2.add_subplot(gs[1, 0])
    ax22 = fig2.add_subplot(gs[1, 1])

    plot_annotator_dist(df, ax11)
    plot_krippendorff(krippendorff_results,ax12) 
    plot_cum_score_dist_model(df, ax21)
    plot_error_level_dist_models(error_count_df, ax22)

    fig2.text(0.01, 0.98, 'A', fontsize=14)
    fig2.text(0.51, 0.98, 'B', fontsize=14)
    fig2.text(0.01, 0.47, 'C', fontsize=14)
    fig2.text(0.51, 0.47, 'D', fontsize=14)

    format_axes(fig2)
    # plt.tight_layout()
    plt.savefig('../Figure_2.png', dpi=300)


######### Figure 3
    plt.cla()
    fig3 = plt.figure(layout="constrained", figsize=(15, 10))

    gs = GridSpec(2, 3, figure=fig3)

    ax21 = fig3.add_subplot(gs[0, 0:3])
    ax31 = fig3.add_subplot(gs[1, 0:3])

    plot_prompt_cum_dist(df, ax21)
    plot_error_level_models_prompt(combined_df, ax31)

    print('five people sunbathing on beach, ', df[df['Prompt'].isin(['five people sunbathing on beach'])].loc[:, 'Total Score'].mean(numeric_only=True), ', variance ', df[df['Prompt'].isin(['five people sunbathing on beach'])].loc[:, 'Total Score'].var(numeric_only=True))
    print('mother or father holding baby, ', df[df['Prompt'].isin(['mother or father holding baby'])].loc[:, 'Total Score'].mean(numeric_only=True), ', variance ', df[df['Prompt'].isin(['mother or father holding baby'])].loc[:, 'Total Score'].var(numeric_only=True))

    print('Welch t-test beach vs baby holding ', scipy.stats.ttest_ind(df[df['Prompt'].isin(['five people sunbathing on beach'])].loc[:, 'Total Score'], df[df['Prompt'].isin(['mother or father holding baby'])].loc[:, 'Total Score'], equal_var=False))
    

    fig3.text(0.01, 0.99, 'A', fontsize=14)
    fig3.text(0.01, 0.49, 'B', fontsize=14)
    
    format_axes(fig3)
    # plt.tight_layout()
    plt.savefig('../Figure_3.png', dpi=300)

######### Figure 4
    plt.cla()

    fig4 = plt.figure(layout="constrained", figsize=(10, 5))

    gs = GridSpec(1, 2, figure=fig4)

    ax11 = fig4.add_subplot(gs[0])
    ax12 = fig4.add_subplot(gs[1])

    plot_error_level_models_type(combined_df,ax11)
    plot_error_level_models_body_part(combined_df, ax12)
    
    fig4.text(0.01, 0.96, 'A', fontsize=14)
    fig4.text(0.51, 0.96, 'B', fontsize=14)

    format_axes(fig4)
    # plt.tight_layout()
    plt.savefig('../Figure_4.png', dpi=300)


#########  Supl Figure 1
    plt.cla()
    sfig1 = plt.figure(layout="constrained", figsize=(15, 10))
    gs = GridSpec(2, 1, figure=sfig1)

    ax11 = sfig1.add_subplot(gs[0])
    ax12 = sfig1.add_subplot(gs[1])

    plot_prompt_error_type(combined_df, ax11)
    plot_prompt_body_type(combined_df, ax12)

    sfig1.text(0.01, 0.99, 'A', fontsize=14)
    sfig1.text(0.01, 0.49, 'B', fontsize=14)

    format_axes(sfig1)
    # plt.tight_layout()
    plt.savefig('../Suppl_Figure_1.png', dpi=300)

#########  Supl Figure 2
    plt.cla()
    sfig1 = plt.figure(layout="constrained", figsize=(25, 15))
    gs = GridSpec(3, 2, figure=sfig1)

    ax11 = sfig1.add_subplot(gs[0,0])
    ax12 = sfig1.add_subplot(gs[0,1])
    ax13 = sfig1.add_subplot(gs[1,0])
    ax14 = sfig1.add_subplot(gs[1,1])
    ax15 = sfig1.add_subplot(gs[2,0])


    plot_error_level_models_prompt(combined_df[combined_df['Error Type'].isin(['Configuration Error'])], ax12)
    plot_error_level_models_prompt(combined_df[combined_df['Error Type'].isin(['Orientation Error'])], ax13)
    plot_error_level_models_prompt(combined_df[combined_df['Error Type'].isin(['Proportion Error'])], ax14)
    plot_error_level_models_prompt(combined_df[combined_df['Error Type'].isin(['Missing Error'])], ax15)
    plot_error_level_models_prompt(combined_df[combined_df['Error Type'].isin(['Extra Error'])], ax11)

    ax12.set_title('Arregate of errors by severity and model and prompt for configuration error')
    ax13.set_title('Arregate of errors by severity and model and prompt for orientation error')
    ax14.set_title('Arregate of errors by severity and model and prompt for proportion error')
    ax15.set_title('Arregate of errors by severity and model and prompt for missing error')
    ax11.set_title('Arregate of errors by severity and model and prompt for extra error')

    sfig1.text(0.01, 0.98, 'A', fontsize=14)
    sfig1.text(0.51, 0.98, 'B', fontsize=14)
    sfig1.text(0.01, 0.64, 'C', fontsize=14)
    sfig1.text(0.51, 0.64, 'D', fontsize=14)
    sfig1.text(0.01, 0.34, 'E', fontsize=14)

    format_axes(sfig1)
    # plt.tight_layout()
    plt.savefig('../Suppl_Figure_2.png', dpi=300)

#########  Supl Figure 3
    plt.cla()
    sfig1 = plt.figure(layout="constrained", figsize=(25, 15))
    gs = GridSpec(3, 2, figure=sfig1)

    ax11 = sfig1.add_subplot(gs[0,0])
    ax12 = sfig1.add_subplot(gs[0,1])
    ax13 = sfig1.add_subplot(gs[1,0])
    ax14 = sfig1.add_subplot(gs[1,1])
    ax15 = sfig1.add_subplot(gs[2,0])


    plot_error_level_models_prompt(combined_df[combined_df['Body Part'].isin(['Face'])], ax12)
    plot_error_level_models_prompt(combined_df[combined_df['Body Part'].isin(['Feet'])], ax13)
    plot_error_level_models_prompt(combined_df[combined_df['Body Part'].isin(['Hands'])], ax14)
    plot_error_level_models_prompt(combined_df[combined_df['Body Part'].isin(['Torso'])], ax15)
    plot_error_level_models_prompt(combined_df[combined_df['Body Part'].isin(['Limbs'])], ax11)

    ax12.set_title('Arregate of errors by severity and model and prompt for face')
    ax13.set_title('Arregate of errors by severity and model and prompt for feet')
    ax14.set_title('Arregate of errors by severity and model and prompt for hands')
    ax15.set_title('Arregate of errors by severity and model and prompt for torso')
    ax11.set_title('Arregate of errors by severity and model and prompt for limbs')

    sfig1.text(0.01, 0.98, 'A', fontsize=14)
    sfig1.text(0.51, 0.98, 'B', fontsize=14)
    sfig1.text(0.01, 0.64, 'C', fontsize=14)
    sfig1.text(0.51, 0.64, 'D', fontsize=14)
    sfig1.text(0.01, 0.34, 'E', fontsize=14)

    format_axes(sfig1)
    # plt.tight_layout()
    plt.savefig('../Suppl_Figure_3.png', dpi=300)



def plot_cum_score_dist(df, ax):
    sns.histplot(df, x="Total Score", fill=True, color='grey', ax=ax)
    ax.set_xlabel('Cumulative score')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of cumulative scores')


def plot_krippendorff(all_krip, ax):

    patches = []
    for i in range(0,6):
        patches.append(mpatches.Patch(label=('Pair ' + str(i + 1))))
    plot_all_krip = pd.DataFrame(all_krip)
    plot_all_krip.columns = columns_all_krip
    sns.swarmplot(data=plot_all_krip, x='Type',y='Value',hue='Pair', palette=colours_krip, size=9, ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    labels = ['Pair 1','Pair 2','Pair 3','Pair 4','Pair 5','Pair 6']
    sns.lineplot(data=plot_all_krip, x='Type',y='Value',hue='Pair', palette=colours_krip, ax=ax,)
    ax.legend_.remove()
    ax.legend(handles,labels,loc='lower center',ncols=2, bbox_to_anchor=(0.6,0.02))
    ax.set_ylabel('Krippendorff\'s Alpha')
    ax.set_xlabel('Evaluation criteria')
    ax.set_title('Inter-rater agreement')


def plot_cum_score_dist_model(df, ax):

    sns.violinplot(x='Model', y='Total Score', data=df, palette=palette_models, inner='point',  gap=.1, cut = 0, fill=False, ax=ax)
    ax.set_xlabel('Model')
    ax.set_ylabel('Cumulative score')
    ax.set_title('Distribution of cumulative scores per model')


def plot_annotator_dist(data, ax):
    sns.violinplot(data=data, x="Annotator", y="Total Score", fill=False, gap=.1, cut=0, inner="point", palette=annotator_colours, ax=ax)
    ax.xaxis.set_ticklabels(['1', '2', '3', '4'])
    ax.set_xlabel('Annotator')
    ax.set_ylabel('Cumulative score')
    ax.set_title('Distribution of cumulative scores per annotator')

def plot_dist_overall_severity(scores_q, ax):

    plot_overall_sev = pd.DataFrame(scores_q)
    plot_overall_sev.columns = ['index','overall_sev','annotator','model']

    hue_order = ['low', 'medium','high']
    sns.countplot(data=plot_overall_sev, x='model',hue='overall_sev', hue_order=hue_order,ax=ax)

    # add shading
    for bar_container, shade in zip(ax.containers, [0.3, 0.6, 0.9]):
        for bar, group_cmap in zip(bar_container, ['Blues', 'Greens', 'Reds']):
            bar.set_color(plt.get_cmap(group_cmap)(shade))

    # Add total values on top of each bar
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    # custom legend
    custom_lines = [Line2D([0], [0], color=plt.get_cmap('Blues')(0.3), lw=4),
                    Line2D([0], [0], color=plt.get_cmap('Blues')(0.6), lw=4),
                    Line2D([0], [0], color=plt.get_cmap('Blues')(0.9), lw=4)]

    ax.legend(custom_lines, ['low', 'medium', 'high'])
            
    ax.set_xlabel('Model')
    ax.set_ylabel('Count')
    ax.set_title('Overall distribution of overall image severity')
    
def plot_error_level_dist_models(error_count_df, ax):
    # Sum cumulative error scores for each model
    model_count_summed_df = error_count_df.groupby('Model').sum().reset_index()

    # Side-by-side bar plot for models without weights
    model_count_melted_df = model_count_summed_df.melt(id_vars=['Model'], value_vars=['a', 'b', 'c'], var_name='Error Level', value_name='Counts')
    sns.barplot(data=model_count_melted_df, x='Model', y='Counts', hue='Error Level', palette=error_level_colours, ci=None, ax=ax)
    ax.set_xlabel('Model')
    ax.set_ylabel('Aggregate')    
    ax.set_title('Aggregate of errors by severity and model')

    # add shading
    for bar_container, shade in zip(ax.containers, [0.3, 0.6, 0.9]):
        for bar, group_cmap in zip(bar_container, colour_models):
            bar.set_color(plt.get_cmap(group_cmap)(shade))

    # custom legend
    custom_lines = [Line2D([0], [0], color=plt.get_cmap('Blues')(0.3), lw=4),
                    Line2D([0], [0], color=plt.get_cmap('Blues')(0.6), lw=4),
                    Line2D([0], [0], color=plt.get_cmap('Blues')(0.9), lw=4)]

    ax.legend(custom_lines, ['a', 'b', 'c'])

    # Add total values on top of each bar
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center', xytext=(0, 9), textcoords='offset points')

def plot_error_level_models_prompt(combined_df, ax):

    # 1. Prompt-based Error Distribution
    prompt_count_summed_df = combined_df.groupby(['Prompt', 'Error Level', 'Model']).sum().reset_index()
    pivot_df = prompt_count_summed_df.pivot_table(index=['Prompt', 'Error Level'], columns='Model', values='Counts').fillna(0)
    prompts = pivot_df.index.get_level_values(0).unique()
    error_levels = pivot_df.index.get_level_values(1).unique()

    bar_width = 0.25
    x = np.arange(len(prompts))

    for i, error_level in enumerate(error_levels):
        bottom = np.zeros(len(prompts))
        for model in pivot_df.columns:
            values = pivot_df.xs(error_level, level='Error Level')[model].values
            ax.bar(x + i * bar_width, values, bottom=bottom, width=bar_width, color=palette_models[model], label=model if i == 0 else "")
            bottom += values
            if model == list(pivot_df.columns)[-1]: 
                for j, val in enumerate(bottom):
                    ax.text(j + i * bar_width, val + 1, f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    for idx, prompt in enumerate(prompts):
        for i, level in enumerate(error_levels):
            ax.text(idx + i * bar_width + bar_width/6, 0, level, ha='right', va='top', fontsize=9, color='black', rotation=0)

    # add shading
    all_colours_shades = []
    for shade in [0.3,0.6,0.9]:
        for colour in colour_models:
            all_colours_shades.append(plt.get_cmap(colour)(shade))
    i = 0
    for bar_container in ax.containers:
        for bar in bar_container:
            bar.set_color(all_colours_shades[i])
        i += 1
 

    ax.set_xticks(x + bar_width * len(error_levels) / 3)
    ax.set_xticklabels(prompts, rotation=45, ha='right')
    ax.tick_params(axis='x', which='major', pad=10)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Models', fontsize='small', title_fontsize='small', loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0.)

    ax.set_xlabel('Prompts and error severity')
    ax.set_ylabel('Aggregate')
    ax.set_title('Aggregate of errors by severity and model and prompt')

    # ax.set_xticklabels(ax.get_xticklabels(),rotation=45, ha='right')
        
def plot_error_level_models_type(combined_df, ax):

    error_type_count_summed_df = combined_df.groupby(['Error Type', 'Error Level', 'Model']).sum().reset_index()
    pivot_df = error_type_count_summed_df.pivot_table(index=['Error Type', 'Error Level'], columns='Model', values='Counts').fillna(0)
    error_types = pivot_df.index.get_level_values(0).unique()
    error_levels = pivot_df.index.get_level_values(1).unique()

    bar_width = 0.25
    x = np.arange(len(error_types))

    for i, error_level in enumerate(error_levels):
        bottom = np.zeros(len(error_types))
        for model in pivot_df.columns:
            values = pivot_df.xs(error_level, level='Error Level')[model].values
            ax.bar(x + i * bar_width, values, bottom=bottom, width=bar_width, color=palette_models[model], label=model if i == 0 else "")
            bottom += values
            if model == list(pivot_df.columns)[-1]: 
                for j, val in enumerate(bottom):
                    ax.text(j + i * bar_width, val + 1, f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    for idx, error_type in enumerate(error_types):
        for i, level in enumerate(error_levels):
            ax.text(idx + i * bar_width + bar_width/6, 0, level, ha='right', va='top', fontsize=9, color='black', rotation=0)

    # add shading
    all_colours_shades = []
    for shade in [0.3,0.6,0.9]:
        for colour in colour_models:
            all_colours_shades.append(plt.get_cmap(colour)(shade))
    i = 0
    for bar_container in ax.containers:
        for bar in bar_container:
            bar.set_color(all_colours_shades[i])
        i += 1
            
    ax.set_xticks(x + bar_width * len(error_levels) / 3)
    ax.set_xticklabels(pd.Index(error_types.to_series().replace({'Configuration Error': 'Configuration\n error', 'Extra Error': 'Extra\n error', 'Missing Error': 'Missing\n error', 'Orientation Error': 'Orientation\n error', 'Proportion Error':'Proportion\n error'})), rotation=0, ha='center')
    ax.tick_params(axis='x', which='major', pad=10)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Models', fontsize='small', title_fontsize='small', loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0.)

    ax.set_xlabel('Error type and error severity')
    ax.set_ylabel('Aggregate')
    ax.set_title('Aggregate of errors by severity/model/type')

def plot_error_level_models_body_part(combined_df, ax):

    body_part_count_summed_df = combined_df.groupby(['Body Part', 'Error Level', 'Model']).sum().reset_index()
    pivot_df = body_part_count_summed_df.pivot_table(index=['Body Part', 'Error Level'], columns='Model', values='Counts').fillna(0)
    body_parts = pivot_df.index.get_level_values(0).unique()
    error_levels = pivot_df.index.get_level_values(1).unique()

    bar_width = 0.25
    x = np.arange(len(body_parts))

    for i, error_level in enumerate(error_levels):
        bottom = np.zeros(len(body_parts))
        for model in pivot_df.columns:
            values = pivot_df.xs(error_level, level='Error Level')[model].values
            ax.bar(x + i * bar_width, values, bottom=bottom, width=bar_width, color=palette_models[model], label=model if i == 0 else "")
            bottom += values
            if model == list(pivot_df.columns)[-1]: 
                for j, val in enumerate(bottom):
                    ax.text(j + i * bar_width, val + 1, f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    for idx, body_part in enumerate(body_parts):
        for i, level in enumerate(error_levels):
            ax.text(idx + i * bar_width + bar_width/6, 0, level, ha='right', va='top', fontsize=9, color='black', rotation=0)
            
    ax.set_xticks(x + bar_width * len(error_levels) / 3)
    ax.set_xticklabels(body_parts, rotation=0, ha='center')
    ax.tick_params(axis='x', which='major', pad=10)

    # add shading
    all_colours_shades = []
    for shade in [0.3,0.6,0.9]:
        for colour in colour_models:
            all_colours_shades.append(plt.get_cmap(colour)(shade))
    
    i = 0
    for bar_container in ax.containers:
        for bar in bar_container:
            bar.set_color(all_colours_shades[i])
        i += 1

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Models', fontsize='small', title_fontsize='small', loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0.)

    ax.set_xlabel('Body part and error severity')
    ax.set_ylabel('Aggregate')
    ax.set_title('Aggregate of errors by severity/model/body part')

def plot_prompt_cum_dist(df, ax):

    # Add a category column based on the prompts
    df['Category'] = df['Prompt'].apply(lambda x: 'Individual' if x in individual_prompts else 'Pair' if x in pair_prompts else 'Group')

    # Melt the DataFrame for visualization of violin plots for prompts
    prompt_melted_df = df.melt(id_vars=['Prompt', 'Category'], value_vars=['Total Score'], value_name='Scores')

    # Define the colors for the categories
    category_colors = {'Pair': 'C0', 'Group': 'C1', 'Individual': 'C2'}
    platte = ['C0', 'C1', 'C2']

    # Plotting violin plots for cumulative scores per prompt using Set2 palette
    sns.violinplot(x='Prompt', y='Scores', hue='Category', data=prompt_melted_df, palette=platte, inner='point', linewidth=1, cut=0, fill=False, ax=ax)

    # Rotate the x-tick labels to make them more readable without wrapping
    labels = prompt_melted_df['Prompt'].unique()
    ax.set_xticks(ticks=range(len(labels)), labels=labels, rotation=45, ha='right')

    # Change the color of legend labels to match the category
    for i, text in enumerate(ax.get_legend().get_texts()):
        text.set_color(platte[i])

     # Add a smaller legend to the rightmost side of the plot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Error types', fontsize='small', title_fontsize='small', loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0.)
    ax.set_ylabel('Cumulative score')
    ax.set_title('Distribution of cumulative scores per prompt')   

def plot_prompt_error_type(combined_df, ax):

    # Aggregate the counts per prompt, error level, and error type
    prompt_error_type_summed_df = combined_df.groupby(['Prompt', 'Error Level', 'Error Type']).sum().reset_index()
    pivot_df = prompt_error_type_summed_df.pivot_table(index=['Prompt', 'Error Level'], columns='Error Type', values='Counts').fillna(0)
    prompts = pivot_df.index.get_level_values(0).unique()
    # prompts = (individual_prompts + pair_prompts + group_prompts)
    error_levels = pivot_df.index.get_level_values(1).unique()
    error_types = pivot_df.columns

    # Plotting
    bar_width = 0.25
    x = np.arange(len(prompts))

    # Iterate over each error type
    for i, error_level in enumerate(error_levels):
        bottom = np.zeros(len(prompts))
        for error_type in error_types:
            values = pivot_df.xs(error_level, level='Error Level')[error_type].values
            ax.bar(x + i * bar_width, values, bottom=bottom, width=bar_width, color=error_type_palette[error_type], label=error_type if i == 0 else "")
            bottom += values
            if error_type == list(error_types)[-1]:  # Only annotate once per bar group
                for j, val in enumerate(bottom):
                    ax.text(j + i * bar_width, val + 1, f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    # Add labels "a", "b", "c" below each bar group
    for idx, prompt in enumerate(prompts):
        for i, level in enumerate(error_levels):
            ax.text(idx + i * bar_width + bar_width/6, 0, level, ha='right', va='top', fontsize=9, color='black', rotation=0)

    # Customize the plot
    ax.set_xticks(x + bar_width * len(error_levels) / 3)
    ax.set_xticklabels(prompts, rotation=45, ha='right')
    ax.tick_params(axis='x', which='major', pad=10)

    # add shading
    all_colours_shades = []
    for shade in [0.3,0.6,0.9]:
        for _,colour in error_type_colors.items():
            all_colours_shades.append(plt.get_cmap(colour)(shade))
    
    i = 0
    for bar_container in ax.containers:
        for bar in bar_container:
            bar.set_color(all_colours_shades[i])
        i += 1

    # Add a smaller legend to the rightmost side of the plot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Error types', fontsize='small', title_fontsize='small', loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0.)

    ax.set_xlabel('Prompt and error severity')
    ax.set_ylabel('Aggregate')
    ax.set_title('Aggregate of errors by type and body type and prompt')

def plot_prompt_body_type(combined_df, ax):
    # Aggregate the counts per prompt, error level, and body part
    prompt_body_part_summed_df = combined_df.groupby(['Prompt', 'Error Level', 'Body Part']).sum().reset_index()
    pivot_df = prompt_body_part_summed_df.pivot_table(index=['Prompt', 'Error Level'], columns='Body Part', values='Counts').fillna(0)
    prompts = pivot_df.index.get_level_values(0).unique()
    error_levels = pivot_df.index.get_level_values(1).unique()
    body_parts = pivot_df.columns

    # Plotting
    bar_width = 0.25
    x = np.arange(len(prompts))

    # Iterate over each error level
    for i, error_level in enumerate(error_levels):
        bottom = np.zeros(len(prompts))
        for body_part in body_parts:
            values = pivot_df.xs(error_level, level='Error Level')[body_part].values
            ax.bar(x + i * bar_width, values, bottom=bottom, width=bar_width, color=body_part_palette[body_part], label=body_part if i == 0 else "")
            bottom += values
            if body_part == list(body_parts)[-1]:  # Only annotate once per bar group
                for j, val in enumerate(bottom):
                    ax.text(j + i * bar_width, val + 1, f'{val:.2f}', ha='center', va='bottom', fontsize=7)

    # Add labels "a", "b", "c" below each bar group
    for idx, prompt in enumerate(prompts):
        for i, level in enumerate(error_levels):
            ax.text(idx + i * bar_width + bar_width / 6, 0, level, ha='right', va='top', fontsize=9, color='black', rotation=0)

    # Customize the plot
    ax.set_xticks(x + bar_width * len(error_levels) / 3)
    ax.set_xticklabels(prompts, rotation=45, ha='right')
    ax.tick_params(axis='x', which='major', pad=10)

    # add shading
    all_colours_shades = []
    for shade in [0.3,0.6,0.9]:
        for _,colour in body_part_colors.items():
            all_colours_shades.append(plt.get_cmap(colour)(shade))
    
    i = 0
    for bar_container in ax.containers:
        for bar in bar_container:
            bar.set_color(all_colours_shades[i])
        i += 1

    # Add a smaller legend to the rightmost side of the plot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Body parts', fontsize='small', title_fontsize='small', loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0.)

    ax.set_xlabel('Prompt and error severity')
    ax.set_ylabel('Aggregate')
    ax.set_title('Aggregate of errors by severity and prompt and body part')

def format_axes(fig):
    for i, ax in enumerate(fig.axes):
        # ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

if __name__ == '__main__':
    main()