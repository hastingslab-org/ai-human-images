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
import math
from scipy.interpolate import splev, splrep
import datetime as dt
from scipy.special import rel_entr
from fractions import Fraction
import pandas as pd
from pathlib import Path
from textwrap import wrap
from matplotlib.gridspec import GridSpec
import re

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

columns_all_krip = ['Pair', 'Type', 'Value']

# Define custom colors for each model
palette_models = {"dall-e3": "lightblue", "sdxl": "lightsalmon", "stablecascade": "lightcoral"}

annotator_colours = {'anno1': '#F6C06E','anno3':'#ffd670', 'anno4':'#8F5A00','anno2':'#d76b48'}
colours_krip = {'anno1-anno2': '#e7965b', 'anno1-anno4': '#c38d37', 'anno3-anno4': '#c79838', 'anno2-anno3': '#eba15c', 'anno2-anno4': '#b36324', 'anno1-anno3': '#fbcb6f'}

error_level_colours = {"a": "darkgreen", "b": "gold", "c": "darkred"}
error_sev_colours = {"low": "darkgreen", "medium": "gold", "high": "darkred"}
# Define custom colors for each error type
error_type_palette = sns.color_palette("Set2", 5)
error_type_colors = {
    'Missing Error': error_type_palette[0],
    'Extra Error': error_type_palette[1],
    'Configuration Error': error_type_palette[2],
    'Orientation Error': error_type_palette[3],
    'Proportion Error': error_type_palette[4]  # Adjust this according to your actual error types
}

# Define custom colors for each body part
body_part_palette = sns.color_palette("Set2", 5)
body_part_colors = {
    'Torso': body_part_palette[0],
    'Limbs': body_part_palette[1],
    'Feet': body_part_palette[2],
    'Hands': body_part_palette[3],
    'Face': body_part_palette[4]  # Adjust this according to your actual body parts
}

def main():

    df, df2, combined_df = get_data()
    # get krippendorff stats
    krippendorff_results = inter_rater_agreement(df)
    dist_overall_sev = get_overall_sev_dist(df)
    error_count_df = get_data_error_level(df2)
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

    fig = plt.figure(layout="constrained", figsize=(20, 25))

    gs = GridSpec(12, 3, figure=fig)
    ax11 = fig.add_subplot(gs[0:3, 0])
    ax12 = fig.add_subplot(gs[0:3, 1])
    ax13 = fig.add_subplot(gs[0:3, -1])
    # identical to ax1 = plt.subplot(gs.new_subplotspec((0, 0), colspan=3))
    ax21 = fig.add_subplot(gs[3:6, 0])
    ax22 = fig.add_subplot(gs[3:6, 1])
    ax23 = fig.add_subplot(gs[3:6, -1])
    ax31 = fig.add_subplot(gs[6:8, 0:-1])
    ax32 = fig.add_subplot(gs[6:8, -1])
    ax41 = fig.add_subplot(gs[8:10, 0:-1])
    ax42 = fig.add_subplot(gs[8:10, -1])
    ax51 = fig.add_subplot(gs[10:13, 0:3])

    plot_cum_score_dist(df, ax11)
    plot_annotator_dist(df, ax12)
    plot_krippendorff(krippendorff_results,ax13)
    plot_cum_score_dist_model(df, ax21)
    plot_dist_overall_severity(dist_overall_sev, ax22)
    plot_error_level_dist_models(error_count_df, ax23)
    plot_error_level_models_prompt(combined_df, ax31)
    plot_error_level_models_type(combined_df,ax32)
    plot_error_level_models_body_part(combined_df, ax42)
    plot_prompt_error_type(combined_df, ax41)
    plot_prompt_body_type(combined_df, ax51)
    
    # fig.suptitle("GridSpec")
    format_axes(fig)
    # plt.tight_layout()
    plt.savefig('grid.png', dpi=300)

def plot_cum_score_dist(df, ax):
    sns.histplot(df, x="Total Score", fill=True, color='grey', ax=ax)
    ax.set_xlabel('Cumulative Score')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of cumulative scores.')


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
    ax.legend(handles,labels,loc='lower right',ncols=2)
    ax.set_ylabel('Krippendorff\'s Alpha')
    ax.set_xlabel('Evaluation Criteria')
    ax.set_title('Inter-rater agreement')


def plot_cum_score_dist_model(df, ax):

    sns.violinplot(x='Model', y='Total Score', data=df, palette=palette_models, inner='point',  gap=.1, cut = 0, fill=False, ax=ax)
    ax.set_xlabel('Model')
    ax.set_ylabel('Cumulative score')
    ax.set_title('Distribution of cumulative scores per model.')


def plot_annotator_dist(data, ax):
    sns.violinplot(data=data, x="Annotator", y="Total Score", fill=False, gap=.1, cut=0, inner="point", palette=annotator_colours, ax=ax)
    ax.xaxis.set_ticklabels(['1', '2', '3', '4'])
    ax.set_xlabel('Annotator')
    ax.set_ylabel('Cumulative score')
    ax.set_title('Distribution of cumulative scores per annotator.')

def plot_dist_overall_severity(scores_q, ax):

    plot_overall_sev = pd.DataFrame(scores_q)
    plot_overall_sev.columns = ['index','overall_sev','annotator','model']
    # plot_overall_sev_melted = plot_overall_sev.melt(id_vars=['model'], value_vars=['low', 'medium', 'high'], var_name='overall_sev', value_name='overall_sev')

    # sns.histplot(data=plot_overall_sev, x='model', hue='overall_sev',multiple='stack', ax=ax)
    hue_order = ['low', 'medium','high']
    sns.countplot(data=plot_overall_sev, x='model',hue='overall_sev',palette=error_sev_colours, hue_order=hue_order,ax=ax)
    # Add total values on top of each bar
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center', xytext=(0, 9), textcoords='offset points')
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
    ax.set_ylabel('Count')    
    ax.set_title('Error Distribution for Each Model')

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
            
    ax.set_xticks(x + bar_width * len(error_levels) / 3)
    ax.set_xticklabels(prompts, rotation=45, ha='right')
    ax.tick_params(axis='x', which='major', pad=10)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Models', fontsize='small', title_fontsize='small', loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0.)

    ax.set_xlabel('Prompts and Error Severity')
    ax.set_ylabel('Count')
    ax.set_title('Count of Errors per Error Severity per Model per Prompt')

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
            
    ax.set_xticks(x + bar_width * len(error_levels) / 3)
    ax.set_xticklabels(error_types, rotation=45, ha='right')
    ax.tick_params(axis='x', which='major', pad=10)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Models', fontsize='small', title_fontsize='small', loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0.)

    ax.set_xlabel('Error Type and Error Severity')
    ax.set_ylabel('Count')
    ax.set_title('Count of Errors per Error Severity per Model per Error Type')

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

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Models', fontsize='small', title_fontsize='small', loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0.)

    ax.set_xlabel('Body Part and Error Severity')
    ax.set_ylabel('Count')
    ax.set_title('Count of Errors per Error Severity per Model per Body Part')

def plot_prompt_error_type(combined_df, ax):

    # Aggregate the counts per prompt, error level, and error type
    prompt_error_type_summed_df = combined_df.groupby(['Prompt', 'Error Level', 'Error Type']).sum().reset_index()
    pivot_df = prompt_error_type_summed_df.pivot_table(index=['Prompt', 'Error Level'], columns='Error Type', values='Counts').fillna(0)
    prompts = pivot_df.index.get_level_values(0).unique()
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
            ax.bar(x + i * bar_width, values, bottom=bottom, width=bar_width, color=error_type_colors[error_type], label=error_type if i == 0 else "")
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

    # Add a smaller legend to the rightmost side of the plot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Error Types', fontsize='small', title_fontsize='small', loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0.)

    ax.set_xlabel('Prompt and Error Severity')
    ax.set_ylabel('Count')
    ax.set_title('Count of Errors per Error Type per Model per Prompt')

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
            ax.bar(x + i * bar_width, values, bottom=bottom, width=bar_width, color=body_part_colors[body_part], label=body_part if i == 0 else "")
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

    # Add a smaller legend to the rightmost side of the plot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Body Parts', fontsize='small', title_fontsize='small', loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0.)

    ax.set_xlabel('Prompt and Error Severity')
    ax.set_ylabel('Count')
    ax.set_title('Count of Errors per Error Severity per Model per Body Part')

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