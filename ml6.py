import pandas as pd
import numpy as np

data = pd.DataFrame(data=pd.read_csv('human.csv'))
person = pd.DataFrame(data=pd.read_csv('person.csv'))

n_male = data['Gender'][data['Gender'] == 'male'].count()
n_female = data['Gender'][data['Gender'] == 'female'].count()
total_ppl = data['Gender'].count()

P_male = n_male/total_ppl
P_female = n_female/total_ppl
data_means = data.groupby('Gender').mean()
data_variance = data.groupby('Gender').var()

male_height_mean = data_means['Height'][data_variance.index == 'male'].values[0]
male_weight_mean = data_means['Weight'][data_variance.index == 'male'].values[0]
male_footsize_mean = data_means['Foot_Size'][data_variance.index== 'male'].values[0]

male_height_variance = data_variance['Height'][data_variance.index == 'male'].values[0]
male_weight_variance = data_variance['Weight'][data_variance.index == 'male'].values[0]
male_footsize_variance = data_variance['Foot_Size'][data_variance.index == 'male'].values[0]

female_height_mean = data_means['Height'][data_variance.index == 'female'].values[0]
female_weight_mean = data_means['Weight'][data_variance.index == 'female'].values[0]
female_footsize_mean = data_means['Foot_Size'][data_variance.index == 'female'].values[0]


female_height_variance = data_variance['Height'][data_variance.index == 'female'].values[0]
female_weight_variance = data_variance['Weight'][data_variance.index == 'female'].values[0]
female_footsize_variance = data_variance['Foot_Size'][data_variance.index == 'female'].values[0]

def p_x_given_y(x, mean_y, variance_y): 

    p = 1/(np.sqrt(2*np.pi*variance_y)) * np.exp((-(x-mean_y)**2)/(2*variance_y)) 
    return p

PMale = P_male * p_x_given_y(person['Height'][0],male_height_mean, male_height_variance) * p_x_given_y(person['Weight'][0], male_weight_mean, male_weight_variance) * p_x_given_y(person['Foot_Size'][0], male_footsize_mean, male_footsize_variance)

PFemale = P_female * p_x_given_y(person['Height'][0], female_height_mean, female_height_variance) * p_x_given_y(person['Weight'][0], female_weight_mean, female_weight_variance) * p_x_given_y(person['Foot_Size'][0], female_footsize_mean, female_footsize_variance)

if(PMale > PFemale):
    print("The given data belongs to Male with Probability of ",PMale)
else:
    print("The given data belongs to Female with Probability of ",PFemale)
