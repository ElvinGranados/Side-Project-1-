import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

school_pro1 = pd.read_csv('student_por.csv')
age_c1 = {}
age_l1 = []
avg_gpa = []
count = 0

for row in school_pro1['age']:
    age_c1[row] = age_c1.get(row,0)+1
    age_l1.append(row)
    count += 1

for i in range(school_pro1.shape[0]):
    val = school_pro1.loc[i,['g1','g2','g3']].mean()
    avg_gpa.append(round(val,2))

school_pro1['avg_gpa'] = avg_gpa

print (count)
'''A histogram of student distribution in this program using Matplotlib'''
plt.hist(age_l1,len(age_c1),color='red')
plt.title('Age of Students in Portuguese Program')
plt.xlabel('Age')
plt.ylabel('Occurrences')
plt.show()

'''I decided to set up an independence test and figure out if there was any relationship
between the status of parents (together or separated) and the child's answers to the yes/no questions.
I tested my hypothesis using the chi squared test at a 5% significance level and setting up 2x2 contigency
tables and using the scipy module to help with the calculations'''

change = ['famsup','higher','romantic','activities','paid','schoolsup','internet','nursery']
for i in change:
    yes_bool = school_pro1[i] == True
    no_bool = school_pro1[i] == False
    school_pro1.loc[yes_bool,i] = 'Yes'
    school_pro1.loc[no_bool,i] = 'No'

for i in change:
    cont_t = pd.crosstab(school_pro1[i],school_pro1['pstatus'])
    (t,p,dof,expected) = stats.chi2_contingency(cont_t)
    chi_c = stats.chi2.isf(q=0.05,df=1)
    if t > chi_c:
        print('There is a statistic significance for {}!'.format(i))
        print(round(t,3),round(chi_c,3))

'''From the data, there is statistical significance between the status of the parents and
extra curricular activities the child participates in which isn't due to chance. This may be
information to delve further into. Other than that the answers to the yes/no questions are not
dependent on marital status of parents'''

'''I wanted to figure out if any other factors within the data set could explain this discovery
One of my hypotheses is that the student is off working to support the family or might be overall
unhealthy due to stress or mental illness'''

filter_bool = school_pro1['pstatus'] == 'A'
filter_bool2 = school_pro1['pstatus'] == 'T'
df_apart = school_pro1.loc[filter_bool]
df_together = school_pro1.loc[filter_bool2]
fam_a_size = df_apart['famsize'].value_counts()
fam_t_size = df_together['famsize'].value_counts()
a_size = fam_a_size/fam_a_size.sum()
t_size = fam_t_size/fam_t_size.sum()
print(a_size)
print(t_size)

ratings = np.arange(1,6)
numeric = ['famrel','freetime','goout','dalc','walc','health']
fig = plt.figure(figsize=(11.5,6))
i = 1

'''I used Matplotlib to graph the ratings students gave in regards to topics
such as family relationship, health, alcohol consumption'''

for column in numeric:
    a_stats = df_apart[column].describe()
    t_stats = df_together[column].describe()
    a_freq = df_apart[column].value_counts()
    a_freq = a_freq/a_freq.sum()
    t_freq = df_together[column].value_counts()
    t_freq = t_freq/t_freq.sum()
    X1 = []
    X2 = []
    for rating in ratings:
        X1.append(a_freq[rating])
        X2.append(t_freq[rating])
    ax = fig.add_subplot(2,3,i)
    ax.bar(ratings-0.125,X1,color='blue',width=0.25,label='apart')
    ax.bar(ratings+0.125,X2,color='red',width=0.25,label='together')
    ax.legend(loc='best')
    i += 1
plt.show()

'''From the top left to the bottom right subplots, the bar graphs represent ratings for
family relationship(1-Very Bad to 5-Excellent), freetime available (1-Very Low to 5-Very High),
going out (1-Very Low to 5-Very High), Weekday Alcohol Consumption (1-Very Low to 5-Very High),
Weekend Alcohol Consumption (1-Very Low to 5-Very High) and Current Health status
(1-Very Bad to 5-Very Good) respectively. The blue bars represent students whose parents are separated
and the red bars represent students whose parents are together'''

'''From the descriptive statistics, all of the sample means of ratings are lower for the group whose parents are
separated versus those who are together'''

'''At first glance, there doesn't appear to be any significant information from the charts and the descriptive
statistics since both groups tend to show similar numbers except for the current health status where the students
whose parents are separated have reported more very bad ratings. I'd like to check the data a bit
further by performing a Mann-Whitney Test given that I can't assume the two groups are normally
distributed and to see if there is indeed a difference between the groups not due to chance.'''

for column in numeric:
    alpha = 0.05
    (s,p) = stats.mannwhitneyu(df_apart[column],df_together[column],use_continuity=True,alternative='two-sided')
    if p < alpha:
        print('The groups are different in distribution!')

'''Let us assume that our two groups are normally distributed and that we can run a two sample t test'''

for column in numeric:
    alpha = 0.05
    (s,p) = stats.ttest_ind(df_apart[column],df_together[column],equal_var=False)
    if p < alpha:
        print ('There is a difference between the groups!')

'''We conclude that we fail to reject that there is no difference between the groups so student's ratings are
not dependent on whether their parents are together or apart. I'll turn to commute time, hours
studied per week and the education of parents to help in answering my hypothesis running the same
procedure as before'''

compare = ['traveltime','studytime','medu','fedu']
for column in compare:
    alpha = 0.05
    stat_a = df_apart[column].describe()
    stat_t = df_together[column].describe()
    print (stat_a)
    print (stat_t)
    (s1,p1) = stats.mannwhitneyu(df_apart[column],df_together[column],use_continuity=True,alternative='two-sided')
    if p1 < alpha:
        print('Groups are different in distribution!')
    (s2,p2) = stats.ttest_ind(df_apart[column],df_together[column],equal_var=False)
    if p2 < alpha:
        print('There is a difference between the groups!')

'''An interesting note to make is that the sample means of education for both parents in the apart group
are higher than those for the together group but there is no significant difference. From these tests, it
appears that there is a confounding factor that would explain the chi squared result obtained in the
earlier part of this script. From prior research, factors such as household income, social/psychological/cognitive
implications, homelife, etc. would distinct the two groups of interest. The dataset does attempt to brush on
these topics however there is no clear distinction, a suggestion to improve the data is to figure out if students are
working a job which could explain why they would not be able to participate in extracurricular activities as well
as indicate levels of mental maturity if priorities are being shifted between family support/school life.'''

'''------------------------------------------------------------------------------'''

'''For the next part of this data analysis, I'd like to delve into alcohol consumption and figure out
if it has any effects on the students of the program. I will graph the level of alcohol drinking a student participates
in versus their average GPA for all 3 marking periods assuming that that student has been at his/her current
level of drinking for sometime'''


fig = plt.figure(figsize=(11,6))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.scatter(school_pro1['dalc'],school_pro1['avg_gpa'],color='green',label='weekday alcohol')
ax1.legend(loc='best')
ax2.scatter(school_pro1['walc'],school_pro1['avg_gpa'],color='blue',label='weekend alcohol')
ax2.legend(loc='best')
plt.ylabel('Average GPA of Student for Semester')
plt.show()

'''So the graphs do show some sign of negative correlation between average GPA and alcohol consumption. Let
us run a pairwise correlation between ordinal/numeric columns and find any interesting observations for all the
data'''

corr_df = school_pro1[['medu','fedu','studytime','failures','famrel','freetime','goout',
'dalc','walc','health','absences','avg_gpa']]

'''The following line of code gives us a matrix of Pearson correlation coefficient values and a graphical
representation of the data. The stronger the correlation, the more of a linear relationship present.'''

corr_table = corr_df.corr()
print (corr_table)
matrix = pd.plotting.scatter_matrix(corr_df,figsize=(14,7))
for ax in matrix.ravel():
    ax.set_xlabel(ax.get_xlabel(),fontsize = 10, rotation = 90)
    ax.set_ylabel(ax.get_ylabel(),fontsize = 10, rotation = 0)
plt.show()

'''I also ran a Spearman Rank Order correlation since I assumed the data to not be normally distributed
and the data is ordinal with a 1-5 scaling. From the result, it turns out the Spearman coefficient values are close to
Pearson coefficient values but this is just to check if the variables could have a non-linear relationship as
well as check the sensitivity of the Pearson r value due to outliers. The p value calculated tell us whether the
relationship is statistically significant'''

rho,p = stats.spearmanr(corr_df)
rho = pd.DataFrame(rho,index=list(corr_df),columns=list(corr_df))
pval = pd.DataFrame(p,index=list(corr_df),columns=list(corr_df))
print(rho)
print(pval)

'''Using pandas I made the array look more presentable and clean to users'''

'''Let us observe some columns and note any significance within the matrix. As a rule of
thumb we will only look at the values whose magnitude is greater than 0.3 which would indicate a low correlation.
Anything lower would be considered negligible or very weak.'''

check = ['medu','fedu','studytime','failures','famrel','freetime','goout',
'dalc','walc','health','absences','avg_gpa']

for item in check:
    pos_bool = rho[item] > 0.3
    neg_bool = rho[item] < -0.3
    res1 = rho.loc[pos_bool,item]
    res2 = rho.loc[neg_bool,item]
    print (res1)
    print (res2)

'''Regarding alcohol, there is no statistical significance that drinking is having a
detrimental effect on the students' life; however from the matrix, we can observe weak correlations (r = 0.35)
between student freetime and going out which would be typical adolescent behavior to be social with friends.
There is also a correlation (r = 0.37) between going out and weekend drinking which could tie into the social
aspect of students enjoying themselves and is most likely done recreationally. Two decently strong correlations
occurred. The first being with the education of both parents (r = 0.65); the more educated an individual is, the more
likely they would be with a partner of similar education. This would coincide with prior psychological research
which indicated intellect/resources being a long term att of partner seeking. The other correlation (r = 0.61) occurred
between weekday and weekend drinking; if an individual indulges in frequent alcohol consumption, it is likely
they would be drinking most days of the week whether school is in session or not. Given that the data is ordinal
it is difficult to measure between a rating of 4-5 as an example to make definite conclusions'''

'''If improvements were to be made to the dataset to really understand the influence of alcohol, we would
want to set up a timed experiment with two groups (experimental vs. control groups) of equal size and follow them
for approximately a semester asking them questions similar to those presented in the dataset. Afterwards, we could
run a two sample dependent test to measure any differences such as school GPA, social life, family life, etc.'''
