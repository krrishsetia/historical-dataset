import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sklearn as sl
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


pd.options.display.max_columns = 2
pd.options.display.max_rows = 10000000



data = pd.read_csv('ages_dataset.csv')


def int_conversion(var):
    a = var.isnumeric()
    return int(a)
def int_conversion_(var):
    return int(var)

def count(var):
    if var == 'Male':
        return 0
    elif var == 'Female':
        return 1
    else:return 2

def cause_of_death(var):
    if type(var) == str:
        a = var.split(';')
    else:a = [var]
    if a[0] == 'unnatural death':
        return 0
    elif a[0] == 'summary execution':
        return 1
    elif a[0] == 'suicide':
        return 2
    elif a[0] == 'natural causes':
        return 3
    elif a[0] == 'internal bleeding':
        return 4
    elif a[0] == 'homicide':
        return 5
    elif a[0] == 'gunshot':
        return 6
    elif a[0] == 'extrajudicial killing':
        return 7
    elif a[0] == 'euthanasia':
        return 8
    elif a[0] == 'era':
        return 9
    elif a[0] == 'epilepsy':
        return 10
    elif a[0] == 'death in battle':
        return 11
    elif a[0] == 'complication':
        return 12
    elif a[0] == 'cardiac arrest':
        return 13
    elif a[0] == 'capital punishment':
        return 14
    elif a[0] == 'accident':
        return 15
    elif a[0] == 'Pospíšil':
        return 16
    elif a[0] == 'Category:People executed by firing squad':
        return 17
    else:return 18
profession_count = []
def profession(var):
    if type(var) == str:
        a = var.split(';')
        for i in range(len(a)):
            a[i].isspace()
            profession_count.append(a[i])
    else:a = [var]
    if a[0] == 'Activist':
        return 0
    elif a[0] == 'Advocate':
        return 1
    elif a[0] == 'Air force officer':
        return 2
    elif a[0] == 'Airman':
        return 3
    elif a[0] == 'Alpine skier':
        return 4
    elif a[0] == 'Amateur wrestler':
        return 5
    elif a[0] == 'Anarcho-syndicalist':
        return 6
    elif a[0] == 'Anatomist':
        return 7
    elif a[0] == 'Animator':
        return 8
    elif a[0] == 'Anthropologist':
        return 9
    elif a[0] == 'Antiquarian seller':
        return 10
    elif a[0] == 'Architect':
        return 11
    elif a[0] == 'Aristocrat':
        return 12
    elif a[0] == 'Art collector':
        return 13
    elif a[0] == 'Art dealer':
        return 14
    elif a[0] == 'Artist':
        return 15
    elif a[0] == 'Assassin':
        return 16
    elif a[0] == 'Astrologer':
        return 17
    elif a[0] == 'Astronaut':
        return 18
    elif a[0] == 'Astronomer':
        return 19
    elif a[0] == 'Athlete':
        return 20
    elif a[0] == 'Author':
        return 21
    elif a[0] == 'Autobiographer':
        return 22
    elif a[0] == 'Ballet dancer':
        return 23
    elif a[0] == 'Bank robber':
        return 24
    elif a[0] == 'Banker':
        return 25
    elif a[0] == 'Bartender':
        return 26
    elif a[0] == 'Bhikkhu':
        return 27
    elif a[0] == 'Biologist':
        return 28
    elif a[0] == 'Bobsledder':
        return 29
    elif a[0] == 'Bodybuilder':
        return 30
    elif a[0] == 'Breeder':
        return 31
    elif a[0] == 'Brewer':
        return 32
    elif a[0] == 'Brother':
        return 33
    elif a[0] == 'Business executive':
        return 34
    elif a[0] == 'Businessperson':
        return 35
    elif a[0] == 'Butcher':
        return 36
    elif a[0] == 'Cabinetmaker':
        return 37
    elif a[0] == 'Calligrapher':
        return 38
    elif a[0] == 'Canoeist':
        return 39
    elif a[0] == 'Canon':
        return 40
    elif a[0] == 'Carpenter':
        return 41
    elif a[0] == 'Cartographer':
        return 42
    elif a[0] == 'Caudillo':
        return 43
    elif a[0] == 'Chairperson':
        return 44
    elif a[0] == 'Chamberlain':
        return 45
    elif a[0] == 'Chef':
        return 46
    elif a[0] == 'Art Christian mystic':
        return 47
    elif a[0] == 'Christians jehovah’s witnesses':
        return 48
    elif a[0] == 'Cinematographer':
        return 49
    elif a[0] == 'Circus performer':
        return 50
    elif a[0] == 'Civil servant':
        return 51
    elif a[0] == 'Clarinetist':
        return 52
    elif a[0] == 'Collector':
        return 53
    elif a[0] == 'Comedian':
        return 54
    elif a[0] == 'Companion':
        return 55
    elif a[0] == 'Competitive diver':
        return 56
    elif a[0] == 'Concentration camp guard':
        return 57
    elif a[0] == 'Concertmaster':
        return 58
    elif a[0] == 'Condottiero':
        return 59
    elif a[0] == 'Conductor':
        return 60
    elif a[0] == 'Conquistador':
        return 61
    elif a[0] == 'Conscientious objection':
        return 62
    elif a[0] == 'Consort':
        return 63
    elif a[0] == 'Contributing editor':
        return 64
    elif a[0] == 'Copperplate engraver':
        return 65
    elif a[0] == 'Coppersmith':
        return 66
    elif a[0] == 'Correspondent':
        return 67
    elif a[0] == 'Count':
        return 68
    elif a[0] == 'Courtesan':
        return 69
    elif a[0] == 'Cowboy':
        return 70
    elif a[0] == 'Coxswain':
        return 71
    elif a[0] == 'Criminal':
        return 72
    elif a[0] == 'Crusader':
        return 73
    elif a[0] == 'Curator':
        return 74
    elif a[0] == 'Customs officer':
        return 75
    elif a[0] == 'Cyberneticist':
        return 76
    elif a[0] == 'Deacon':
        return 77
    elif a[0] == 'Dentist':
        return 78
    elif a[0] == 'Dermatologist':
        return 79
    elif a[0] == 'Designer':
        return 80
    elif a[0] == 'Disc jockey':
        return 81
    elif a[0] == 'Dog breeder':
        return 82
    elif a[0] == 'Dominican friar':
        return 83
    elif a[0] == 'Drawer':
        return 84
    elif a[0] == 'Dressage rider':
        return 85
    elif a[0] == 'Drummer':
        return 86
    elif a[0] == 'Duke':
        return 87
    elif a[0] == 'Editor':
        return 88
    elif a[0] == 'Egyptologist':
        return 89
    elif a[0] == 'Electrician':
        return 90
    elif a[0] == 'Engineer':
        return 91
    elif a[0] == 'Engraver':
        return 92
    elif a[0] == 'Entrepreneur':
        return 93
    elif a[0] == 'Equestrian':
        return 94
    elif a[0] == 'Esperantist':
        return 95
    elif a[0] == 'Executioner':
        return 96
    elif a[0] == 'Explorer':
        return 97
    elif a[0] == 'Farmer':
        return 98
    elif a[0] == 'Fencer':
        return 99
    elif a[0] == 'Feudatory':
        return 100
    elif a[0] == 'Fighter pilot':
        return 101
    elif a[0] == 'Film producer':
        return 102
    elif a[0] == 'Flying ace':
        return 103
    elif a[0] == 'Formula one driver':
        return 104
    elif a[0] == 'General officer':
        return 105
    elif a[0] == 'Geographer':
        return 106
    elif a[0] == 'Inventor':
        return 107
    elif a[0] == 'Journalist':
        return 108
    elif a[0] == 'Judge':
        return 109
    elif a[0] == 'Jurist':
        return 110
    elif a[0] == 'Lady-in-waiting':
        return 111
    elif a[0] == 'Lawyer':
        return 112
    elif a[0] == 'Librarian':
        return 113
    elif a[0] == 'Marineoffizier':
        return 114
    elif a[0] == 'Merchant':
        return 115
    elif a[0] == 'Military personnel':
        return 116
    elif a[0] == 'Monarch':
        return 117
    elif a[0] == 'Monk':
        return 118
    elif a[0] == 'Motorcycle racer':
        return 119
    elif a[0] == 'Nun':
        return 120
    elif a[0] == 'Nurse':
        return 121
    elif a[0] == 'Ornithologist':
        return 122
    elif a[0] == 'Philosopher':
        return 123
    elif a[0] == 'Physician':
        return 124
    elif a[0] == 'Pianist':
        return 125
    elif a[0] == 'Police officer':
        return 126
    elif a[0] == 'Politician':
        return 127
    elif a[0] == 'Presbyter':
        return 128
    elif a[0] == 'Psychiatrist':
        return 129
    elif a[0] == 'Psychologist':
        return 130
    elif a[0] == 'Publisher':
        return 131
    elif a[0] == 'Rabbi':
        return 132
    elif a[0] == 'Racing automobile driver':
        return 133
    elif a[0] == 'Regent':
        return 134
    elif a[0] == 'Religious figure':
        return 135
    elif a[0] == 'Researcher':
        return 136
    elif a[0] == 'Resistance fighter':
        return 137
    elif a[0] == 'Rower':
        return 138
    elif a[0] == 'Ruler':
        return 139
    elif a[0] == 'Serial killer':
        return 140
    elif a[0] == 'Sovereign':
        return 141
    elif a[0] == 'Statesperson':
        return 142
    elif a[0] == 'Teacher':
        return 143
    elif a[0] == 'Torturer':
        return 144
    elif a[0] == 'Translator':
        return 145
    else:return 146

data['Id'] = data['Id'].apply(int_conversion)
data['Gender'] = data['Gender'].apply(count)
data['Occupation'] = data['Occupation'].apply(profession)
data['Manner of death'] = data['Manner of death'].apply(cause_of_death)
data.drop(['Name','Short description','Country','Associated Countries','Associated Country Coordinates (Lat/Lon)','Associated Country Life Expectancy'],axis=1,inplace=True)
data.dropna(axis=0,how='any',inplace=True)
data['Death year'] = data['Death year'].apply(int_conversion_)
data['Age of death'] = data['Age of death'].apply(int_conversion_)
x = data['Manner of death'].values.reshape(-1,1)
y = data['Age of death'].values.ravel()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)
profession_count = pd.Series(profession_count)
lr = KNeighborsClassifier()
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred_round = np.round(y_pred)
"""
plt.scatter(x_test,y_test)
plt.plot(x_test,y_pred)"""
"""data_max = profession_count.value_counts()
data_max = data_max.nlargest(10)
sns.lineplot(data_max)"""

print('mean squared error:',metrics.mean_squared_error(y_test,y_pred))
print('kappa:',metrics.cohen_kappa_score(y_test,y_pred_round))
matrix = metrics.confusion_matrix(y_test,y_pred_round)
print('balanced accuracy:',metrics.balanced_accuracy_score(y_test,y_pred_round))
print('regular accuracy:',metrics.accuracy_score(y_test,y_pred_round))
print('f1:',metrics.f1_score(y_test,y_pred_round,average='weighted'))
print('precision:',metrics.precision_score(y_test,y_pred_round,average='weighted'))
display = metrics.ConfusionMatrixDisplay(matrix)
display.plot()
plt.show()
