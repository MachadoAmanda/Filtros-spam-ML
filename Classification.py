import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------
# Lendo banco de dados
data = pd.read_csv('sms_senior.csv', encoding='latin1')

## Plot target in barplot
##------------------------

# Mapeando classes para visualização
data['IsSpam']=data['IsSpam'].map({'no':'Não spam','yes':'Spam'})
ax = sns.countplot(data['IsSpam'],label="Sum")
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
plt.xticks(rotation=0, horizontalalignment="center")
plt.title("Ocorrências dos emails")
plt.xlabel("Classificação")
plt.ylabel("Quantidade")
plt.show()


# Mapeando classes para binário, onde 0 correnponde a emails comuns e 1 a spams
data['IsSpam']=data['IsSpam'].map({'Não spam':0,'Spam':1})
print(data.head())

# Checando por valores faltantes na base de dados
print('############################################################################')
for column in list(data.columns):
    no_missing = data[column].isnull().sum()
    if no_missing > 0:
        print(column + ' : ' + str(no_missing))
    else:
        print(column + ' : sem valores faltantes')
print('############################################################################')


## Calculando correlação entre as 20 palavras relevantes com mais pontuação
##-------------------------------------------------------------------------

# Separando colunas da base de dados para analisar
data_corr = data.loc[:, ['call','free','Common_Word_Count','txt','claim','mobile','prize','stop','won','nokia','text','reply','urgent','service','guaranteed','cash','win','now','contact','box']]

# Calculando correlação
correlation = data_corr.corr()
mask = np.zeros_like(correlation, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Gerando figura para análise de correlação
f, ax = plt.subplots(figsize=(20, 20))

cmap = sns.diverging_palette(180, 20, as_cmap=True)
sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=1, vmin =-1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
plt.show()

# A coluna !!prize!! apresentou correlação maior ou igual a 0.6 com a coluna !!won!! uma das duas colunas deve ser retirada.
# Por ter mais correlações próximas de 0.6 com outras colunas, a coluna !!prize!! foi escolhida para remover das features.

#----------------------------------------------------------------------------
## Machine learning models
##------------------------

# Study cases
# Case 1 ------------ All features
X_c1 = data[data.columns[1:151]]
y_c1 = data['IsSpam']
X_train_c1, X_test_c1, y_train_c1, y_test_c1 = train_test_split(X_c1, y_c1, test_size=0.30)

# Case 2 ------------ Just 20 features with best score using chi² as measure
data_c2= data.loc[:, ['call','free','Common_Word_Count','txt','claim','mobile','stop','won','nokia','text','reply','urgent','service','guaranteed','cash','win','now','contact','box']]
X_c2 = data_c2
y_c2 = data['IsSpam']
X_train_c2, X_test_c2, y_train_c2, y_test_c2 = train_test_split(X_c2, y_c2, test_size=0.30)

dic_plot_measures = {'Case 1': {'ac':[], 'f1': [], 'auc':[]}, 'Case 2':{'ac':[], 'f1': [], 'auc':[]}}
# Confusion Matrix
def predict(X_test, y_test, case, classifier):
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    ac = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    dic_plot_measures[case]['ac'].append(ac)
    dic_plot_measures[case]['f1'].append(f1)
    dic_plot_measures[case]['auc'].append(auc)
    print(case + ' : accuracy ' + str(ac))
    print(case + ' : f1 score ' + str(f1))
    print(case + ' : ROC AUC ' + str(auc))

# ----------------------------------------------------------------------------------------------
# Random forest --- Case 1
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train_c1, y_train_c1)

# Confusion matrix
predict(X_test_c1, y_test_c1, 'Case 1', classifier)

# Random forest ---- Case 2
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train_c2, y_train_c2)

# Confusion matrix
predict(X_test_c2, y_test_c2, 'Case 2', classifier)
#-----------------------------------------------------------------------------------------------
# Suport Vector Machine (SVM) --- Case 1
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train_c1, y_train_c1)

# Confusion matrix
predict(X_test_c1, y_test_c1, 'Case 1', classifier)

# Suport Vector Machine (SVM) ---- Case 2
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train_c2, y_train_c2)

# Confusion matrix
predict(X_test_c2, y_test_c2, 'Case 2', classifier)
#----------------------------------------------------------------------------
# Plotting comparison
plotdata_c1 = pd.DataFrame(dic_plot_measures['Case 1'],
        index=['RF', 'SVM']
    )
plotdata_c2 = pd.DataFrame(dic_plot_measures['Case 2'],
        index=['RF', 'SVM']
    )

plotdata_c1.plot(kind="bar", grid=True)
plt.xticks(rotation=0, horizontalalignment="center")
plt.ylim([0.825, 0.97])
plt.suptitle("Métricas para RF e SVM para caso1")
plt.xlabel("Caso")
plt.ylabel("Indicador")
plt.show()

plotdata_c2.plot(kind="bar", grid=True)
plt.xticks(rotation=0, horizontalalignment="center")
plt.ylim([0.75, 0.97])
plt.suptitle("Métricas para RF e SVM para caso 2")
plt.xlabel("Caso")
plt.ylabel("Indicador")
plt.show()






















#-----------------------------------------------------------------------------------------------
#                                     EXTRA METHODS
#-----------------------------------------------------------------------------------------------
# # Elastic Net --- Case 1
# classifier = ElasticNet(alpha=1.0, l1_ratio=0.5)
# classifier.fit(X_train_c1, y_train_c1)
#
# # Confusion matrix
# y_pred_c1 = classifier.predict(X_test_c1)
# cm = confusion_matrix(y_test_c1, y_pred_c1)
# print(cm)
# accuracy_score(y_test_c1, y_pred_c1)
# print(str(accuracy_score(y_test_c1, y_pred_c1)))
#
# # Elastic Net---- Case 2
# classifier = ElasticNet(alpha=1.0, l1_ratio=0.5)
# classifier.fit(X_train_c2, y_train_c2)
#
# # Confusion matrix
# y_pred_c2 = classifier.predict(X_test_c2)
# cm = confusion_matrix(y_test_c2, y_pred_c2)
# print(cm)
# accuracy_score(y_test_c2, y_pred_c2)
# print(str(accuracy_score(y_test_c2, y_pred_c2)))
#------------------------------------------------------------------------------------------------
# # Nearest Centroids (NC) --- Case 1
# classifier = NearestCentroid()
# classifier.fit(X_train_c1, y_train_c1)
#
# # Confusion matrix
# y_pred_c1 = classifier.predict(X_test_c1)
# cm = confusion_matrix(y_test_c1, y_pred_c1)
# print(cm)
# accuracy_score(y_test_c1, y_pred_c1)
# print(str(accuracy_score(y_test_c1, y_pred_c1)))
#
# # Nearest Centroids (NC) --- Case 2
# classifier = NearestCentroid()
# classifier.fit(X_train_c2, y_train_c2)
#
# # Confusion matrix
# y_pred_c2 = classifier.predict(X_test_c2)
# cm = confusion_matrix(y_test_c2, y_pred_c2)
# print(cm)
# accuracy_score(y_test_c2, y_pred_c2)
# print(str(accuracy_score(y_test_c2, y_pred_c2)))

