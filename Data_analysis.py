import preprocessing as pp
import csv
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import pandas as pd
import seaborn as sns
from datetime import datetime




if __name__ == '__main__':

    ## Núvem de palavras
    ##------------------

    preprocessed_emails = []
    notprocessed_emails = []
    # Lendo o arquivo csv com padrão ISO-8859-1
    with open("sms_senior.csv", "r", encoding='latin1') as f:
        cr = csv.reader(f, delimiter=",")
        for i, line in enumerate(cr):
            notprocessed_emails.append(line[0])
            email_processed = pp.preProcess(line[0]) # pré processando e-mails
            preprocessed_emails.append(" ".join(email_processed))

    # Juntando as palavras para processar pela núvem
    text = " ".join(email for email in preprocessed_emails)
    text2 = " ".join(email for email in notprocessed_emails)
    #
    # Gerando núvem de palavras
    wordcloud = WordCloud(background_color="white").generate(text)
    wordcloud2 = WordCloud(background_color="white").generate(text2)

    # Plotando núvens geradas
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(wordcloud, interpolation='bilinear')
    axarr[1].imshow(wordcloud2, interpolation='bilinear')
    axarr[0].set_title('Pré processado')
    axarr[1].set_title('Sem processamento')
    axarr[0].axis("off")
    axarr[1].axis("off")
    plt.suptitle('Nuvem de palavras')
    plt.show()

    ## Contando palavras
    ##------------------

    # Função de contar palavras
    def word_count(str):
        counts = dict()
        words = str.split()

        for word in words:
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1

        return counts

    counts = word_count(text)
    counts2 = word_count(text2)

    # Produzindo gráfico de barras para comparar dados com e sem pre processamento
    fig, axs = plt.subplots(1, 2, squeeze=False)

    count_words = pd.DataFrame(list(counts.items()))
    count_words.columns = ["Palavras", "Numero de palavras"]
    sns.barplot(x='Palavras', y='Numero de palavras', ax=axs[0][0], data=count_words.sort_values(by="Numero de palavras", ascending=False)[0:20]).set_title("Pré processado")

    count_words2 = pd.DataFrame(list(counts2.items()))
    count_words2.columns = ["Palavras", "Numero de palavras"]
    sns.barplot(x='Palavras', y='Numero de palavras', ax=axs[0][1], data=count_words2.sort_values(by="Numero de palavras", ascending=False)[0:20]).set_title("Sem processamento")

    axs[0][0].tick_params(labelrotation=45)
    axs[0][1].tick_params(labelrotation=45)

    plt.suptitle('Numero de palavras sem e com pré processamento')
    plt.tight_layout()
    plt.show()

#--------------------------------------------------------------------------------------------
    ## Analisanndo a partir da  classificação de spam e não spam (Pré processando e-mails)
    ##------------------------------------------------------------------------------------

    spam_emails = []
    notspam_emails = []
    with open("sms_senior.csv", "r", encoding='latin1') as f:
        cr = csv.reader(f, delimiter=",")
        # Separa por classificação
        for i, line in enumerate(cr):
            email_processed = pp.preProcess(line[0])
            if line[-1] == 'no':
                notspam_emails.append(" ".join(email_processed))
            else:
                spam_emails.append(" ".join(email_processed))

    # Junta palavras
    text = " ".join(email for email in spam_emails)
    text2 = " ".join(email for email in notspam_emails)
    counts = word_count(text)
    counts2 = word_count(text2)

    # Cria gráfico de barras para todas as palavras do e-mail inteiro
    fig, axs = plt.subplots(1, 2, squeeze=False)

    count_words = pd.DataFrame(list(counts.items()))
    count_words.columns = ["Palavra", "Numero de palavras"]
    sns.barplot(x='Palavra', y='Numero de palavras', ax=axs[0][0], data=count_words.sort_values(by="Numero de palavras", ascending=False)[0:20]).set_title("E-mail spam")

    count_words2 = pd.DataFrame(list(counts2.items()))
    count_words2.columns = ["Palavra", "Numero de palavras"]
    sns.barplot(x='Palavra', y='Numero de palavras', ax=axs[0][1], data=count_words2.sort_values(by="Numero de palavras", ascending=False)[0:20]).set_title("E-mail não spam")

    axs[0][0].tick_params(labelrotation=45)
    axs[0][1].tick_params(labelrotation=45)

    plt.suptitle('Numero de palavras do e-mail inteiro para cada classe (Com pré processamento)')
    plt.tight_layout()
    plt.show()

#---------------------------------------------------------------------------------------------
    ## Analisando colunas de palavras relevantes
    ##------------------------------------------

    data = pd.read_csv("sms_senior.csv", encoding='latin1')

    # Separando por classificação
    notSpam = data.loc[data['IsSpam'] == 'no']
    Spam = data.loc[data['IsSpam'] == 'yes']

    # Somando colunas para cada classe
    name_col = []
    sum_col = []
    name_col2 = []
    sum_col2 = []
    for col in notSpam.columns[1:149]:
        name_col.append(col)
    for column in notSpam.columns[1:149]:
        sum_col.append(notSpam[column].sum())

    for col in Spam.columns[1:149]:
        name_col2.append(col)
    for column in Spam.columns[1:149]:
        sum_col2.append(Spam[column].sum())

    counts = pd.DataFrame([name_col, sum_col]).T
    counts2 = pd.DataFrame([name_col2, sum_col2]).T

    # Plotando gráfico de barra para os casos
    fig, axs = plt.subplots(1, 2, squeeze=False)

    count_words = pd.DataFrame(counts)
    count_words.columns = ["Palavra", "Numero de palavras"]
    sns.barplot(x='Palavra', y='Numero de palavras', ax=axs[0][0], data=count_words.sort_values(by="Numero de palavras", ascending=False)[0:20]).set_title("E-mail spam")
    #plt.xticks(rotation=30, horizontalalignment="center")

    count_words2 = pd.DataFrame(counts2)
    count_words2.columns = ["Palavra", "Numero de palavras"]
    sns.barplot(x='Palavra', y='Numero de palavras', ax=axs[0][1], data=count_words2.sort_values(by="Numero de palavras", ascending=False)[0:20]).set_title("E-mail não spam")

    axs[0][0].tick_params(labelrotation=45)
    axs[0][1].tick_params(labelrotation=45)

    plt.suptitle('Numero de palavras relevantes para cada classe')
    plt.tight_layout()
    plt.show()
#-----------------------------------------------------------------------------------
    ## Analisando correlação entre features e target
    ##----------------------------------------------

    data = pd.read_csv("sms_senior.csv", encoding='latin1')
    X = data.iloc[:, 1:151]
    y = data.iloc[:, -1]

    # Usando métrica de Chi² para calcular a pontuação de cada palavra relevante
    bestfeatures = SelectKBest(score_func=chi2, k=10)
    fit = bestfeatures.fit(X, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)

    # Concatenando DataFrames para visualização
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Palavra', 'Pontuação']

    # Vinte palavras com maior pontuação
    print(featureScores.nlargest(20, 'Pontuação'))

#------------------------------------------------------------------------------------
    ## Classificação de e-mails por mês
    ##-------------------------------

    sns.set_style("dark")

    # Lendo os dados processando a coluna de data/hora
    custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    data = pd.read_csv('sms_senior.csv', parse_dates=['Date'], date_parser=custom_date_parser, encoding='latin1')
    data['month'] = data['Date'].dt.month
    data['day'] = data['Date'].dt.day

    # Separando classificação
    notSpam = data.loc[data['IsSpam'] == 'no']
    Spam = data.loc[data['IsSpam'] == 'yes']

    # Spam
    Janeiro_Spam = Spam.loc[Spam['month'] == 1]
    Fevereiro_Spam = Spam.loc[Spam['month'] == 2]
    Marco_Spam = Spam.loc[Spam['month'] == 3]

    # Não spam
    Janeiro_notSpam = notSpam.loc[notSpam['month'] == 1]
    Fevereiro_notSpam = notSpam.loc[notSpam['month'] == 2]
    Marco_notSpam = notSpam.loc[notSpam['month'] == 3]

    # Criando dataFrame para visualizar por mês
    plotdata = pd.DataFrame({
        "Spam": [Janeiro_Spam.shape[0], Fevereiro_Spam.shape[0], Marco_Spam.shape[0]],
        "Not spam": [Janeiro_notSpam.shape[0], Fevereiro_notSpam.shape[0], Marco_notSpam.shape[0]]
    },
        index=['Janeiro', 'Fevereiro', 'Março']
    )
    ax = plotdata.plot(kind="bar")

    # Plotando valores no topo da barra
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

    plt.xticks(rotation=0, horizontalalignment="center")
    plt.title("Quantidade de e-mails classificados por mês")
    plt.xlabel("Mês")
    plt.ylabel("Quantidade de emails")
    plt.show()
#--------------------------------------------------------------------------------------------
    ## Estatisticas com relação à coluna (Word_Count)
    ##----------------------------------------------

    # Lendo os dados processando a coluna de data/hora
    custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    data = pd.read_csv('sms_senior.csv', parse_dates=['Date'], date_parser=custom_date_parser, encoding='latin1')
    #data['year'] = data['Date'].dt.year
    data['month'] = data['Date'].dt.month
    data['day'] = data['Date'].dt.day

    # Dicionário de meses
    months = {'Janeiro  ':1, 'Fevereiro':2,'Marco    ':3}

    # Maximo
    print("----------------------------------------------------------------------------------------")
    print('----------------Estatisticas com relação à coluna (Word_Count)--------------------------')
    print("----------------------------------------------------------------------------------------")
    print("Mês    | Max | Min |       Media        |  Mediana  |     Variancia      | Desvio padrao\n")
    for month in list(months.keys()):
        dataframe = data.loc[data['month'] == months[str(month)]]
        max_: str = str(dataframe['Word_Count'].max())
        min = str(dataframe['Word_Count'].min())
        mean = str(dataframe['Word_Count'].mean())
        median = str(dataframe['Word_Count'].median())
        var = str(dataframe['Word_Count'].var())
        std = str(dataframe['Word_Count'].std())
        print(str(month) + " : " + max_ + " | " + min + " | " + mean + " | " + median + " | " + var + " | " + std)
    print("----------------------------------------------------------------------------------------")

#-----------------------------------------------------------------------------------------------
    ## Dia de cada mês com a maior (sequência) de e-mails não spams
    ##-----------------------------------------------------------

    # Lendo novamente os dados para garantir que nada foi alterado
    custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    data = pd.read_csv('sms_senior.csv', parse_dates=['Date'], date_parser=custom_date_parser, encoding='latin1')
    data['month'] = data['Date'].dt.month
    data['day'] = data['Date'].dt.day

    # Dicionário de meses presentes na base de dados
    months = {'Janeiro  ': 1, 'Fevereiro': 2, 'Marco    ': 3}

    for month in list(months.keys()):
        print(str(month) + " ---------------------------------------------------------")
        month_dataframe = data.loc[data['month'] == months[str(month)]]
        first_day = 0
        dic_counter = {}
        counter = 0
        # Iterate each day
        for index, row in month_dataframe.iterrows():
            day = row['day']
            if day == first_day:
                if row['IsSpam'] == 'no':
                    counter += 1
                else:
                    # Mantém somente a maior sequencia
                    if dic_counter[str(day)] < counter:
                        dic_counter[str(day)] = counter
                    counter = 0
            else:
                # Na mudança do dia é necessário checar a contagem
                if day > first_day != 0:
                    if dic_counter[str(first_day)] < counter:
                        dic_counter[str(first_day)] = counter
                # Inicialização
                dic_counter[str(day)] = 0
                first_day = day
                counter = 0
                if row['IsSpam'] == 'no':
                    counter += 1

        # Calculando a sequência maior dentre todos os dias do mês
        max_key = max(dic_counter, key=dic_counter.get)
        max_value = max(dic_counter.values())

        # Resultados
        print('O dia ' + str(max_key) + " obteve a maior sequencia de e-mails comuns (" + str(max_value) + ") no mês de " + str(month))
        #print(dic_counter) # Dicionário contendo o número de sequências de e-mails comuns para cada mês e dia