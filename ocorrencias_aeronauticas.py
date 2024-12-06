
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

st.title("Ocorrências Aeronáuticas")

# Carregar os dados
ocorrencias = pd.read_csv("https://raw.githubusercontent.com/Fernand-Rosa/AcidentesAereosBD/main/ocorrencia.csv", sep=';', encoding='latin-1')
fator_contribuinte = pd.read_csv("https://raw.githubusercontent.com/Fernand-Rosa/AcidentesAereosBD/main/fator_contribuinte.csv", sep=';', encoding='latin-1')
aeronave = pd.read_csv("https://raw.githubusercontent.com/Fernand-Rosa/AcidentesAereosBD/main/aeronave.csv", sep=';', encoding='latin-1')
recomendacao = pd.read_csv("https://raw.githubusercontent.com/Fernand-Rosa/AcidentesAereosBD/main/recomendacao.csv", sep=';', encoding='latin-1')
ocorrencia_tipo = pd.read_csv("https://raw.githubusercontent.com/Fernand-Rosa/AcidentesAereosBD/main/ocorrencia_tipo.csv", sep=';', encoding='latin-1')

# Dicionário para mapear os nomes das bases de dados aos DataFrames
bases_de_dados = {
    'Ocorrências': ocorrencias,
    'Fator Contribuinte': fator_contribuinte,
    'Aeronave': aeronave,
    'Recomendação': recomendacao,
    'Ocorrência Tipo': ocorrencia_tipo
}

# Seleção das bases de dados
base_de_dados_tipo = st.multiselect("Selecione a base de dados a ser visualizada", list(bases_de_dados.keys()))

# Exibir os dados selecionados
for base in base_de_dados_tipo:
    st.subheader(base)
    st.dataframe(bases_de_dados[base])

# Análise dos tipos de ocorrências
if st.button("Analisar Tipos de Ocorrências"):
    # Merge dos dados
    merged_df = pd.merge(ocorrencias, ocorrencia_tipo, left_on='codigo_ocorrencia1', right_on='codigo_ocorrencia1', how='left')
    # Contagem dos tipos de ocorrências
    tipos_ocorrencias_counts = merged_df['ocorrencia_tipo'].value_counts()

    # Exibir gráfico dos tipos de ocorrências
    st.subheader("Distribuição dos Tipos de Ocorrências")
    plt.figure(figsize=(10, 6))
    tipos_ocorrencias_counts.plot(kind='bar', color='lightcoral')
    plt.title('Distribuição dos Tipos de Ocorrências')
    plt.xlabel('Tipo de Ocorrência')
    plt.ylabel('Número de Ocorrências')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(plt)

# Análise de correlação entre variáveis contínuas
if st.button("Analisar somatório de tipos de ocorrências"):
    # Exibir a contagem de tipos de ocorrências
    st.subheader("Contagem de Tipos de Ocorrências")
    ocorrencia_tipo_counts = ocorrencia_tipo['ocorrencia_tipo'].value_counts()
    st.dataframe(ocorrencia_tipo_counts)

# Análise de Fator Contribuinte
if st.button("Analisar Fator Contribuinte"):
    fator_contribuinte_counts = fator_contribuinte['fator_nome'].value_counts()
    st.subheader("Distribuição dos Fatores Contribuintes")
    plt.figure(figsize=(10, 6))
    fator_contribuinte_counts.plot(kind='bar', color='skyblue')
    plt.title('Distribuição dos Fatores Contribuintes')
    plt.xlabel('Fator Contribuinte')
    plt.ylabel('Número de Ocorrências')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(plt)

# Implementação do Random Forest para prever o nível de dano da aeronave
if st.button("Prever Nível de Dano da Aeronave"):
    st.subheader("Previsão do Nível de Dano da Aeronave")
    
    # Selecionar as colunas relevantes para a previsão
    features = ['aeronave_tipo_veiculo', 'aeronave_fabricante', 'aeronave_modelo', 'aeronave_ano_fabricacao', 'aeronave_motor_tipo', 'aeronave_motor_quantidade', 'aeronave_assentos', 'aeronave_pais_fabricante', 'aeronave_pais_registro']
    target = 'aeronave_nivel_dano'

    # Preprocessamento dos dados
    aeronave_data = aeronave[features + [target]].dropna()
    X = pd.get_dummies(aeronave_data[features])
    y = aeronave_data[target]

    # Dividir os dados em conjunto de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Treinar o modelo de Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Fazer previsões no conjunto de teste
    y_pred = model.predict(X_test)

    # Exibir os resultados
    st.text("Relatório de Classificação:")
    st.text(classification_report(y_test, y_pred))

    st.text("Matriz de Confusão:")
    st.dataframe(confusion_matrix(y_test, y_pred))
