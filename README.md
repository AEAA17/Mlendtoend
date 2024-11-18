# Previsão de Valores de Imóveis na Califórnia 🏡
### Este projeto utiliza aprendizado de máquina para prever o valor médio de imóveis em distritos da Califórnia, utilizando um conjunto de dados com características geográficas, demográficas e econômicas. Baseado no livro de Aurélien Géron, Mãos á Obras: Aprendizado de Máquina com Scikit-Learn, Keras & TensorFlow, este projeto explora o pipeline completo de um modelo de aprendizado de máquina, desde a análise exploratória até a avaliação do modelo final.

# Objetivo

### Desenvolver modelos preditivos para estimar os preços médios de imóveis e analisar o desempenho de diferentes algoritmos de aprendizado de máquina, como regressão linear, árvores de decisão e florestas aleatórias.

# Estrutura do Projeto
### 1. Carregamento e Visualização dos Dados
### O conjunto de dados é carregado e analisado inicialmente para compreender suas características e detectar possíveis problemas, como dados faltantes ou desbalanceamentos.

### 2. Exploração dos Dados 
### Inclui análises de correlação, criação de novos atributos e visualizações para entender melhor os fatores que influenciam o preço dos imóveis.

### 3. Preparação dos Dados
### Foi construído um pipeline para pré-processamento dos dados, incluindo tratamento de valores ausentes, escalonamento de variáveis e codificação de categorias.

### 4. Treinamento e Avaliação de Modelos
### Foram avaliados os modelos de Regressão Linear, Árvore de Decisão e Floresta Aleatória, comparando métricas como RMSE e MAE.

### 5. Otimização de Hiperparâmetros
### Utilizou-se Grid Search para otimizar os parâmetros do modelo de Floresta Aleatória, alcançando uma melhora significativa no desempenho.

# Bibliotecas Utilizadas

```
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint
from scipy import stats
import joblib
```


## Carregando e verificando dados:
```
# Carregando o dataset do arquivo "housing.csv" disponibilizado no repositório de Aurélien Géron:
# Link: https://github.com/ageron/handson-ml2/blob/master/02_end_to_end_machine_learning_project.ipynb
df = pd.read_csv("housing.csv")

# Visualizando as primeiras linhas do DataFrame para inspecionar a estrutura e os dados
print(df)  # O DataFrame possui 10 colunas: 9 numéricas e 1 categórica ('ocean_proximity')

# Exibindo informações gerais do DataFrame, como tipos de dados e valores ausentes
df.info()  
# Observação: Todas as colunas estão com os tipos de dados corretos.
# A coluna 'total_bedrooms' contém valores ausentes.

# Gerando um resumo estatístico das colunas numéricas
print(df.describe())

# Exibindo a contagem de cada categoria na coluna categórica 'ocean_proximity'
print(df["ocean_proximity"].value_counts())  
# Observação: A coluna categórica possui 5 categorias distintas.

# Criando histogramas para as colunas numéricas do DataFrame
df.hist(bins=50, figsize=(20, 15))
plt.show()
```
![histogramas para as colunas numéricas do DataFrame]([images/histogramas_para_as_colunas numéricas_do_DataFrame.png](https://github.com/AEAA17/Mlendtoend/blob/76e10335552c917f5e63e2ecf586c745943b8452/images/histogramas%20para%20as%20colunas%20num%C3%A9ricas%20do%20DataFrame.png))

```
# Observações adicionais:
# - A coluna 'median_income' não representa a renda em dólares, mas foi dimensionada (valores vão de 0.5 a 15).
#   Esse problema será tratado posteriormente por meio de estratificação.
# - As colunas 'housing_median_age' e 'median_house_value' também estão dimensionadas.
# - A variável 'median_house_value', que será nossa variável-alvo, apresenta um problema: valores limite (capping).
```
##  Explorando e Visualizando os dados do df:
```
# Exibindo a distribuição geográfica dos imóveis com base na longitude e latitude.
# O parâmetro alpha reduz a opacidade dos pontos para facilitar a visualização da densidade de dados.
df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1) 
plt.show()

![distribuição geográfica dos imóveis](images/grafico.png)

# Observação: Os dados estão concentrados principalmente em áreas como São Francisco, Los Angeles, San Diego, Sacramento e Fresno.

# Comparando dados geográficos com o preço dos imóveis
# Representando a localização geográfica e adicionando variáveis contextuais:
# - O tamanho dos pontos (parâmetro s) é proporcional à população (dividida por 100 para ajuste).
# - A cor dos pontos (parâmetro c) representa o valor médio dos imóveis, com um mapa de cores contínuo (cmap).
# - O colorbar à direita indica a escala de valores.
df.plot(
    kind="scatter",
    x="longitude",
    y="latitude",
    alpha=0.4, 
    s=df["population"] / 100,
    label="Population",
    figsize=(10, 7),
    c="median_house_value",
    cmap=plt.get_cmap("jet"),
    colorbar=True,
    sharex=False
)
plt.show()

# Observação: As maiores rendas se concentram em áreas como São Francisco e Los Angeles, evidenciando uma correlação entre o valor dos imóveis e a localização.

![Comparando dados geográficos com o preço dos imóveis](images/grafico.png)
```

## Buscando correlações
```
# Calculando a matriz de correlação para as colunas numéricas
# A função `corr` calcula a correlação de Pearson entre as variáveis numéricas no DataFrame.
# Em seguida, filtramos as correlações relacionadas à variável "median_house_value" e as ordenamos em ordem decrescente.
corr_matrix = df.corr(numeric_only=True)
corr_matrix["median_house_value"].sort_values(ascending=False)
print(corr_matrix)

# Visualizando a correlação entre múltiplas variáveis
# Selecionamos atributos relevantes para análise e utilizamos a função `scatter_matrix` para criar gráficos de dispersão.
# Esses gráficos mostram a relação entre "median_house_value", "median_income", "total_rooms" e "housing_median_age".
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(df[attributes], figsize=(12, 8))
plt.show()

![correlação entre múltiplas variáveis](images/grafico.png)

# Observação: A variável mais correlacionada ao preço do imóvel ("median_house_value") é a renda média ("median_income").
# Vamos explorar essa correlação em mais detalhes com um gráfico de dispersão.

# Criando um gráfico de dispersão para analisar a relação entre "median_income" e "median_house_value".
# - O parâmetro `alpha` ajusta a transparência dos pontos para evitar sobreposição excessiva.
# - `plt.axis` define os limites para os eixos, com a faixa de valores observada no dataset.
df.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.4)
plt.axis([0, 16, 0, 550000])
plt.show()

![gráfico de dispersão para analisar a relação entre "median_income" e "median_house_value"](images/grafico.png)

# Observações:
# - O gráfico mostra uma tendência ascendente clara: quanto maior a renda média, maior o preço do imóvel.
# - Há um limite máximo de USD 500 mil, o que resulta em uma linha reta horizontal nesse valor, indicando uma restrição nos dados.
# - Linhas retas adicionais em valores como 450 mil e 350 mil sugerem outras restrições ou características no conjunto de dados.

```
##  Explorando mais correlações 
```
# Criando novas variáveis derivadas para enriquecer a análise:
# - "rooms_per_household": média de cômodos por residência.
# - "bedrooms_per_room": proporção de quartos entre o total de cômodos.
# - "population_per_household": número médio de pessoas por residência.
df["rooms_per_household"] = df["total_rooms"] / df["households"]  
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
df["population_per_household"] = df["population"] / df["households"]

# Calculando novamente a matriz de correlação para incluir as novas variáveis.
corr_matrix = df.corr(numeric_only=True)

# Ordenando as correlações com a variável-alvo ("median_house_value") em ordem decrescente.
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# Observações sobre os resultados:
# - **rooms_per_household**: possui uma correlação positiva com "median_house_value". Isso sugere que imóveis com mais cômodos por residência tendem a ser mais caros.
# - **bedrooms_per_room**: apresenta uma correlação negativa mais forte. Isso indica que, quanto maior a proporção de quartos em relação ao total de cômodos, menor é o valor do imóvel.
# - **population_per_household**: mostra uma correlação negativa fraca, sugerindo que a densidade populacional por residência tem pouca influência no preço dos imóveis.

```


## Definição das variáveis auxiliares e Construção do pipeline
```
# Selecionando as colunas relevantes para cálculos personalizados.
col_names = ["total_rooms", "total_bedrooms", "population", "households"]

# Identificando os índices das colunas no DataFrame, para referência no pipeline.
rooms_ix, bedrooms_ix, population_ix, households_ix = [
    df.columns.get_loc(c) for c in col_names
]

# Definição de uma classe personalizada para adicionar atributos combinados ao conjunto de dados.
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # Parâmetro para decidir incluir "bedrooms_per_room".
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):  # Método fit (obrigatório em classes do scikit-learn). Retorna o próprio objeto.
        return self

    def transform(self, X):  # Método transform para criar novos atributos.
        # Calcula atributos derivados: cômodos por residência e população por residência.
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        
        # Condicional para adicionar "bedrooms_per_room" apenas se especificado.
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

# Construção de um pipeline para pré-processamento de dados numéricos.
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),  # Preenche valores ausentes com a mediana.
    ('attribs_adder', CombinedAttributesAdder()),  # Adiciona atributos combinados.
    ('std_scaler', StandardScaler()),  # Padroniza os dados numéricos (média = 0, desvio padrão = 1).
])

# Criando categorias de renda para estratificação dos dados.
df["income_cat"] = pd.cut(
    df["median_income"], 
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],  # Define os intervalos das categorias.
    labels=[1, 2, 3, 4, 5]  # Nomeia as categorias de 1 a 5.
)

# Realizando uma divisão estratificada com base nas categorias de renda.
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df["income_cat"]):
    strat_train_set = df.loc[train_index]  # Conjunto de treino estratificado.
    strat_test_set = df.loc[test_index]  # Conjunto de teste estratificado.

# Separando os atributos (features) e o rótulo (label) no conjunto de treino.
housing_features = strat_train_set.drop(["median_house_value", "income_cat"], axis=1)  # Atributos.
housing_labels = strat_train_set["median_house_value"].copy()  # Rótulo.

# Reutilizando o pipeline para processar os atributos numéricos.
num_attribs = housing_features.select_dtypes(include=[np.number]).columns  # Atributos numéricos.
cat_attribs = ["ocean_proximity"]  # Atributo categórico.

# Construção de um pipeline completo para processar dados numéricos e categóricos.
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),  # Pipeline para atributos numéricos.
    ("cat", OneHotEncoder(), cat_attribs),  # Codificação one-hot para atributos categóricos.
])

# Aplicando o pipeline completo para preparar os dados.
housing_prepared = full_pipeline.fit_transform(housing_features)

# Verificando a forma do conjunto de dados processado.
print(housing_prepared.shape)
```
## Treinando e avaliando modelos
```
# Instanciando o modelo de Regressão Linear.
lin_reg = LinearRegression()

# Treinando o modelo com os dados preparados e os rótulos (valores reais).
lin_reg.fit(housing_prepared, housing_labels)
LinearRegression()

# Selecionando um pequeno conjunto de dados para teste (5 primeiras linhas do DataFrame original).
some_data = df.iloc[:5]  # Conjunto de dados de exemplo.
some_labels = housing_labels.iloc[:5]  # Rótulos correspondentes ao conjunto de exemplo.

# Aplicando o pipeline completo para preparar o conjunto de dados de exemplo.
some_data_prepared = full_pipeline.transform(some_data)

# Fazendo previsões no conjunto de dados de exemplo.
print("Predictions:", lin_reg.predict(some_data_prepared))  # Valores previstos pelo modelo.
print("Labels:", list(some_labels))  # Rótulos reais para comparação.

# Realizando previsões em todo o conjunto de treinamento.
housing_predictions = lin_reg.predict(housing_prepared)

# Calculando o erro quadrático médio (MSE) das previsões no conjunto de treinamento.
lin_mse = mean_squared_error(housing_labels, housing_predictions)

# Calculando a raiz do erro quadrático médio (RMSE) para interpretação mais intuitiva.
lin_rmse = np.sqrt(lin_mse)
print("Root Mean Squared Error (RMSE):", lin_rmse)

# Calculando o erro absoluto médio (MAE) para avaliar as diferenças absolutas entre previsões e rótulos.
lin_mae = mean_absolute_error(housing_labels, housing_predictions)
print("Mean Absolute Error (MAE):", lin_mae)

# Observação: O RMSE e o MAE são altos em comparação à faixa de valores de preço médio das casas 
# (aproximadamente 120.000 a 265.000). Isso indica que o modelo de regressão linear simples não tem
# precisão suficiente e pode não ser adequado para este problema.
```

## Modelo de Arvore de decisão
```
# Instanciando o modelo de Árvore de Decisão.
tree_reg = DecisionTreeRegressor(random_state=42)

# Treinando o modelo com os dados preparados e os rótulos (valores reais).
tree_reg.fit(housing_prepared, housing_labels)

# Fazendo previsões no conjunto de treinamento.
housing_predictions = tree_reg.predict(housing_prepared)

# Calculando o erro quadrático médio (MSE) e a raiz do erro quadrático médio (RMSE).
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print("Tree RMSE (Treinamento):", tree_rmse)

# Observação: Um RMSE de 0 ou próximo de 0 no conjunto de treinamento é extremamente raro e sugere que o modelo
# pode estar superajustado (overfitting), ou seja, aprendendo detalhes específicos dos dados de treinamento em vez
# de padrões generalizáveis. Para avaliar melhor o desempenho, usamos validação cruzada.

# Validação cruzada com 10 divisões para avaliar o desempenho do modelo de forma mais robusta.
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, 
                         scoring="neg_mean_squared_error", cv=10)

# Convertendo os scores para valores positivos e calculando o RMSE.
tree_rmse_scores = np.sqrt(-scores)

# Função para exibir métricas de desempenho.
def display_scores(scores):
    print("Scores em cada rodada de validação cruzada:", scores)
    print("Média dos scores:", scores.mean())
    print("Desvio padrão dos scores:", scores.std())

# Exibindo os resultados da validação cruzada para a Árvore de Decisão.
display_scores(tree_rmse_scores)

# Observação: Apesar de a Árvore de Decisão parecer muito boa no treinamento (RMSE próximo de 0),
# a validação cruzada mostra um desempenho consideravelmente pior, com média de RMSE de aproximadamente 71.629 
# e desvio padrão de 2.914. 

# Avaliando o modelo de Regressão Linear com validação cruzada para comparação.
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, 
                             scoring="neg_mean_squared_error", cv=10)

# Convertendo os scores para valores positivos e calculando o RMSE.
lin_rmse_scores = np.sqrt(-lin_scores)

# Exibindo os resultados da validação cruzada para o modelo de Regressão Linear.
display_scores(lin_rmse_scores)

# Observação: Comparando os resultados, o modelo de Regressão Linear apresenta um desempenho mais consistente, 
# com uma média de RMSE de aproximadamente 69.104 e desvio padrão de 2.880. Isso sugere que, embora o modelo de 
# Regressão Linear não seja perfeito, ele generaliza melhor do que a Árvore de Decisão.
```

## Modelo Floresta Randomica 
```
# Instanciando o modelo de Floresta Aleatória com 100 estimadores (árvores) e semente para reprodutibilidade.
forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)

# Treinando o modelo com os dados preparados e os rótulos (valores reais).
forest_reg = forest_reg.fit(housing_prepared, housing_labels)

# Fazendo previsões no conjunto de treinamento.
housing_predictions = forest_reg.predict(housing_prepared)

# Calculando o erro quadrático médio (MSE) e a raiz do erro quadrático médio (RMSE) no conjunto de treinamento.
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print("Forest RMSE (Treinamento):", forest_rmse)

# Observação: O RMSE baixo no treinamento indica que o modelo ajustou bem os dados de treinamento, mas isso 
# não garante que ele generaliza bem para novos dados. Para verificar isso, usamos validação cruzada.

# Validação cruzada com 10 divisões para avaliar o desempenho do modelo em diferentes subconjuntos.
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, 
                                scoring="neg_mean_squared_error", cv=10)

# Convertendo os scores negativos para positivos e calculando o RMSE para cada rodada.
forest_rmse_scores = np.sqrt(-forest_scores)

# Exibindo os resultados da validação cruzada.
display_scores(forest_rmse_scores)

# Observação: O modelo de Floresta Aleatória apresenta um desempenho superior comparado à Regressão Linear
# e à Árvore de Decisão, com uma média de RMSE de aproximadamente 50.435 e um desvio padrão de 2.203. 
# Isso sugere que o modelo tem uma boa capacidade de generalização.

# Pontos a considerar:
# - Embora o desempenho da Floresta Aleatória seja melhor, ainda há espaço para melhorias.
# - Ajustes adicionais podem incluir a otimização dos hiperparâmetros (como número de estimadores, profundidade máxima, etc.)
```

##  Aprimorando o modelo
```
# O Grid Search é um método interativo que realiza uma busca exaustiva através de todas as combinações possíveis 
# de hiperparâmetros fornecidos. Isso permite identificar a combinação que resulta no melhor desempenho do modelo.

# Definindo a grade de hiperparâmetros a ser testada. Cada dicionário representa uma configuração possível.
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},  # Testando diferentes números de árvores e características
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},  # Testando sem bootstrap e outras combinações
]

# Inicializando o GridSearchCV com a Floresta Aleatória, a grade de parâmetros e a validação cruzada de 5 divisões.
# O critério de avaliação é o erro quadrático médio negativo (neg_mean_squared_error), o que nos permite otimizar o modelo.
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)

# Treinando o GridSearch com os dados de entrada (housing_prepared) e os rótulos (housing_labels).
grid_search.fit(housing_prepared, housing_labels)

# Exibindo os melhores parâmetros encontrados pelo GridSearch.
print("Melhores parâmetros encontrados:", grid_search.best_params_)

# Usando os melhores parâmetros encontrados para treinar o modelo.
forest_reg_1 = RandomForestRegressor(max_features=8, n_estimators=30, random_state=42)
forest_reg_1.fit(housing_prepared, housing_labels)

# Fazendo previsões com o modelo otimizado.
housing_predictions = forest_reg_1.predict(housing_prepared)

# Calculando o erro quadrático médio (MSE) e a raiz do erro quadrático médio (RMSE) para as previsões.
forest_1mse = mean_squared_error(housing_labels, housing_predictions)
forest_1rmse = np.sqrt(forest_1mse)
print("RMSE após ajuste de hiperparâmetros:", forest_1rmse)

# Avaliando o desempenho do modelo ajustado com validação cruzada (10 divisões).
forest_scores1 = cross_val_score(forest_reg_1, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)

# Convertendo os scores negativos para positivos e calculando o RMSE para cada rodada da validação cruzada.
forest_rmse_scores1 = np.sqrt(-forest_scores1)

# Exibindo os resultados da validação cruzada com os scores de RMSE.
display_scores(forest_rmse_scores1)

# Observação: Comparando os resultados do modelo original (forest_reg) e do modelo ajustado (forest_reg_1),
# vemos uma melhoria significativa no desempenho. A média do RMSE diminuiu, e o desvio padrão também foi reduzido,
# o que indica uma maior precisão e maior consistência nas previsões do modelo ajustado.
```

## Avaliando o modelo com o conjunto de teste
 ```
# Inicializando e treinando o modelo com os melhores hiperparâmetros encontrados anteriormente
forest_reg_1 = RandomForestRegressor(max_features=8, n_estimators=30, random_state=42)

# Treinando o modelo final com todos os dados de treinamento (housing_prepared e housing_labels)
final_model = forest_reg_1.fit(housing_prepared, housing_labels)

# Preparando os dados de teste (X_test) removendo a coluna "median_house_value" (rótulo)
X_test = strat_test_set.drop("median_house_value", axis=1)

# Extraindo os rótulos reais (y_test), que são os valores da variável "median_house_value" do conjunto de teste
y_test = strat_test_set["median_house_value"].copy()

# Transformando os dados de teste com o pipeline já ajustado (full_pipeline) para garantir que eles tenham o mesmo formato dos dados de treino
X_test_prepared = full_pipeline.transform(X_test)

# Realizando as previsões no conjunto de teste utilizando o modelo final treinado
final_predictions = final_model.predict(X_test_prepared)

# Calculando o erro quadrático médio (MSE) para as previsões do modelo no conjunto de teste
final_mse = mean_squared_error(y_test, final_predictions)

# Calculando a raiz do erro quadrático médio (RMSE) para avaliar a magnitude do erro
final_rmse = np.sqrt(final_mse)

# Exibindo o RMSE final, que indica a precisão do modelo no conjunto de teste
print("RMSE no conjunto de teste:", final_rmse)

# Calculando um intervalo de confiança para os erros quadráticos das previsões
# Usamos a distribuição t-Student para estimar a confiabilidade dos erros
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2

# Calculando o intervalo de confiança para o erro quadrático médio das previsões
c = np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                             loc=squared_errors.mean(),
                             scale=stats.sem(squared_errors)))

# Exibindo o intervalo de confiança (c) para os erros quadráticos das previsões
print("Intervalo de confiança para os erros quadráticos:", c)

# Comentário sobre o RMSE obtido no conjunto de teste
# O valor do RMSE é 47873, o que indica o erro médio das previsões do modelo em relação aos valores reais no conjunto de teste.
```

## Salvando o modelo

```
# O 'joblib.dump' é utilizado para serializar o modelo e armazená-lo em disco, permitindo que ele seja carregado novamente em outro momento sem precisar ser re-treinado.

joblib.dump(final_model, "mlcalifornia.pkl")
```
