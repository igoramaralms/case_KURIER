# Case XPTO Data Solutions: Classificação Hierárquica de Chamados

Este repositório contém a solução fim a fim para o desafio técnico de Cientista/Analista de Dados Júnior da **XPTO Data Solutions / KURIER**. O objetivo central deste projeto foi automatizar a triagem e categorização de chamados, tickets de suporte e feedbacks textuais gerados por clientes (Varejo Digital, Logística e Serviços Financeiros).

---

## O Desafio (Contexto de Negócio)
A triagem manual dos registros gerava:
- Alto tempo de resposta (SLA prejudicado).
- Dificuldade de priorização de incidentes críticos.
- Falta de métricas consolidadas por tipo de problema.

**Meta:** Desenvolver um modelo de Classificação Textual que organize essas informações automaticamente em duas camadas: **Classe Macro** (Roteamento de Departamento) e **Classe Detalhada** (Especialista do Problema).

---

## Arquitetura da Solução e Etapas do Projeto

O projeto foi construído sob uma ótica MLOps, priorizando não apenas a acurácia, mas a aplicabilidade real em ambiente corporativo, dividindo-se em 3 Módulos Principais (Notebooks):

### 1. Análise Exploratória (EDA) 
`01_EDA_antigravity.ipynb`
- **Diagnóstico da Qualidade:** Validação de ausência de dados nulos e estatísticas de balanceamento (todas as 5 Classes Macros possuíam contagens equivalentes).
- **Alerta de Risco (Data Leakage):** Identificação antecipada de altíssima duplicação textual no dataset bruto (o que causa viés cognitivo grave no treinamento de IA).
- **Distribuição Temporal:** Análise de sazonalidade dos tickets utilizando as flags de datas mapeadas.

### 2. Pré-processamento 
`02_preprocess_antigravity.ipynb`
- Construção de pipelines comparativos testando **Regex**, **NLTK (Stemming)**, **SpaCy (Lemmatization)** e **Modelos Densos (Embeddings)**.
- **Decisão Arquitetural:** O SpaCy superou os mitigadores léxicos padrões ao generalizar tempos verbais (pt-BR) de forma inteligente, diminuindo o ruído estrutural.
- Exportação da base `dataset_processado_final.pkl` otimizando uso de memória. 

### 3. Modelagem
`03_modeling_antigravity.ipynb`
- **Treinamento Frequencial:** Estratégia de negócio onde o modelo aprende com os "pesos" da frequência real mas é rigidamente provado numa base ortogonal cega (Teste 100% deduplicado e inédito).
- **Disputa de Algoritmos:** Avaliação de SVM, Logistic Regression, Naive Bayes e Random Forest. 
- **Melhor Modelo:** `LinearSVC` associado ao `TF-IDF`. Por atuar sobre jargões técnicos corporativos ("boleto", "senha", "estorno"), a lógica esparsa de vetores foi superior à redes densas para o escopo.

---

## Tecnologias Utilizadas

- **Python 3.12**
- **Pandas / NumPy** (Engenharia de Dados)
- **Scikit-Learn** (Pipelines MLOps, TF-IDF, LogisticRegression, LinearSVC)
- **SpaCy** (Processamento de Linguagem Natural - pt_core_news_sm)
- **Seaborn / Matplotlib** (Dataviz e Feature Importance Insights)
- **Joblib** (Binários de Exportação de Modelo para Deploy Rest/FastAPI)

---

## Como Reproduzir

1. Instale as dependências:
   ```bash
   pip install pandas numpy scikit-learn spacy matplotlib seaborn joblib
   python -m spacy download pt_core_news_sm
   ```

2. Navegue pelos cadernos jupyter na ordem cronológica (01 -> 02 -> 03), pois os dados processados e features mapeadas são herdados linearmente. 

3. Acesse a pasta `/models` para localizar os arquivos `.joblib` exportados e prontos para consumo por microsserviços do backend da XPTO Data Solutions.