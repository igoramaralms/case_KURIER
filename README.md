# Classificação Hierárquica de Chamados

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
`01_EDA.ipynb`
- **Diagnóstico da Qualidade:** Validação de ausência de dados nulos e estatísticas de balanceamento (todas as 5 Classes Macros possuíam contagens equivalentes).
- **Alerta de Risco (Data Leakage):** Identificação antecipada de altíssima duplicação textual no dataset bruto (o que causa viés cognitivo grave no treinamento de IA).
- **Distribuição Temporal:** Análise de sazonalidade dos tickets utilizando as flags de datas mapeadas.

### 2. Pré-processamento 
`02_preprocess.ipynb`
- Construção de pipelines comparativos testando **Regex**, **NLTK (Stemming)**, **SpaCy (Lemmatization)** e **Modelos Densos (Embeddings)**.
- **Decisão Arquitetural:** O SpaCy superou os mitigadores léxicos padrões ao generalizar tempos verbais (pt-BR) de forma inteligente, diminuindo o ruído estrutural.
- Exportação da base `dataset_processado_final.pkl` otimizando uso de memória. 

### 3. Modelagem
`03_modeling.ipynb`
- **Treinamento Frequencial:** Estratégia de negócio onde o modelo aprende com os "pesos" da frequência real mas é rigidamente provado numa base ortogonal cega (Teste 100% deduplicado e inédito).
- **Disputa de Algoritmos:** Avaliação de SVM, Logistic Regression, Naive Bayes e Random Forest. 
- **Melhor Modelo:** `LinearSVC` associado ao `TF-IDF`. Por atuar sobre jargões técnicos corporativos ("boleto", "senha", "estorno"), a lógica esparsa de vetores foi superior à redes densas para o escopo.

### 4. Aplicação e Observabilidade
`app.py`
- **Single Page Application (SPA):** Navegação em sessão fluida entre o ambiente de simulação e o Dashboard analítico.
- **Simulador Interativo:** Testes de predições na prática provando a dupla validação (O ticket só recebe *Triagem Recomendada* se confiança > 75% e > 65%).
- **Dashboard de Insights:** Módulo rodando com `Plotly` exibindo KPIs táticos (Taxa Analítica Autônoma vs Divergência), Scatter e Boxplots exibindo graficamente os insights.

---

## Tecnologias Utilizadas

- **Python 3.12**
- **Pandas / NumPy** (Engenharia de Dados)
- **Scikit-Learn** (Pipelines MLOps, TF-IDF, LogisticRegression, LinearSVC)
- **SpaCy** (Processamento de Linguagem Natural - pt_core_news_sm)
- **Streamlit** (Desenvolvimento da Interface Web Frontend SPA)
- **Plotly** (Gráficos Dinâmicos e Interativos no Dashboard Web)
- **Seaborn / Matplotlib** (Dataviz e Feature Importance Insights Estáticos)
- **Joblib** (Binários de Exportação de Modelo para Deploy Local/API)

---

## Como Reproduzir

1. Instale as dependências essenciais:
   ```bash
   pip install pandas numpy scikit-learn spacy matplotlib seaborn joblib streamlit plotly
   python -m spacy download pt_core_news_sm
   ```

2. Navegue pelos cadernos Jupyter na ordem cronológica (01 -> 02 -> 03), pois os dados processados e features limpas são herdados linearmente ao longo do estudo.

3. Execute primeiro a exportação dos arquivos `.joblib` (gerados ao final do `03_modeling`) que ficam guardados silenciosamente em `/models`.

4. Inicie o Simulador Frontend e o Analytics Dashboard Web:
   ```bash
   streamlit run app.py
   ```