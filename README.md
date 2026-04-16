# ⚖️ Justiça Preditiva: Análise e Mitigação de Viés Racial no Algoritmo COMPAS

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine_Learning-black?style=for-the-badge)

## 📌 O Problema de Negócio (Contexto)
Sistemas de pontuação de risco, como o COMPAS (Correctional Offender Management Profiling for Alternative Sanctions), são amplamente utilizados no sistema de justiça criminal dos EUA para prever a reincidência de réus. No entanto, investigações como a da *ProPublica* ("Machine Bias") revelaram que o algoritmo apresenta disparidades raciais significativas, classificando réus afro-americanos com falsos positivos de alto risco em taxas muito maiores do que réus brancos. 

Este projeto explora o conflito inerente entre a **paridade preditiva** e o **equilíbrio da taxa de erro**, demonstrando que a neutralidade dos dados é um mito e que modelos de Machine Learning precisam de governança algorítmica.

## 🎯 Objetivos do Projeto
1. **Replicação:** Reproduzir a análise investigativa da ProPublica sobre as taxas de falsos positivos e negativos por grupo racial.
2. **Modelagem:** Treinar um modelo preditivo para avaliar as métricas de performance e disparidade.
3. **Mitigação e Explicabilidade:** Aplicar técnicas de IA Explicável (XAI) e equalização de chances para mitigar o viés racial nas previsões, buscando um modelo mais justo.

## 🛠️ Tecnologias e Bibliotecas Utilizadas
- **Linguagem:** Python
- **Manipulação e Análise de Dados:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Fairness & Explicabilidade (XAI):** Fairlearn, SHAP (SHapley Additive exPlanations)
- **Visualização:** Matplotlib, Seaborn

## 📊 Principais Resultados e Impacto
A aplicação de técnicas de mitigação de viés demonstrou que é possível intervir na predição bruta para garantir mais equidade sem destruir a capacidade preditiva do modelo:
- **Redução da Disparidade:** Os erros entre os grupos raciais foram **equalizados em 30%**.
- **Trade-off Performance vs. Justiça:** A mitigação das disparidades raciais e garantia de decisões mais justas custou um impacto de **apenas 3,22% na performance total** do modelo, provando a viabilidade técnica da justiça algorítmica na prática.

## 🚀 Como executar este projeto

1. Clone este repositório:
   ```
   git clone [https://github.com/otaviofabbro/analise-vies-racial-compas.git](https://github.com/otaviofabbro/analise-vies-racial-compas.git)
   
2. Instale as dependências necessárias (recomenda-se o uso de um ambiente virtual):
```
   pip install pandas numpy scikit-learn xgboost shap fairlearn matplotlib seaborn
```
3. Abra o arquivo Projeto_COMPAS.ipynb em um ambiente Jupyter (Jupyter Notebook, JupyterLab, VS Code ou Google Colab).

4. Execute as células sequencialmente. Nota: O notebook já realiza o download automático da base de dados original da ProPublica, não sendo necessário baixar arquivos CSV manualmente.
