# Tempo de Tramitação de Processos no Estado de São Paulo (2023-2024): Previsão e Fatores de Atraso

Este projeto tem como objetivo prever o tempo de tramitação de processos judiciais no Estado de São Paulo, entre dezembro/2023 e dezembro/2024, bem como identificar os principais fatores de atraso.

A pesquisa foi desenvolvida como parte de um Trabalho de Conclusão de Curso (TCC), utilizando ciência de dados aplicada ao Judiciário.

## Estrutura do Projeto

├── dados/ # Bases originais e tratadas  
│   ├── tjsp_processos_sp.csv  
│   └── tjsp_processos_tratado.csv  
│  
├── graficos/ # Gráficos gerados pelos scripts  
│   └── *.png  
│  
├── scripts/ # Scripts organizados por etapa  
│   ├── tratamento_dados.py  
│   ├── analise_exploratoria.py  
│   ├── modelagem_preditiva.py  
│   ├── modelagem_aprimorada.py  
│   ├── analise_fatores_atraso.py  
│   ├── revisao_modelagem_preditiva.py  
│   ├── revisao_modelagem_aprimorada.py  
│   └── modelagem_final_corrigida.py  # Random Forest com encoding seguro (sem data leakage)

## Etapas do Projeto

### 1. Tratamento dos Dados
Limpeza e padronização da base, remoção de nulos e transformação de variáveis temporais.

### 2. Análise Exploratória
Entendimento da distribuição dos dados e comportamento de variáveis relevantes.

### 3. Modelagem Preditiva (básica)
Criação de modelos com variáveis numéricas:
- Regressão Linear
- Árvore de Decisão
- Random Forest

### 4. Modelagem Aprimorada
Inclusão de variáveis categóricas (`Nome orgao`, `Grau`, `Municipio`) para melhorar a explicabilidade.

### 5. Análise de Fatores de Atraso
Identificação de órgãos, graus e municípios com maior tempo médio de tramitação e volume de processos.

### 6. Revisão da Modelagem
- **Revisão 1:** reaplicação dos modelos preditivos apenas com variáveis numéricas, eliminando qualquer possibilidade de vazamento de dados.
- **Revisão 2:** reintrodução das variáveis categóricas (`Nome orgao`, `Municipio`, `Grau`) com encoding supervisionado (`TargetEncoder`), aplicadas de forma segura (após divisão treino/teste).  
  - Foi adotado um filtro de representatividade mínima (≥ 3 processos) para `Nome orgao` e `Municipio`, visando reduzir ruído estatístico e overfitting institucional.

## Resultados

- O modelo final, após revisão e controle de vazamento de dados (data leakage), apresentou os seguintes resultados:
  - **MAE:** 15.83  
  - **RMSE:** 23.13  
  - **R²:** 0.01

O desempenho preditivo do modelo foi limitado. Isso se deve principalmente ao fato de que a base de dados utilizada, embora pública e padronizada, não inclui variáveis com alto poder explicativo sobre o tempo de tramitação de processos.
Fatores críticos que impactam diretamente a duração dos processos — como tipo de ação, número de partes envolvidas, movimentações processuais específicas, perfil dos juízes, acúmulo de trabalho real por servidor, entre outros — não estavam disponíveis na base estruturada do CNJ (DataJud), o que restringe a capacidade de modelagem.
Além disso, a base representa dados agregados por unidade judicial, e não por processo individual, o que impede o uso de abordagens mais finas e personalizadas de previsão.
Ainda assim, o modelo permitiu identificar padrões de correlação e fatores associados a maior lentidão, como a jurisdição, o município de atuação, e o volume de processos suspensos (`Sus_Dec_2024`), contribuindo para diagnósticos institucionais.
As variáveis com maior importância no modelo foram:  Nome orgao, Municipio, Sus_Dec_2024, TPCPL_Dec_2024_num, entre outras.

### Interpretação das Correlações

As análises de correlação realizadas no projeto permitiram identificar associações relevantes entre variáveis institucionais e o tempo de tramitação dos processos.
Por exemplo, observou-se que o tempo para conclusão (`TPCPL_Dec_2024_num`) apresenta correlação positiva com o tempo até sentença, sugerindo que unidades mais lentas na fase final também enfrentam lentidão na fase inicial.
Da mesma forma, variáveis como número de suspensões (`Sus_Dec_2024`) e o município de atuação mostraram associação com o desempenho processual. No entanto, é importante destacar que correlação não implica causalidade: essas relações indicam padrões estatísticos, mas não permitem concluir que uma variável causa diretamente a outra. Ainda assim, tais correlações são valiosas para orientar investigações futuras, priorizar auditorias internas e fundamentar estratégias de gestão judicial baseadas em evidências.

## Fonte dos Dados

Os dados utilizados neste projeto foram obtidos por meio do painel público do Conselho Nacional de Justiça (CNJ), disponível no DataJud:

**Painel de Estatísticas do CNJ (DataJud)**  
https://www.cnj.jus.br/painel/datajud/

- O arquivo `tjsp_processos_sp.csv`, presente na pasta `/dados`, contém a base original bruta referente ao Tribunal de Justiça de São Paulo (TJSP), baixada no mês de abril de 2025.
- O arquivo `tjsp_processos_tratado.csv` é a versão tratada e padronizada, usada na análise e modelagem preditiva.
- O tratamento dos dados foi realizado no script `tratamento_dados.py`.

> A base inclui variáveis como volume de processos, congestionamento, tempo médio de tramitação e indicadores de produtividade por órgão judicial, município e grau de jurisdição.

## Softwares, Ambiente e Bibliotecas

- Python 3.12  
- Pandas  
- Scikit-learn  
- Seaborn  
- Matplotlib  
- Category Encoders  
- Anaconda com Spyder

## Conclusão

Com base nos dados do Tribunal de Justiça de São Paulo e no uso de técnicas de aprendizado de máquina, foi possível desenvolver e validar um modelo preditivo estatisticamente correto, embora com desempenho modesto (R² = 0.01 após ajuste). Isso evidencia a complexidade da previsão com dados públicos disponíveis, mas ainda assim permite extrair padrões relevantes e direcionar futuras investigações.

A inclusão controlada de variáveis categóricas, como nome do órgão, município e grau de jurisdição, foi decisiva para ampliar a capacidade explicativa do modelo, permitindo identificar gargalos e padrões de atraso associados a características específicas dos tribunais e localidades.

Durante o processo, foi identificado que muitos órgãos judiciais possuíam amostragens muito pequenas (menos de 3 ou 5 processos), o que comprometia a confiabilidade de suas estimativas. Para mitigar esse problema, foi adotado um filtro mínimo de frequência para inclusão dessas categorias na modelagem, o que pode ter deixado de fora realidades importantes, porém menos representadas.

Além disso, a base utilizada, apesar de pública e confiável, é uma fotografia de indicadores consolidados. Isso significa que não foi possível identificar a fundo os motivos específicos dos atrasos, apenas os fatores correlacionados. A ausência de dados operacionais detalhados (como tipo de demanda, perfil dos processos ou movimentações específicas) limita a interpretação causal.

Mesmo com essas limitações, o projeto entrega um modelo robusto em termos metodológicos e interpretações valiosas para auxiliar decisões estratégicas no Poder Judiciário. Ele também fornece uma base sólida para estudos futuros que desejem incorporar variáveis adicionais, análises qualitativas ou cruzamentos com dados operacionais para aprofundar os diagnósticos.

## Referências

- Conselho Nacional de Justiça (CNJ). Justiça em Números 2023. Brasília: CNJ, 2023. Disponível em: https://www.cnj.jus.br  
- Taylor, M. M.; Da Ros, L. (2015). O custo da justiça no Brasil: uma análise comparada dos gastos do judiciário. *Revista de Administração Pública*, 49(3), 677–697. https://doi.org/10.1590/0034-76121354  
- Hastie, T.; Tibshirani, R.; Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. 2ª ed. New York: Springer.  
- Santos, F. (2021). *Análise de Dados: Manual do Profissional de Dados*. São Paulo: Ciência Moderna.  
- Faero, L. P.; Belfiore, P. (2017). *Manual de Análise de Dados*. São Paulo: Blucher.  
- Porto, A. C. (2019). O impacto da transformação digital no judiciário brasileiro: uma análise da adoção de tecnologias no sistema judicial. *Revista Eletrônica do TRT da 9ª Região*, v. 9, n. 96.  
- Ribeiro, D. (2024). *Gestão de Grandes Escritórios de Advocacia: Oportunidades com Inteligência de Dados*. São Paulo: Fórum Jurídico de Gestão.

## Autoria

Desenvolvido por **Luciana Zedan de Carvalho** – [@lzedcar](https://github.com/lzedcar)  
Curso de Data Science e Analytics - MBA USP/Esalq 2025
