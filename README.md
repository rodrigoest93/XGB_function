---
title: "Função para Tunar modelo de XGBoost"
date: "10/03/2021"
author: "Rodrigo Almeida Figueira"
output: 
  github_document:
    toc: true
---


```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
library(tidymodels)
```

## Motivação

Este material foi feito inspirado pelo curso de R para Machine Learning, ministrado pela [Curso-R](https://curso-r.com/), em que mostra uma heurística sobre a tunagem do Modelo de XGBoost. O material está disponível para qualquer pessoa [neste link](https://curso-r.github.io/202006-intro-ml/), baixando o arquivo `exemplos/12-xgboost.Rmd`.

## O que é XGBoost?

O modelo de [XGBoost](https://curso-r.github.io/main-intro-ml/slides/03-modelos-de-arvores.html#31) é uma extensão do Gradiente Boost, que é um modelo baseado em árvores. Se você deseja entender melhor o que é isso, sugiro [este material](https://brunaw.com/slides/satrday-sp/tidyverse-para-AM.html#1) e o repositória que está exposto nele. 

Bom, o XGBoost conta com sete hiperparâmetros para encontrarmos a melhor combinação de um modelo. De que forma fazemos isso? Através da tunagem dos hiperparâmeros. Porém, quanto maior o número destes, mais demorado será a execução do modelo e maior capacidade computacional você vai precisar. 

Se tivéssemos uma máquina com memória infinita, poderíamos testar infinitas possibilidades de hiperparâmetros para encontrarmos os melhores possíveis. No entando, sabemos que: uma máquina infinita custa infinitos dinheiros (R$) e ainda sim não conseguiria testar infinitas combinações e; se rodarmos em nuvem infinitamente isso, pode nos gerar também uma quantia absurda.

Com isso, foram criadas diversar maneiras de executar essa tunagem, e aqui, vou expor, em forma de função, a maneira que aprendi no curso que fiz, como falei anteriormente.

## Porque uma função para tunar XGBoost?

Muitas vezes, no nosso dia a dia, estamos bem atarefados com nossos afazeres. Na minha experiência pessoal, sempre gosto de entender melhor qualquer situação problema que vem até mim, cada particularidade e cada circunstância. No entando, isso nem sempre é possível. Com isso, algo que me facilite a execução deste processo de tunagem foi bem pertinente.

Outro motivo que me fez montar essa função, foi o fato de que sempre é bom ter um "chute inicial" para os nossos trabalhos. Como já dizia o filósofo: *Nada se cria, tudo se copia*. Resumindo, com esta função tenho bons hiperparâmetros iniciais para que, caso eu queira, posteriormente, conseguir ajustar meu modelo.

## Modelagem
Sem mais delongas, agora vamos a um teste de modelagem com a função de tunagem do XGBoost. Caso queira ter acesso a ela, só baixar o script e utilizar a função `source("nome do arquivo")` no seu script de modelagem.


### Base trabalhada

Como sou da área de People Analytics, trabalharei com uma base que utilizo sempre de funcionários de uma empresa, onde contém informações de colaboradores desligados e ativos da mesma. Como este não é o foco deste material, não vamos nos aprofundar nas questões fora da função de tunagem.

Abaixo temos as variáveis trabalhadas e suas informações de estrutura:

```{r, message=FALSE, warning=FALSE, echo=TRUE}
base <- rio::import("../Case_Turnover_v2.csv")

base %>% 
  select_if(is.numeric) %>% 
    skimr::skim()

base %>% 
  select_if(is.character) %>% 
    skimr::skim()
```

Em um processo normal de estudo, jamais passaria para a próxima etapa sem antes fazer uma boa análise exploratória, a fim de entender as variáveis que estão na base e trabalhar possíveis tratamentos das mesmas.


### Pré processamento

Nesta etapa faremos a transformação da variável independente, dividiremos a base de treino e teste e trataremos possíveis problemas que as variáveis dependentes tenham. 

```{r, message=FALSE, warning=FALSE, echo=TRUE}
base <- base %>% 
  mutate(
    desligado = as.factor(desligado)
  )

set.seed(100)
treino_teste <- base %>%  initial_split(0.05, strata = desligado)

base_treino <- training(treino_teste)
base_teste <- testing(treino_teste)

# Data prep
base_recipe <- recipe(
  desligado ~., data = base_treino
) %>% 
  step_zv(all_predictors()) %>%
  step_modeimpute(all_nominal(), -all_outcomes()) %>%
  step_medianimpute(all_numeric()) %>%
  step_novel(all_nominal(), -all_outcomes()) %>%
  step_dummy(all_nominal(), -all_outcomes())
```

No final do processo nossa base ficará assim:

```{r, message=FALSE, warning=FALSE, echo=TRUE}
# veirficar como ficou
juice(prep(base_recipe)) %>% head() %>% knitr::kable()

```

### Carregamento da função

Agora vamos carregar a função que está em um script em meu diretório.

```{r, message=FALSE, warning=FALSE, echo=TRUE}
source("../Function_XGBoost_output.R")
```

Após isso, basta utilizarmos a função {tunagem_xgb()}, que tem os argumentos `base_treino` e `base_recipe` (esta é a base pré-processada). Ela pedirá o número de folds que você deseja repartir sua base para validação cruzada.

Obs: para este exemplo já deixei selecionada o número de folds fixo em 5, mas no script é o usuário quem define.


```{r, message=FALSE, echo=TRUE}
tunagem_xgb(base_treino, base_recipe)
```

Como podem ver, esta função vai retornar os valores dos melhores hiperparâmetros e salvar o objeto `xgb_wkf`, que nada mais é que o workflow (objeto que carrega todos os elementos do modelo) utilizado na modelagem. Precisaremos deste objeto para tetar nosso modelo.

### Teste do modelo

Após utilização da função, vamos testar nosso modelo com o objeto `xgb_wkf`, como disse anteriormente:

```{r, message=FALSE, warning=FALSE, echo=TRUE}
xgb_last_fit <- xgb_wkf %>% 
  last_fit(
    split = treino_teste,
    metrics = metric_set(roc_auc, accuracy)
  )
```

### Resultados do modelo

Agora saberemos as métricas do nosso modelo e a verificação da curva ROC. Aqui utilizei acurácia e auroc, mas claro que isso fica a critério de cada um:


```{r, message=FALSE, warning=FALSE, echo=TRUE}
collect_metrics(xgb_last_fit) %>% knitr::kable()

# curva roc
xgb_last_fit %>% 
  collect_predictions() %>% 
  roc_curve(desligado,`.pred_1`) %>% 
  autoplot() +
  coord_flip()
```

Também conseguimos verificar a importância das variáveis do modelo:

```{r, message=FALSE, warning=FALSE, echo=TRUE}
xgb_last_fit %>% 
  pluck(".workflow", 1) %>%   
  pull_workflow_fit() %>% 
  vip::vip(num_features = 20)
```

Após este processo todo, você pode salvar este modelo e utilizar como quiser!

## Considerações

Este material serve para quem deseja aplicar o processo de tunagem dos hiperparâmetros do modelo de XGBoost de maneira mais simples e quem sabe conseguir um bom chute inicial destes argumentos. Não recomendo aplicar a função diretamente a uma base, sem a análise exploratória e sem conhecimento prévio dos hiperparâmetros. No mais, aceito contribuições, sugestões e críticas! Pode acessar [meu Github](https://github.com/rodrigoest93) ou [meu Linkedin](https://www.linkedin.com/in/rodrigoalmeidafigueira/).



