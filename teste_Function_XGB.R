
library(tidymodels)

base <- rio::import("Case_Turnover_v2.csv")

base <- base %>% 
  mutate(
    desligado = as.factor(desligado)
  )
set.seed(100)
treino_teste <- base %>%  initial_split(0.75, strata = desligado)

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

# veirficar como ficou
prep(base_recipe)

source("Function_XGBoost.R")

tunagem_xgb(base_treino, base_recipe)

xgb_last_fit <- xgb_wkf %>% 
  last_fit(
    split = treino_teste,
    metrics = metric_set(roc_auc, accuracy)
  )


collect_metrics(xgb_last_fit)  

# curva roc
xgb_last_fit %>% 
  collect_predictions() %>% 
  roc_curve(desligado,`.pred_1`) %>% 
  autoplot() +
  coord_flip()

# importancia das variaveis
xgb_last_fit %>% 
  pluck(".workflow", 1) %>%   
  pull_workflow_fit() %>% 
  vip::vip(num_features = 20)

fit_adult_modelo <- fit(xgb_last_fit, base)

scores <- data_adult_val %>%
  mutate(
    more_than_50k = predict(fit_adult_modelo, new_data = data_adult_val, type = "prob")$`.pred_>50K`
  ) %>%
  select(id, more_than_50k)

