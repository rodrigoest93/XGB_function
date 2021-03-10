
  

#### funcao tunagem XGBoost ####

tunagem_xgb <- function(base_treino, base_recipe) {
    # validacao cruzada
    folds <- list("Numero de folds"= 0)
    folds <- svDialogs::dlg_form(folds, "Validação Cruzada - Número de partes que você vai repartir seu modelo para validação: ")$res
    folds <- unlist(folds)
    reamostragem <- vfold_cv(base_treino, v = folds)
    
#### hiperparametros #### 
 print("Agora vamos preencher os hiperparâmeotros!")   
    
    # trees
    tr <- list(
      "Coloque o 1o valor: " = 0,
      "Coloque o 2o valor: " = 0,
      "Coloque o 3o valor: " = 0
    )
    tr <- svDialogs::dlg_form(tr, "Tree - Número de Árvores (passos) - Escolha 3 valores >= 100 e <= 1500")$res
    tr <- as.vector(unlist(tr))
    
    # learn_rate
    lr <- list(
      "Coloque o 1o valor: " = 0,
      "Coloque o 2o valor: " = 0,
      "Coloque o 3o valor: " = 0
    )
    lr <- svDialogs::dlg_form(lr, "Learn Rate - Tamanho do passo - Escolha 3 valores > 0 e < 0.5")$res
    lr <- as.vector(unlist(lr))
    
    # min_n
    mn <- list(
      "Coloque o 1o valor: " = 0,
      "Coloque o 2o valor: " = 0,
      "Coloque o 3o valor: " = 0
    )
    mn <- svDialogs::dlg_form(mn, "Min n - Quantidade mínima de observações por nó - Escolha 3 valores >=5 e <= 90")$res
    mn <- as.vector(unlist(mn))
    
    # trees
    te <- list(
      "Coloque o 1o valor: " = 0,
      "Coloque o 2o valor: " = 0,
      "Coloque o 3o valor: " = 0
    )
    te <- svDialogs::dlg_form(te, "Tree Depth - Profundidade máxima da árvore - Escolha 3 valores >= 3 e <= 10")$res
    te <- as.vector(unlist(te))
    
    # loss_reduction
    ld <- list(
      "Coloque o 1o valor: " = 0,
      "Coloque o 2o valor: " = 0,
      "Coloque o 3o valor: " = 0
    )
    ld <- svDialogs::dlg_form(ld, "Loss Reduction - Parâmetro regularizador (CP) - Escolha 3 valores >= 0 <= 2")$res
    ld <- as.vector(unlist(ld))
    
    #mtry
    mt <- list(
      "Coloque o 1o valor: " = 0,
      "Coloque o 2o valor: " = 0,
      "Coloque o 3o valor: " = 0
      )
    mt <- svDialogs::dlg_form(mt, "Mtry - Quantidade de variáveis sorteadas por árvore - Escolha 3 valores >= 0.1 e <= 1")$res
    mt <- as.vector(unlist(mt))
    
    #sample_size
    ss <- list(
      "Coloque o 1o valor: " = 0,
      "Coloque o 2o valor: " = 0,
      "Coloque o 3o valor: " = 0
    )
    ss <- svDialogs::dlg_form(ss, "Sample Size - Proporção de linhas para sortear por árvore - Escolha 3 valores >= 0.5 e <= 1")$res
    ss <- as.vector(unlist(ss))
    
#### learn_rate e trees ####     
  xgb <- boost_tree(
    min_n = 5,
    mtry = 0.8,
    trees = tune(),
    tree_depth = 4,
    learn_rate = tune(),
    loss_reduction = 0,
    sample_size = 0.8 
  ) %>%
    set_mode("classification") %>% 
    set_engine("xgboost")
  
  # colocar no workflow
  xgb_wkf <- workflow() %>% 
    add_model(xgb) %>% 
    add_recipe(base_recipe)
  
  # matriz para tunagem
  matriz <- expand.grid(
    learn_rate = lr,
    trees = tr
  )
  
  # tunagem
  tunagem_1 <- xgb_wkf %>% 
    tune_grid(
      resamples = reamostragem,
      grid = matriz,
      control = control_grid(save_pred = TRUE, verbose = FALSE, allow_par = TRUE),
      metrics = metric_set(roc_auc)
    )

  # melhor hparametro
  best_1 <- tunagem_1 %>% select_best(metric = "roc_auc")

print("A 1a tunagem já foi! :)")

# --------------------------------------------------------------------------- #

#### min_n e tree_depth ####

  xgb <- boost_tree(
    min_n = tune(),
    mtry = 0.8,
    trees = best_1$trees,
    tree_depth = tune(),
    learn_rate = best_1$learn_rate,
    loss_reduction = 0,
    sample_size = 0.8 
  ) %>%
    set_mode("classification") %>% 
    set_engine("xgboost")
  
  # colocar no workflow
  xgb_wkf <- workflow() %>% 
    add_model(xgb) %>% 
    add_recipe(base_recipe)
  
  # matriz para tunagem
  matriz <- expand.grid(
    tree_depth = te,
    min_n = mn
  )
  
  # tunagem
  tunagem_2 <- xgb_wkf %>% 
    tune_grid(
      resamples = reamostragem,
      grid = matriz,
      control = control_grid(save_pred = TRUE, verbose = FALSE, allow_par = TRUE),
      metrics = metric_set(roc_auc)
    )
  
  # melhor hparametro
  best_2 <- tunagem_2 %>% select_best(metric = "roc_auc")

print("Terminou a tunagem 2! (:")

# --------------------------------------------------------------------------- #

#### loss_reduction ####

xgb <- boost_tree(
  min_n = best_2$min_n,
  mtry = 0.8,
  trees = best_1$trees,
  tree_depth = best_2$tree_depth,
  learn_rate = best_1$learn_rate,
  loss_reduction = tune(),
  sample_size = 0.8 
) %>%
  set_mode("classification") %>% 
  set_engine("xgboost")

# colocar no workflow
xgb_wkf <- workflow() %>% 
  add_model(xgb) %>% 
  add_recipe(base_recipe)

# matriz para tunagem
matriz <- expand.grid(
  loss_reduction = ld
)

# tunagem
tunagem_3 <- xgb_wkf %>% 
  tune_grid(
    resamples = reamostragem,
    grid = matriz,
    control = control_grid(save_pred = TRUE, verbose = FALSE, allow_par = TRUE),
    metrics = metric_set(roc_auc)
  )

best_3 <- tunagem_3 %>% select_best(metric = "roc_auc")

print("Foi a 3a tunagem 3! :)")

# --------------------------------------------------------------------------- #

#### mtry e sample_size ####
xgb <- boost_tree(
  min_n = best_2$min_n,
  mtry = tune(),
  trees = best_1$trees,
  tree_depth = best_2$tree_depth,
  learn_rate = best_1$learn_rate,
  loss_reduction = best_3$loss_reduction,
  sample_size = tune() 
) %>%
  set_mode("classification") %>% 
  set_engine("xgboost")

# colocar no workflow
xgb_wkf <- workflow() %>% 
  add_model(xgb) %>% 
  add_recipe(base_recipe)

# matriz para tunagem
matriz <- expand.grid(
  sample_size = ss,
  mtry = mt
)

# tunagem
tunagem_4 <- xgb_wkf %>% 
  tune_grid(
    resamples = reamostragem,
    grid = matriz,
    control = control_grid(save_pred = TRUE, verbose = FALSE, allow_par = TRUE),
    metrics = metric_set(roc_auc)
  )

# melhor hparametro
best_4 <- tunagem_4 %>% select_best(metric = "roc_auc")

print("Tunagem 4 ok! Tá quase acabando! (:")


#### tree e learn_rate ####
xgb <- boost_tree(
  min_n = best_2$min_n,
  mtry = best_4$mtry,
  trees = tune(),
  tree_depth = best_2$tree_depth,
  learn_rate = tune(),
  loss_reduction = best_3$loss_reduction,
  sample_size = best_4$sample_size 
) %>%
  set_mode("classification") %>% 
  set_engine("xgboost")

# colocar no workflow
xgb_wkf <- workflow() %>% 
  add_model(xgb) %>% 
  add_recipe(base_recipe)

# matriz para tunagem
matriz <- expand.grid(
  learn_rate = c(0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3),
  trees = c(100, 250, 500, 1000, 1500, 2000, 3000)
)

# tunagem
tunagem_5 <- xgb_wkf %>% 
  tune_grid(
    resamples = reamostragem,
    grid = matriz,
    control = control_grid(save_pred = TRUE, verbose = FALSE, allow_par = TRUE),
    metrics = metric_set(roc_auc)
  )

# melhor hparametro
best_5 <- tunagem_5 %>% select_best(metric = "roc_auc")

print("Vamos atualizar o modelo! :D")

#### atualizar wkf com modelo selecionado ####  
#atualizar workflow
  xgb_wkf <<- xgb_wkf %>% 
    finalize_workflow(best_5)
print("Seu modelo está pronto para ser testado!")
print("Utilize o objeto xgb_wk para o teste")
}
