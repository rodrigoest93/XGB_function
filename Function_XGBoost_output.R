
  
#### funcao tunagem XGBoost ####
tunagem_xgb <- function(base_treino, base_recipe) {
    # validacao cruzada
    folds <- list("Numero de folds"= 0)
    folds <- svDialogs::dlg_form(folds, "Validação Cruzada - Número de partes que você vai repartir seu modelo para validação: ")$res
    folds <- unlist(folds)
    reamostragem <- vfold_cv(base_treino, v = folds)
    
#### hiperparametros #### 
 print("Agora vamos rodar os hiperparâmetros!")   
    
    # trees
    tr <- c(100, 500, 1000, 1500)
    
    # learn_rate
    lr <- c(0.05, 0.1, 0.2, 0.3)
    
    # min_n
    mn <- c(5, 15, 30, 60, 90)
    
    # trees
    te <- c(3, 4, 6, 8, 10)
    
    # loss_reduction
    ld <- c(0, 0.05, 0.1, 0.15, 0.25, 0.35, 0.45, 0.5, 1, 2)
    
    #mtry
    mt <- seq(0.1, 1.0, length.out = 10)
    
    #sample_size
    ss <- seq(0.5, 1.0, length.out = 10)
  
#### learn_rate e trees ####     
  xgb <- parsnip::boost_tree(
    min_n = 5,
    mtry = 0.8,
    trees = tune(),
    tree_depth = 4,
    learn_rate = tune(),
    loss_reduction = 0,
    sample_size = 0.8 
  ) %>%
    parsnip::set_mode("classification") %>% 
    parsnip::set_engine("xgboost")

  # colocar no workflow
  xgb_wkf <- workflows::workflow() %>% 
    workflows::add_model(xgb) %>% 
    workflows::add_recipe(base_recipe)
  
  # matriz para tunagem
  matriz <- tidyr::expand_grid(
    learn_rate = lr,
    trees = tr
  )
  
  # tunagem
  tunagem_1 <- xgb_wkf %>% 
    tune::tune_grid(
      resamples = reamostragem,
      grid = matriz,
      control = control_grid(save_pred = TRUE, verbose = FALSE, allow_par = TRUE),
      metrics = metric_set(roc_auc)
    )

  # melhor hparametro
  best_1 <- tunagem_1 %>% tune::select_best(metric = "roc_auc")

print("Melhores hiperparâmetros: ")
print(paste("Lean_rate = ", best_1$learn_rate))
print(paste("Trees = ", best_1$trees))

# --------------------------------------------------------------------------- #

#### min_n e tree_depth ####

  xgb <- parsnip::boost_tree(
    min_n = tune(),
    mtry = 0.8,
    trees = best_1$trees,
    tree_depth = tune(),
    learn_rate = best_1$learn_rate,
    loss_reduction = 0,
    sample_size = 0.8 
  ) %>%
    parsnip::set_mode("classification") %>% 
    parsnip::set_engine("xgboost")
    
  # colocar no workflow
  xgb_wkf <- workflows::workflow() %>% 
    workflows::add_model(xgb) %>% 
    workflows::add_recipe(base_recipe)
  
  # matriz para tunagem
  matriz <- tidyr::expand_grid(
    tree_depth = te,
    min_n = mn
  )
  
  # tunagem
  tunagem_2 <- xgb_wkf %>% 
    tune::tune_grid(
      resamples = reamostragem,
      grid = matriz,
      control = control_grid(save_pred = TRUE, verbose = FALSE, allow_par = TRUE),
      metrics = metric_set(roc_auc)
    )
  
  # melhor hparametro
  best_2 <- tunagem_2 %>% tune::select_best(metric = "roc_auc")

print(paste("Tree_depth = ", best_2$tree_depth))
print(paste("Min_n = ", best_2$min_n))

# --------------------------------------------------------------------------- #

#### loss_reduction ####

xgb <- parsnip::boost_tree(
  min_n = best_2$min_n,
  mtry = 0.8,
  trees = best_1$trees,
  tree_depth = best_2$tree_depth,
  learn_rate = best_1$learn_rate,
  loss_reduction = tune(),
  sample_size = 0.8 
) %>%
  parsnip::set_mode("classification") %>% 
  parsnip::set_engine("xgboost")

# colocar no workflow
xgb_wkf <- workflows::workflow() %>% 
  workflows::add_model(xgb) %>% 
  workflows::add_recipe(base_recipe)

# matriz para tunagem
matriz <- tidyr::expand_grid(
  loss_reduction = ld
)

# tunagem
tunagem_3 <- xgb_wkf %>% 
  tune::tune_grid(
    resamples = reamostragem,
    grid = matriz,
    control = control_grid(save_pred = TRUE, verbose = FALSE, allow_par = TRUE),
    metrics = metric_set(roc_auc)
  )

best_3 <- tunagem_3 %>% tune::select_best(metric = "roc_auc")

print(paste("Loss_reduction = ", best_3$loss_reduction))

# --------------------------------------------------------------------------- #

#### mtry e sample_size ####
xgb <- parsnip::boost_tree(
  min_n = best_2$min_n,
  mtry = tune(),
  trees = best_1$trees,
  tree_depth = best_2$tree_depth,
  learn_rate = best_1$learn_rate,
  loss_reduction = best_3$loss_reduction,
  sample_size = tune() 
) %>%
  parsnip::set_mode("classification") %>% 
  parsnip::set_engine("xgboost")

# colocar no workflow
xgb_wkf <- workflows::workflow() %>% 
  workflows::add_model(xgb) %>% 
  workflows::add_recipe(base_recipe)

# matriz para tunagem
matriz <- tidyr::expand_grid(
  sample_size = ss,
  mtry = mt
)

# tunagem
tunagem_4 <- xgb_wkf %>% 
  tune::tune_grid(
    resamples = reamostragem,
    grid = matriz,
    control = control_grid(save_pred = TRUE, verbose = FALSE, allow_par = TRUE),
    metrics = metric_set(roc_auc)
  )

# melhor hparametro
best_4 <- tunagem_4 %>% tune::select_best(metric = "roc_auc")

print(paste("Sample_size = ", best_4$sample_size))
print(paste("Mtry = ", best_4$mtry))



#### tree e learn_rate ####
xgb <- parsnip::boost_tree(
  min_n = best_2$min_n,
  mtry = best_4$mtry,
  trees = tune(),
  tree_depth = best_2$tree_depth,
  learn_rate = tune(),
  loss_reduction = best_3$loss_reduction,
  sample_size = best_4$sample_size 
) %>%
  parsnip::set_mode("classification") %>% 
  parsnip::set_engine("xgboost")

# colocar no workflow
xgb_wkf <- workflows::workflow() %>% 
  workflows::add_model(xgb) %>% 
  workflows::add_recipe(base_recipe)

# matriz para tunagem
matriz <- tidyr::expand_grid(
  learn_rate = c(0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3),
  trees = c(100, 250, 500, 1000, 1500, 2000, 3000)
)

# tunagem
tunagem_5 <- xgb_wkf %>% 
  tune::tune_grid(
    resamples = reamostragem,
    grid = matriz,
    control = control_grid(save_pred = TRUE, verbose = FALSE, allow_par = TRUE),
    metrics = metric_set(roc_auc)
  )

# melhor hparametro
best_5 <- tunagem_5 %>% tune::select_best(metric = "roc_auc")

#### atualizar wkf com modelo selecionado ####  
#atualizar workflow
  xgb_wkf <<- xgb_wkf %>% 
    tune::finalize_workflow(best_5)

print("Seu modelo está pronto para ser testado!")

print("Utilize o objeto xgb_wk para o teste do modelo")

}

