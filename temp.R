library(png)

readPNG('figures/br_graph.png') %>% plot

library(magick)

br <- image_read('figures/br_graph.svg')
br <- image_read('br_graph.pdf')

cc <- image_read('figures/cc_graph.svg')
cc <- image_read('cc_graph.pdf')

lp <- image_read('figures/lp_graph.svg')

image_append(c(image_append(c(br, cc), stack = TRUE), lp)) %>% image_write('figures/pt.svg')

library(tidyverse)

df <- read_csv('data/chest_labels.csv')

colnames(df)[1:6] <- c("index", "labels", "follow_up", "id", "age", "gender")

L <- unique(unlist(sapply(df$labels, function(a) {
  temp <- unlist(strsplit(a, '\\|'))
  temp[temp != 'No Finding']
})))
L <- sort(L)
Y_mat <- t(sapply(df$labels, function(a) {
  temp <- unlist(strsplit(a, "\\|"))
  as.numeric(L %in% temp)
}))

X <- df %>% select(follow_up, age, gender)
rm(df)

X$age <- sapply(X$age, function(a) {
  metric <- unlist(strsplit(a, ""))[nchar(a)]
  age <- as.numeric(strtrim(a, 3))
  ifelse(metric == "Y", age, floor(age/12))
})

X$age[X$age>100] <- 100

X$age <- X$age/max(X$age)

X$gender <- as.numeric(factor(X$gender)) - 1

X$follow_up <- as.numeric(X$follow_up)

X$follow_up <- X$follow_up/max(X$follow_up)

X %>% mutate(y = Y_mat[,9]) %>% 
  ggplot(aes(follow_up, age)) + geom_point(aes(color = factor(y)))# + geom_smooth(method = "lm")

summary(X)

X <- as.matrix(X)

library(keras)

inp <- layer_input(shape = 3)
out1 <- inp %>% 
  layer_dense(512) %>% layer_batch_normalization() %>% layer_activation('relu') %>%
  layer_dense(512) %>% layer_batch_normalization() %>% layer_activation('relu') %>%
  layer_dense(14, activation = 'sigmoid')
out2 <- out1 %>% 
  layer_dense(64) %>% layer_batch_normalization() %>% layer_activation('relu') %>%
  layer_dense(64) %>% layer_batch_normalization() %>% layer_activation('relu') %>%
  layer_dense(14, activation = 'sigmoid')

nn <- keras_model(inp, list(out1, out2))

nn %>% compile(optimizer = "sgd", loss = 'binary_crossentropy')

nn %>% 
  fit(X, list(Y_mat, Y_mat), batch_size = 64, epochs = 30, validation_split = 0.15)

Y_hat <- nn %>% predict(X, verbose=1)

?AUC::auc()

??mlr::auc



