# ---- Data processing and dimensionality reduction ----

# Get required packages
source('./R/00 Requirements.R')

# Get functions to read MNIST data
source('./R/00 Read MNIST script.R')

# ---- Download the data files ----

# Create data folder if it doesn't exist
dir.create(file.path('.', 'Data'), showWarnings = FALSE)

# Create models folder if it doesn't exist
dir.create(file.path('.', 'R/Models'), showWarnings = FALSE)

# Training images
download.file('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
              destfile =  './Data/train_images.gz')
download.file('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
              destfile =  './Data/train_labels.gz')
download.file('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
              destfile =  './Data/test_images.gz')
download.file('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
              destfile =  './Data/test_labels.gz')

# List downloaded files
files <- list.files('./Data/')

# Unzip them
lapply(paste0('./Data/', files), gunzip)

# Load data
train <- load_image_file('Data/train_images')
test <- load_image_file('Data/test_images')

train$y <- load_label_file('Data/train_labels')
test$y <- load_label_file('Data/test_labels')

# Look at an example
show_digit(train$x[5,])
show_digit(train$x[1,])

# ---- Exploration ----

# Frequency of different digits
freqplot <- table(train$y) %>% 
  as.data.frame() %>%
  rename(digit = "Var1") %>%
  arrange(digit, desc(Freq)) %>%
  mutate(digit = as.factor(digit)) %>%
  ggplot(aes(x = digit, y = Freq, fill = digit)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  scale_fill_viridis(discrete = TRUE) +
  theme(panel.grid = element_blank(),
        legend.position = "none")

# ---- PCA ----

# Get features to reduce. Take a random sample from the full set, to reduce computing time.
i <- sample(1:nrow(train$x), size = 10000)
f <- train$x[i,]
labels <- train$y[i]

# Run pca
pca <- prcomp(f)

# View a scree plot
screeplot(pca, type = "line")

# Calculate proportion of variance explained
ev <- pca$sdev**2
desc <- rbind(
  SD = sqrt(ev),
  Proportion = ev/sum(ev),
  Cumulative = cumsum(ev)/sum(ev))

# View proportion explained by first 2 components
desc[,1:2]

# Get number of components needed to explain more than 80% of the variation
c <- (min(which(desc[3,] > 0.8)))
desc[,1:c]


# Plot cumulative variance explained
desc %>%
  t %>%
  as.data.frame() %>%
  mutate(PC = as.numeric(row.names(.))) %>%
  slice(1:c) %>%
  ggplot(aes(x = PC, y = Cumulative, group = 1)) +
  geom_line(colour = "dodgerblue4", size = 1) +
  expand_limits(y = 1) +
  theme_minimal() +
  labs(title="Cumulative variance explained",
       x = "Principal component",
       y = "Proportion of variance explained")


# Visualise the components
pca_plot <- data.frame(digit = factor(labels),
           pc1 = pca$x[,1],
           pc2 = pca$x[,2]) %>%
  ggplot(aes(x = pc1, y = pc2, colour = digit)) +
  geom_point() +
  scale_colour_brewer(palette = "Paired") +
  theme_minimal() +
  guides(colour = guide_legend(override.aes = list(size = 5)))

# ---- t-SNE ----

# t-SNE: more separation and easier to view with MNIST (get reference)
# http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf
tsne <- Rtsne(f, dims = 2, perplexity=30, verbose=TRUE, max_iter = 500)

# Visualise the components
tsne.data <- data.frame(digit = factor(labels),
                        index = i,
                        pc1 = tsne$Y[,1],
                        pc2 = tsne$Y[,2])
tsne_plot <- tsne.data %>%
  ggplot(aes(x = pc1, y = pc2, colour = digit)) +
  geom_point() +
  scale_colour_brewer(palette = "Paired") +
  theme_minimal() +
  guides(colour = guide_legend(override.aes = list(size = 5)))

# Examples
# Some numbers in the 5s space
five.space <- tsne.data %>% filter(pc1 > 0, pc1 < 1.5, pc2 > -10, pc2 < -9)
tmp = five.space %>% filter(digit == 5) %>% sample_n(1) %>% select(index) %>% unlist
tmp2 = five.space %>% filter(digit != 5) %>% sample_n(1) %>% select(index) %>% unlist

# Save plots
png(filename="./Plots/example_fives.png")
par(mfrow=c(1,2))
show_digit(train$x[tmp,])
show_digit(train$x[tmp2,])
dev.off()


# ---- Save plots and data ----

# For use in rmarkdown file
plot.list <- list("tsne_plot",
                  "pca_plot",
                  "cum_plot",
                  "freqplot")

# Save each plot
lapply(plot.list, function(plot) saveRDS(object = get(plot), file = paste0('./Plots/', plot, ".rds")))

# Save pca model
saveRDS(pca, './R/Models/pca.rds')

# Record progress
print("Data processed and dimentionality reduction complete")
rm(list = ls())
gc()
