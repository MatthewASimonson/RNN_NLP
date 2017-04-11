# MxNet Example: 
# Use an LSTM RNN model to build a char-level language model
# 
# Matthew A Simonson PhD, 4-10-2017

# load packages:
###########

require(mxnet)

# Code functions used in analysis:
###########

# Function to make a dictionary for all unique characters in the text:
make.dict <- function(text, max.vocab=1000000) {
  text <- strsplit(text, '') # split string into separate parts when ' ' is ecountered
  dic <- list() # use list for dictionary
  idx <- 1 # index starting at 1
  for (c in text[[1]]) { # append dictionary when next character is encountered
    if (!(c %in% names(dic))) {
      dic[[c]] <- idx
      idx <- idx + 1
    }
  }
  if (length(dic) == max.vocab - 1)
    dic[["UNKNOWN"]] <- idx
  cat(paste0("Total unique char: ", length(dic), "\n"))
  return (dic)
}


make.data <- function(file.path, seq.len=32, max.vocab=10000, dic=NULL) {
  fi <- file(file.path, "r")
  text <- paste(readLines(fi), collapse="\n")
  close(fi)
  
  if (is.null(dic))
    dic <- make.dict(text, max.vocab) # create dictionary from characters
  lookup.table <- list()
  for (c in names(dic)) {
    idx <- dic[[c]]
    lookup.table[[idx]] <- c 
  }
  
  char.lst <- strsplit(text, '')[[1]]
  num.seq <- as.integer(length(char.lst) / seq.len)
  char.lst <- char.lst[1:(num.seq * seq.len)]
  data <- array(0, dim=c(seq.len, num.seq))
  idx <- 1
  for (i in 1:num.seq) {
    for (j in 1:seq.len) {
      if (char.lst[idx] %in% names(dic))
        data[j, i] <- dic[[ char.lst[idx] ]]-1
      else {
        data[j, i] <- dic[["UNKNOWN"]]-1
      }
      idx <- idx + 1
    }
  }
  return (list(data=data, dic=dic, lookup.table=lookup.table))
}

# Function to drop tail
drop.tail <- function(X, batch.size) {
  shape <- dim(X)
  nstep <- as.integer(shape[2] / batch.size)
  return (X[, 1:(nstep * batch.size)])
}

# Function to get label of X
get.label <- function(X) {
  label <- array(0, dim=dim(X))
  d <- dim(X)[1]
  w <- dim(X)[2]
  for (i in 0:(w-1)) {
    for (j in 1:d) {
      label[i*d+j] <- X[(i*d+j)%%(w*d)+1]
    }
  }
  return (label)
}

# cumulative density function
cdf <- function(weights) {
  total <- sum(weights)
  result <- c()
  cumsum <- 0
  for (w in weights) {
    cumsum <- cumsum+w
    result <- c(result, cumsum / total)
  }
  return (result)
}

search.val <- function(cdf, x) {
  l <- 1
  r <- length(cdf) 
  while (l <= r) {
    m <- as.integer((l+r)/2)
    if (cdf[m] < x) {
      l <- m+1
    } else {
      r <- m-1
    }
  }
  return (l)
}

#  random output or fixed output by choosing largest probability
choice <- function(weights) {
  cdf.vals <- cdf(as.array(weights))
  x <- runif(1)
  idx <- search.val(cdf.vals, x)
  return (idx)
}

# Make output

make.output <- function(prob, sample=FALSE) {
  if (!sample) {
    idx <- which.max(as.array(prob))
  }
  else {
    idx <- choice(prob)
  }
  return (idx)
}

# set working directory:
###########

setwd("/Users/masimonson/Documents/Code/R/R_2017/MxNetExample/NLP")

# Get the training, validation data:
###########

system("wget https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tinyshakespeare/input.txt")

## Specify text file path:
###########

file.path <- "input.txt"

## Set the basic network parameters:
###########

batch.size <-  100
seq.len <-  100
num.hidden <-  32
num.embed <-  16
num.lstm.layer <-  16
num.round <-  1
learning.rate<-  0.1
wd<- 0.00001
clip_gradient<- 1
update.period <-  1

# Format data for input into network,
# including raw data,
# dictionary,
# and lookup table
###########

ret <- make.data("input.txt", seq.len=seq.len) # specify input text and length of sequence limit to read
X <- ret$data # data 
dic <- ret$dic # dictionary
lookup.table <- ret$lookup.table # lookup table

vocab <- length(dic) # number of unique characters (size of dictionary)

shape <- dim(X)
train.val.fraction <- 0.9 # set percentage of data to be used as training data
size <- shape[2] 

# split in to training and validation sets (specifying columns of data)
X.train.data <- X[, 1:as.integer(size * train.val.fraction)]
X.val.data <- X[, -(1:as.integer(size * train.val.fraction))]

#
X.train.data <- drop.tail(X.train.data, batch.size)
X.val.data <- drop.tail(X.val.data, batch.size)

X.train.label <- get.label(X.train.data)
X.val.label <- get.label(X.val.data)

X.train <- list(data=X.train.data, label=X.train.label)
X.val <- list(data=X.val.data, label=X.val.label)


# Train a general lstm model:
model <- mx.lstm(X.train, X.val, 
                 ctx=mx.cpu(),
                 num.round=num.round, 
                 update.period=update.period,
                 num.lstm.layer=num.lstm.layer, 
                 seq.len=seq.len,
                 num.hidden=num.hidden, 
                 num.embed=num.embed, 
                 num.label=vocab,
                 batch.size=batch.size, 
                 input.size=vocab,
                 initializer=mx.init.uniform(0.1), 
                 learning.rate=learning.rate,
                 wd=wd,
                 clip_gradient=clip_gradient)

# function called mx.lstm.inference will build inference from lstm model
# and then use function mx.lstm.forward to get forward output from the inference.
# Build inference from model:

infer.model <- mx.lstm.inference(num.lstm.layer=num.lstm.layer,
                                 input.size=vocab,
                                 num.hidden=num.hidden,
                                 num.embed=num.embed,
                                 num.label=vocab,
                                 arg.params=model$arg.params,
                                 ctx=mx.cpu())

# generate a sequence of 75 chars using function mx.lstm.forward:

start <- 'a' # starting index in dictionary
seq.len <- 75 # specify length of output sequence
random.sample <- TRUE

last.id <- dic[[start]]
out <- "The "
for (i in (1:(seq.len-1))) {
  input <- c(last.id-1)
  ret <- mx.lstm.forward(infer.model, input, FALSE)
  infer.model <- ret$model
  prob <- ret$prob
  last.id <- make.output(prob, random.sample)
  out <- paste0(out, lookup.table[[last.id]])
}
cat (paste0(out, "\n"))

# For a custom RNN model, you can replace mx.lstm with mx.rnn to train an RNN model.
# You can replace mx.lstm.inference and mx.lstm.forward with mx.rnn.inference and
# mx.rnn.forward to build inference from an RNN model and get the forward result from
# the inference model.
