library(RJSONIO)
library(itertools)
library(iterators)
library(doParallel)
library(parallelly)

library(transport)

#####################
#####################
#####################
### Sequence Data ### 
  
sequence_data <- function(data, K) {
    list_of_batches <- list()
    
    T <- nrow(data)
    
    nr_batches_int <- T %/% K
    
    for (i in seq_len(nr_batches_int)) {
      start_idx <- (i - 1) * K + 1
      end_idx <- i * K
      list_of_batches[[i]] <- data[start_idx:end_idx, ]
    }
    
    return(list_of_batches)
}


get_threshold_Wd <- function(list_of_data, eps, method = "networkflow", p = 2) {
  nc <- ncol(list_of_data[[1]])
  nr <- nrow(list_of_data[[1]])
  combinations <- combn(list_of_data, 2)
  wd_values <-  eps * sapply(1:ncol(combinations), function(i) {
    wasserstein(pp(matrix(combinations[[1, i]],ncol = nc, nrow = nr)), 
                pp(matrix(combinations[[2, i]], ncol = nc, nrow = nr)), 
                method="networkflow", p=p)
  })
  return(max(wd_values))
  }


get_wd_optimized <- function(batch, list_of_data, p = 2) {
    nc <- ncol(batch)
    nr <- nrow(batch)
    wd_values <- sapply(list_of_data, function(refbatch) {
                        wasserstein(pp(matrix(batch,ncol = nc,nrow=nr)), 
                                    pp(matrix(refbatch,ncol = nc,nrow=nr)), 
                                    p = p, method="networkflow")
    })
  return(max(wd_values))
}


run_WATCH<- function(data, kappa = 3, mu = 8, eps = 1.5, K = 9) {
    data_split <- sequence_data(data, K)
    T <- length(data_split)
    
    current_distribution <- list()
    cp_list <- integer(0)  # Initialize as an empty integer vector (for better performance)
    
    for (i in 1:T) {
      if (length(current_distribution) < kappa) {
        current_distribution <- c(current_distribution, list(data_split[[i]]))
        if (length(current_distribution) >= kappa) {
          threshold <- get_threshold_Wd(current_distribution, eps)
        }
      } else {
        swd <- get_wd_optimized(data_split[[i]], current_distribution)
        
        if (swd > threshold) {
          cp <- (i-1) * K
          cp_list <- c(cp_list, cp)
          current_distribution <- list(data_split[[i]])
        } else if (length(current_distribution) < mu) {
          current_distribution <- c(current_distribution, list(data_split[[i]]))
          threshold <- get_threshold_Wd(current_distribution, eps)
        } else {
          current_distribution <- current_distribution[-1]
          current_distribution <- c(current_distribution, list(data_split[[i]]))
          threshold <- get_threshold_Wd(current_distribution, eps)
        }
      }
    }
    
    return(cp_list)
  }

  #####################
  #####################
  #####################


META_DATA <- fromJSON('config.json')
DATASET_PATH <- META_DATA$Dataset_DIR
DATASETS <- META_DATA$Datasets
RESULTS_PATH <- META_DATA$Results_DIR
GRID <- META_DATA$METHODS$WATCH
print(GRID)

grid <- expand.grid(K = GRID$K, kappa = GRID$kappa, mu = GRID$mu, eps=GRID$eps)
print(dim(grid))

load.utils <- function() {
   utils.script <- file.path('~/Github/Benchmark_SWATCH/methods', 'utils.R')
   source(utils.script)
}

defaults <- list(K = 9, eps=1.5, kappa = 3, mu = 8)


n.cores <- parallelly::availableCores()
print(n.cores)
my.cluster <- parallelly::makeClusterPSOCK(workers = n.cores)
print(my.cluster)
doParallel::registerDoParallel(cl = my.cluster)
foreach::getDoParRegistered()

print(DATASETS)
load.utils()


##############
for (name in DATASETS){ 
    tmp_path <- paste(file.path(DATASET_PATH,name),".json", sep="")
    print(tmp_path)
    ### check if MNIST
    if (grepl('MNIST',name)){
      data <- load.dataset(tmp_path,FALSE)
    }
    else{
    data <-  load.dataset(tmp_path)
    }
    mat <- data$mat


    #defaults
    print(run_WATCH(data = mat, K = defaults$K, kappa = defaults$kappa, mu = defaults$mu, eps = defaults$eps))
    start.time <- Sys.time()
        result <- tryCatch({
                locs <- run_WATCH(data = mat, K = defaults$K, kappa = defaults$kappa, mu = defaults$mu, eps = defaults$eps)
                list(locations=locs, error=NULL)
                 list(SETTING = paste(defaults$K, defaults$eps,
                              defaults$kappa, defaults$mu,sep = "_"), info = list(Method="WATCH", params = defaults, cp=locs, runtime = 0, error = NULL))
            }, error=function(e) {
                return(list(SETTING = paste(defaults$K, defaults$eps,
                              defaults$kappa, defaults$mu,sep = "_"), info = list(Method="WATCH", params = defaults, cp=locs, runtime = NULL, error = e$message)))
            })
        stop.time <- Sys.time()
        runtime <- difftime(stop.time, start.time, units="secs")
        if (is.null(result$info$error)){
         result$info$runtime <- runtime
        }

        output_dir <- file.path(RESULTS_PATH,name)
        print(output_dir)

        if (!dir.exists(output_dir)){
            dir.create(output_dir)
            print("create new dir")
        }
        default_output_dir <- file.path(output_dir,"default")
        if (!dir.exists(default_output_dir)){
            dir.create(default_output_dir)
            print("create new dir")
        }
         # else safe with name default_NAMEMETHOD
         else {
               outJson <- toJSON(result,pretty=T)
               file_name = file.path(default_output_dir,"default_WATCH.json")
               write(outJson,file_name)
         }


#### GRID SEARCH #### 

   out_long <- foreach(K = grid$K,
                    mu = grid$mu,
                    kappa= grid$kappa,
                    eps = grid$eps,
                    .combine = 'c',
                    .packages = c('transport','RJSONIO')
                    ) %dopar% {
                        tmp_params <- list(K = K, eps = eps, mu = mu, kappa = kappa)
                        result <- tryCatch({
                        start.time <- Sys.time()
                        locs <- run_WATCH(data = mat, K = K, eps= eps, mu = mu, kappa = kappa)
                         stop.time <- Sys.time()
                        runtime <- difftime(stop.time, start.time, units="secs")
        list(SETTING = paste(tmp_params$K, tmp_params$eps,
                              tmp_params$kappa, tmp_params$mu,sep = "_"), info = list(Method="WATCH", params = tmp_params, cp=locs, runtime = runtime, error = NULL))
    }, error=function(e) {
        return(list(SETTING = paste(tmp_params$K, tmp_params$eps,
                              tmp_params$kappa, tmp_params$mu,sep = "_"), info = list(Method="WATCH", params = tmp_params, cp=locs, runtime = NULL, error = e$message)))
    })
        }

   temp_output_dir <- file.path(output_dir,"oracle_WATCH")
        if (!dir.exists(temp_output_dir)){
            dir.create(temp_output_dir)
            print("create new dir")
        }
   #print(out_long)

   for (i in 1:length(out_long)){
        if(!is.null(out_long[i]$SETTING)){
            temp_file <- c(out_long[i], out_long[i+1])
            outJson <- toJSON(temp_file,pretty=T)
            file_name = file.path(temp_output_dir,paste(temp_file$SETTING,"json",sep="."))
            write(outJson,file_name)
        }
    }

}
