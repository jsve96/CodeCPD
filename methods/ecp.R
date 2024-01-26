library(ecp)
library(RJSONIO)
library(itertools)
library(iterators)
library(doParallel)
library(parallelly)

META_DATA <- fromJSON('config.json')
DATASET_PATH <- META_DATA$Dataset_DIR
DATASETS <- META_DATA$Datasets
RESULTS_PATH <- META_DATA$Results_DIR
GRID <- META_DATA$METHODS$ECP
UTILS <- META_DATA$UTILS
print(GRID)
# ECP": {
#             "algorithm": [
#                 "e.agglo",
#                 "e.divisive"
#             ],
#             "siglvl": [
#                 0.01,
#                 0.05
#             ],
#             "minsize": [
#                 2,
#                 30
#             ],
#             "alpha": [
#                 0.5,
#                 1.0,
#                 1.5
#             ]
grid <- expand.grid(algorithm = GRID$algorithm, siglvl = GRID$siglvl, minsize = GRID$minsize, alpha = GRID$alpha)
print(dim(grid))

load.utils <- function() {
   source(UTILS)
}



defaults <- list(k =  NULL , algorithm = "e.divisive", alpha = 1, cost = 1, minsize=30, runs = 199, siglvl = 0.05)


n.cores <- parallelly::availableCores()
print(n.cores)
my.cluster <- parallelly::makeClusterPSOCK(workers = n.cores)
print(my.cluster)
doParallel::registerDoParallel(cl = my.cluster)
foreach::getDoParRegistered()

print(DATASETS)
load.utils()


for (name in DATASETS){ 
    tmp_path <- paste(file.path(DATASET_PATH,name),".json", sep="")
    print(tmp_path)

    if (grepl('MNIST',name)){
      data <- load.dataset(tmp_path,FALSE)
    }
    else{
    data <-  load.dataset(tmp_path)
    }
    # mat <- data$mat
    # data <-  load.dataset(tmp_path)
    mat <- data$mat

   #### DEFAULT PARAMETER ####

   start.time <- Sys.time()
        result <- tryCatch({
                fit <-  e.divisive(mat, sig.lvl=defaults$siglvl, R=defaults$runs,
                              min.size=defaults$minsize, alpha=defaults$alpha)
                locs <- fit$estimates
                list(locations=locs, error=NULL)
                 list(SETTING = paste(defaults$siglvl, defaults$runs,
                              defaults$minsize, defaults$alpha,sep = "_"), info = list(Method="ECP", params = defaults, cp=locs, runtime = 0, error = NULL))
            }, error=function(e) {
                return(list(SETTING = paste(defaults$siglvl, defaults$runs,
                              defaults$minsize, defaults$alpha,sep = "_"), info = list(Method="ECP", params = defaults, cp=locs, runtime = NULL, error = e$message)))
            })
        stop.time <- Sys.time()
        runtime <- difftime(stop.time, start.time, units="secs")
        if (is.null(result$info$error)){
         result$info$runtime <- runtime
        }
        
        print(runtime)
        
        output_dir <- file.path(RESULTS_PATH,name)
        #print(output_dir)

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
               file_name = file.path(default_output_dir,"default_ECP.json")
               write(outJson,file_name)
         }

   #### GRID SEARCH #### 

   out_long <- foreach(algorithm = grid$algorithm,
                    siglvl = grid$siglvl,
                    minsize= grid$minsize,
                    alpha = grid$alpha,
                    .combine = 'c',
                    .packages = c('ecp','RJSONIO')
                    ) %dopar% {
                        tmp_params <- list(algorithm = algorithm, siglvl = siglvl, minsize = minsize, alpha = alpha)
                        result <- tryCatch({
                        start.time <- Sys.time()
                        if (tmp_params$algorithm == "e.agglo"){
                           fit <- e.agglo(mat,alpha = alpha)
                           locs <- fit$estimates
                        }
                        else {
                           fit <- e.divisive(mat, sig.lvl=tmp_params$siglvl, R=defaults$runs,
                              min.size=tmp_params$minsize, alpha=tmp_params$alpha)
                           locs <- fit$estimates
                        }
                         stop.time <- Sys.time()
                        runtime <- difftime(stop.time, start.time, units="secs")
        list(SETTING = paste(tmp_params$siglvl, tmp_params$algorithm,
                              tmp_params$minsize, tmp_params$alpha,sep = "_"), info = list(Method="ECP", params = tmp_params, cp=locs, runtime = runtime, error = NULL))
    }, error=function(e) {
        return(list(SETTING = paste(tmp_params$siglvl, tmp_params$algorithm,
                              tmp_params$minsize, tmp_params$alpha,sep = "_"), info = list(Method="ECP", params = tmp_params, cp=locs, runtime = NULL, error = e$message)))
    })
        }

   temp_output_dir <- file.path(output_dir,"oracle_ecp")
        if (!dir.exists(temp_output_dir)){
            dir.create(temp_output_dir)
            print("create new dir")
        }
   print(out_long)

   for (i in 1:length(out_long)){
        if(!is.null(out_long[i]$SETTING)){
            temp_file <- c(out_long[i], out_long[i+1])
            outJson <- toJSON(temp_file,pretty=T)
            file_name = file.path(temp_output_dir,paste(temp_file$SETTING,"json",sep="."))
            write(outJson,file_name)
        }
    }


        
}
