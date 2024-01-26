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
GRID <- META_DATA$METHODS$KCPA
print(GRID)

grid <- expand.grid(maxcp = GRID$maxcp, cost = GRID$cost)
print(dim(grid))

load.utils <- function() {
   utils.script <- file.path('~/Github/Benchmark_SWATCH/methods', 'utils.R')
   source(utils.script)
}



defaults <- list(maxcp = 'max', cost = 1)


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
    mat <- data$mat
    # data <-  load.dataset(tmp_path)
    # mat <- data$mat
    defaults$L <- dim(mat)[1]
   #### DEFAULT PARAMETER ####

   start.time <- Sys.time()
        result <- tryCatch({
                fit <-  kcpa(mat, L = defaults$L, C = defaults$cost)
                locs <- fit
                list(locations=locs, error=NULL)
                 list(SETTING = paste(defaults$L, defaults$cost,sep = "_"), info = list(Method="KCPA", params = defaults, cp=locs, runtime = 0, error = NULL))
            }, error=function(e) {
                return(list(SETTING = paste(defaults$L, defaults$cost ,sep = "_"), info = list(Method="KCPA", params = defaults, cp=locs, runtime = NULL, error = e$message)))
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
               file_name = file.path(default_output_dir,"default_KCPA.json")
               write(outJson,file_name)
         }

   ### GRID SEARCH #### 

#    out_long <- foreach(maxcp = grid$maxcp, cost = grid$cost,
#                     .combine = 'c',
#                     .packages = c('ecp','RJSONIO')
#                     ) %dopar% {
#                         tmp_params <- list(maxcp = maxcp , cost = cost)
#                         result <- tryCatch({
#                         start.time <- Sys.time()
#                         if (tmp_params$maxcp == "max"){
#                            tmp_params$L <- dim(mat)[1]
#                         }
#                         else {
#                            tmp_params$L <- 5
#                         }
#                         fit <- kcpa(mat, tmp_params$L, tmp_params$cost)
#                         locs <- fit
#                          stop.time <- Sys.time()
#                         runtime <- difftime(stop.time, start.time, units="secs")
#         list(SETTING = paste(tmp_params$L, tmp_params$cost, tmp_params$maxcp ,sep = "_"), info = list(Method="KCPA", params = tmp_params, cp=locs, runtime = runtime, error = NULL))
#     }, error=function(e) {
#         return(list(SETTING = paste(tmp_params$L, tmp_params$cost, tmp_params$maxcp ,sep = "_"), info = list(Method="KCPA", params = tmp_params, cp=locs, runtime = NULL, error = e$message)))
#     })
#         }

#    temp_output_dir <- file.path(output_dir,"oracle_kcpa")
#         if (!dir.exists(temp_output_dir)){
#             dir.create(temp_output_dir)
#             print("create new dir")
#         }
#    print(out_long)

#    for (i in 1:length(out_long)){
#         if(!is.null(out_long[i]$SETTING)){
#             temp_file <- c(out_long[i], out_long[i+1])
#             outJson <- toJSON(temp_file,pretty=T)
#             file_name = file.path(temp_output_dir,paste(temp_file$SETTING,"json",sep="."))
#             write(outJson,file_name)
#         }
#     }


        
}
