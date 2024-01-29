library(ocp)
library(RJSONIO)
library(itertools)
library(iterators)
library(doParallel)
library(parallelly)

META_DATA <- fromJSON('config.json')
DATASET_PATH <- META_DATA$Dataset_DIR
DATASETS <- META_DATA$Datasets
RESULTS_PATH <- META_DATA$Results_DIR
GRID <- META_DATA$METHODS$BOCPD
UTILS <- META_DATA$UTILS
print(GRID)
grid <- expand.grid(lambda = GRID$lambda, prior_a = GRID$prior_a, prior_b = GRID$prior_b, prior_k = GRID$prior_k)
print(dim(grid))

print(UTILS)

load.utils <- function() {
   #utils.script <- file.path('~/Github/Benchmark_SWATCH/methods', 'utils.R')
   #utils.script <- file.path(UTILS)
   #source(utils.script)
   source(UTILS)
}

hazard_func <- function(x, lambda) {
        const_hazard(x, lambda=params$lambda)

    }


defaults <- list(missPts="none",
                     cpthreshold=0.5, # unused by us
                     truncRlim=10^(-4),
                     minRlength=1,
                     maxRlength=10^4, # bigger than any of our datasets
                     minsep=1,
                     maxsep=10^4 # bigger than any of our datasets
                     )

Detection <-function(mat,params,model.params,name){
#params --> default parameter other than model parameter
    start.time <- Sys.time()
    result <- tryCatch({
        fit <- onlineCPD(mat, oCPD=NULL, missPts=params$missPts,
                         hazard_func=hazard_func, 
                         probModel=list("gaussian"),
                         init_params=model.params,
                         multivariate=params$multivariate,
                         cpthreshold=params$cpthreshold,
                         truncRlim=params$truncRlim,
                         minRlength=params$minRlength,
                         maxRlength=params$maxRlength,
                         minsep=params$minsep,
                         maxsep=params$maxsep
                         )
        locs <- as.vector(fit$changepoint_lists$maxCPs[[1]])
        list(locations=locs, error=NULL)
    }, error=function(e) {
        return(list(locations=NULL, error=e$message))
    })
    stop.time <- Sys.time()
    runtime <- difftime(stop.time, start.time, units="secs")
    out <- list(Name=name, Method="BOCPD", params = model.params, cp=result$locations)
}


load.utils()

#Make cluster for parallelization
n.cores <- parallelly::availableCores()
print(n.cores)
my.cluster <- parallelly::makeClusterPSOCK(workers = n.cores)
print(my.cluster)
doParallel::registerDoParallel(cl = my.cluster)
foreach::getDoParRegistered()

print(DATASETS)

#loop over Datasets to only read dataset once and run grid search (only imports libs once per method)
for (name in DATASETS){

tmp_path <- file.path(DATASET_PATH,name)
unzip_dir <- file.path(dirname(DATASET_PATH),"methods/python/unzip.py")

 params <- defaults


#run python script 
#1) check if zip file
#2) if zip file then unzip into tmp and return temp path
out_script <- system(paste("python3",unzip_dir,file.path(dirname(DATASET_PATH),"tmp"),tmp_path),intern=TRUE)

isZip <- as.logical(out_script)

zipped_files_path <-  file.path(dirname(DATASET_PATH),"tmp")

#part where datasets are in zip file
if (isZip){
    output_dir <- file.path(RESULTS_PATH,name)
    if (!dir.exists(output_dir)){
             dir.create(output_dir)
             print("create new dir")
    }
    for(file in list.files(zipped_files_path)[1:3]){
        fname <- sub("\\.[^.]+$", "", file)
        file_dir = file.path(output_dir,fname)
        if (!dir.exists(file_dir)){
         dir.create(file_dir)
        }
        if (grepl('MNIST',fname)){
            data <- load.dataset(file.path(zipped_files_path,file),FALSE)  
        }
        else{
            data <- load.dataset(file.path(zipped_files_path,file),TRUE)
        }
        mat <- data$mat

       out_long <- foreach(lambda_in = grid$lambda,
                    prior_a = grid$prior_a,
                    prior_b = grid$prior_b,
                    prior_k = grid$prior_k,
                    .combine = 'c',
                    .packages = c('ocp','RJSONIO')
                    ) %dopar% {
                        tmp_params <- list(lambda = lambda_in, prior_a=prior_a, prior_b = prior_b , prior_k = prior_k)
                        result <- tryCatch({
                            #hazard_func=function(x, lambda) { const_hazard(x, lambda)}
                    start.time <- Sys.time()
                    fit <- ocp::onlineCPD(mat, oCPD=NULL, missPts=params$missPts,
                         hazard_func=function(x, lambda) { const_hazard(x, lambda=lambda_in)}, 
                         probModel=list("gaussian"),
                         init_params=list(list(m=0, k=prior_k, a=prior_a,b=prior_b)),
                         multivariate=params$multivariate,
                         cpthreshold=params$cpthreshold,
                         truncRlim=params$truncRlim,
                         minRlength=params$minRlength,
                         maxRlength=params$maxRlength,
                         minsep=params$minsep,
                         maxsep=params$maxsep
                         )
                         stop.time <- Sys.time()
        runtime <- difftime(stop.time, start.time, units="secs")
        locs <- as.vector(fit$changepoint_lists$maxCPs[[1]])
       list(SETTING = paste(lambda_in,prior_a,prior_b,prior_k,sep = "_"), info = list(Method="BOCPD", params = tmp_params, cp=locs, runtime = runtime, error = NULL))
    }, error=function(e) {
        return(list(SETTING = paste(lambda_in,prior_a,prior_b,prior_k,sep = "_"), info = list(Method="BOCPD", params = tmp_params, cp=NULL, runtime = NULL, error=e$message)))
    })
        }

    #create BOCPD oracle Dir
    BOCPD_ORACLE_DIR <- file.path(file_dir,"oracle_bocpd")
    if (!dir.exists(BOCPD_ORACLE_DIR)){
             dir.create(BOCPD_ORACLE_DIR)
             print("create new dir")
    }

     for (i in 1:length(out_long)){
        if(!is.null(out_long[i]$SETTING)){
            temp_file <- c(out_long[i], out_long[i+1])
            outJson <- toJSON(temp_file,pretty=T)
            file_name = file.path(BOCPD_ORACLE_DIR,paste(temp_file$SETTING,"json",sep="."))
            write(outJson,file_name)
        }
}
    }
}
#no zip files
else{
    output_dir <- file.path(RESULTS_PATH,name)
    print(paste(file.path(DATASET_PATH,name),".json", sep=""))
    data <- load.dataset(paste(file.path(DATASET_PATH,name),".json", sep=""),TRUE)
    mat <- data$mat
    out_long <- foreach(lambda_in = grid$lambda,
                    prior_a = grid$prior_a,
                    prior_b = grid$prior_b,
                    prior_k = grid$prior_k,
                    .combine = 'c',
                    .packages = c('ocp','RJSONIO')
                    ) %dopar% {
                        tmp_params <- list(lambda = lambda_in, prior_a=prior_a, prior_b = prior_b , prior_k = prior_k)
                        result <- tryCatch({
                            #hazard_func=function(x, lambda) { const_hazard(x, lambda)}
                    start.time <- Sys.time()
                    fit <- ocp::onlineCPD(mat, oCPD=NULL, missPts=params$missPts,
                         hazard_func=function(x, lambda) { const_hazard(x, lambda=lambda_in)}, 
                         probModel=list("gaussian"),
                         init_params=list(list(m=0, k=prior_k, a=prior_a,b=prior_b)),
                         multivariate=params$multivariate,
                         cpthreshold=params$cpthreshold,
                         truncRlim=params$truncRlim,
                         minRlength=params$minRlength,
                         maxRlength=params$maxRlength,
                         minsep=params$minsep,
                         maxsep=params$maxsep
                         )
                         stop.time <- Sys.time()
        runtime <- difftime(stop.time, start.time, units="secs")
        locs <- as.vector(fit$changepoint_lists$maxCPs[[1]])
       list(SETTING = paste(lambda_in,prior_a,prior_b,prior_k,sep = "_"), info = list(Method="BOCPD", params = tmp_params, cp=locs, runtime = runtime, error = NULL))
    }, error=function(e) {
        return(list(SETTING = paste(lambda_in,prior_a,prior_b,prior_k,sep = "_"), info = list(Method="BOCPD", params = tmp_params, cp=NULL, runtime = NULL, error=e$message)))
    })
        }


       if (!dir.exists(output_dir)){
            dir.create(output_dir)
            print("create new dir")
        }


        temp_output_dir <- file.path(output_dir,"oracle_bocpd")
        if (!dir.exists(temp_output_dir)){
            dir.create(temp_output_dir)
            print("create new dir")
        }
  
   for (i in 1:length(out_long)){
        if(!is.null(out_long[i]$SETTING)){
            temp_file <- c(out_long[i], out_long[i+1])
            outJson <- toJSON(temp_file,pretty=T)
            file_name = file.path(temp_output_dir,paste(temp_file$SETTING,"json",sep="."))
            write(outJson,file_name)
        }
    }


}
}
# for (name in DATASETS){ 
#     tmp_path <- paste(file.path(DATASET_PATH,name),".json", sep="")
#     print(tmp_path)
#     #it <- ihasNext(product(lambda = GRID$lambda, prior_a = GRID$prior_a, prior_b = GRID$prior_b, prior_k = GRID$prior_k))
#     if (grepl('MNIST',name)){
#       data <- load.dataset(tmp_path,FALSE)
#     }
#     else{
#     data <-  load.dataset(tmp_path)
#     }
#     mat <- data$mat
#     defaults$multivariate = ncol(mat) > 1 

#     params <- defaults


#         #default model parameter
#         params$prior_k = 1.0
#         params$prior_a = 1.0
#         params$prior_b = 1.0
#         params$lambda = 100

#         model.params <- list(list(m=0, k=params$prior_k, a=params$prior_a,
#                                         b=params$prior_b))
#         start.time <- Sys.time()
#         result <- tryCatch({
#                 fit <- onlineCPD(mat, oCPD=NULL, missPts=params$missPts,
#                                 hazard_func=hazard_func, 
#                                 probModel=list("gaussian"),
#                                 init_params=model.params,
#                                 multivariate=params$multivariate,
#                                 cpthreshold=params$cpthreshold,
#                                 truncRlim=params$truncRlim,
#                                 minRlength=params$minRlength,
#                                 maxRlength=params$maxRlength,
#                                 minsep=params$minsep,
#                                 maxsep=params$maxsep
#                                 )
#                 locs <- as.vector(fit$changepoint_lists$maxCPs[[1]])
#                 list(locations=locs, error=NULL)
#             }, error=function(e) {
#                 return(list(locations=NULL, error=e$message))
#             })
#         stop.time <- Sys.time()
#         runtime <- difftime(stop.time, start.time, units="secs")
#         out <- list(Name=name, Method="BOCPD", params = model.params, cp=result$locations)
#         print(result$locations)
#         print(runtime)
#         outJson <- toJSON(out)
#         #where to store results 
#         #check if dir already exist
#         output_dir <- file.path(RESULTS_PATH,name)
#         #print(output_dir)
#         if (!dir.exists(output_dir)){
#             dir.create(output_dir)
#             print("create new dir")
#         }
#         default_output_dir <- file.path(output_dir,"default")
#         if (!dir.exists(default_output_dir)){
#             dir.create(default_output_dir)
#             print("create new dir")
#         }
        
#         out_long <- foreach(lambda_in = grid$lambda,
#                     prior_a = grid$prior_a,
#                     prior_b = grid$prior_b,
#                     prior_k = grid$prior_k,
#                     .combine = 'c',
#                     .packages = c('ocp','RJSONIO')
#                     ) %dopar% {
#                         tmp_params <- list(lambda = lambda_in, prior_a=prior_a, prior_b = prior_b , prior_k = prior_k)
#                         result <- tryCatch({
#                             #hazard_func=function(x, lambda) { const_hazard(x, lambda)}
#                     start.time <- Sys.time()
#                     fit <- ocp::onlineCPD(mat, oCPD=NULL, missPts=params$missPts,
#                          hazard_func=function(x, lambda) { const_hazard(x, lambda=lambda_in)}, 
#                          probModel=list("gaussian"),
#                          init_params=list(list(m=0, k=prior_k, a=prior_a,b=prior_b)),
#                          multivariate=params$multivariate,
#                          cpthreshold=params$cpthreshold,
#                          truncRlim=params$truncRlim,
#                          minRlength=params$minRlength,
#                          maxRlength=params$maxRlength,
#                          minsep=params$minsep,
#                          maxsep=params$maxsep
#                          )
#                          stop.time <- Sys.time()
#         runtime <- difftime(stop.time, start.time, units="secs")
#         locs <- as.vector(fit$changepoint_lists$maxCPs[[1]])
#        list(SETTING = paste(lambda_in,prior_a,prior_b,prior_k,sep = "_"), info = list(Method="BOCPD", params = tmp_params, cp=locs, runtime = runtime, error = NULL))
#     }, error=function(e) {
#         return(list(SETTING = paste(lambda_in,prior_a,prior_b,prior_k,sep = "_"), info = list(Method="BOCPD", params = tmp_params, cp=NULL, runtime = NULL, error=e$message)))
#     })
#         }
#         # while (hasNext(it)){
#     #     #         x <- nextElem(it)
#     #     #         model.params <- list(list(m=0, k=x$prior_k, a=x$prior_a,
#     #     #                                 b=x$prior_b))
#     #     #         params$lambda <- x$lambda
#     #     #         out <- Detection(mat,params,model.params,name)
#     #     #     print(i)
#     #     # i=i+1
#     #     # }
#     temp_output_dir <- file.path(output_dir,"oracle_bocpd")
#         if (!dir.exists(temp_output_dir)){
#             dir.create(temp_output_dir)
#             print("create new dir")
#         }


#     #dump json file for each experiment
#     for (i in 1:length(out_long)){
#         if(!is.null(out_long[i]$SETTING)){
#             temp_file <- c(out_long[i], out_long[i+1])
#             outJson <- toJSON(temp_file,pretty=T)
#             file_name = file.path(temp_output_dir,paste(temp_file$SETTING,"json",sep="."))
#             write(outJson,file_name)
#         }
#     }

    # outJson <- toJSON(out_long,pretty=T)
    # file_name = file.path(temp_output_dir,"results.json")
    # write(outJson,file_name)









