#' ---
#' title: Utilities shared between R code
#' author: G.J.J. van den Burg
#' date: 2019-09-29
#' license: See the LICENSE file.
#' copyright: 2019, The Alan Turing Institute
#' ---

library(RJSONIO)

printf <- function(...) invisible(cat(sprintf(...)));

#' Load a TCPDBench dataset
#'
#' This function reads in a JSON dataset in TCPDBench format (see TCPD 
#' repository for schema) and creates a matrix representation of the dataset.  
#' The dataset is scaled in the process.
#'
#' @param filename Path to the JSON file
#' @return List object with the raw data in the \code{original} field, the time 
#' index in the \code{time} field, and the data matrix in the \code{mat} field.
#'
load.dataset <- function(filename,normalize=TRUE)
{
    data <- fromJSON(filename)

    # reformat the data to a data frame with a time index and the data values
    tidx <- data$time$index
    exp <- 0:(data$n_obs - 1)
    if (all(tidx == exp) && length(tidx) == length(exp)) {
        tidx <- NULL
    } else {
        tidx <- data$time$index
    }

    mat <- NULL

    for (j in 1:data$n_dim) {
        s <- data$series[[j]]
        v <- NULL
        for (i in 1:data$n_obs) {
            val <- s$raw[[i]]
            if (is.null(val)) {
                v <- c(v, NA)
            } else {
                v <- c(v, val)
            }
        }
        mat <- cbind(mat, v)
    }

    # We normalize to avoid issues with numerical precision.
    if (normalize) {
    mat <- scale(mat)
    }
    out <- list(original=data,
                time=tidx,
                mat=mat)
    
    return(out)
}


Save_Result <- function(file,name,method){
        0
}

print('utils loaded')

