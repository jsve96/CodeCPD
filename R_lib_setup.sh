#!/bin/bash

echo "Script executed from: ${PWD}"

RLIBS="${PWD}/rlib"
echo "${RLIBS}"

export _R_CHECK_INSTALL_DEPENDS_="TRUE"
#Rcpp
Rscript -e "install.packages('https://cran.r-project.org/src/contrib/Rcpp_1.0.11.tar.gz', dependencies=TRUE, lib='${RLIBS}')"
#Devtools
Rscript -e "install.packages('https://cran.r-project.org/src/contrib/devtools_2.4.5.tar.gz', dependencies=TRUE, lib='${RLIBS}')"
#ECP/KCPA
Rscript -e "install.packages('https://cran.r-project.org/src/contrib/Archive/ecp/ecp_3.1.1.tar.gz', dependencies=TRUE, lib='${RLIBS}')"
#BOCPD
Rscript -e "install.packages('https://cran.r-project.org/src/contrib/ocp_0.1.1.tar.gz', dependencies=TRUE, lib='${RLIBS}')"
#RJSONIO
Rscript -e "install.packages('https://cran.r-project.org/src/contrib/RJSONIO_1.3-1.8.tar.gz', dependencies=TRUE, lib='${RLIBS}')"
#Iterators
Rscript -e "install.packages('https://cran.r-project.org/src/contrib/iterators_1.0.14.tar.gz', dependencies=TRUE, lib='${RLIBS}')"
#Itertools
Rscript -e "install.packages('https://cran.r-project.org/src/contrib/itertools_0.1-3.tar.gz', dependencies=TRUE, lib='${RLIBS}')"
#Parallelization
Rscript -e "install.packages('https://cran.r-project.org/src/contrib/foreach_1.5.2.tar.gz', dependencies=TRUE, lib='${RLIBS}')"
Rscript -e "install.packages('https://cran.r-project.org/src/contrib/doParallel_1.0.17.tar.gz', dependencies=TRUE, lib='${RLIBS}')"
Rscript -e "install.packages('https://cran.r-project.org/src/contrib/parallelly_1.36.0.tar.gz', dependencies=TRUE, lib='${RLIBS}')"
#transport
Rscript -e "install.packages('https://cran.r-project.org/src/contrib/data.table_1.14.8.tar.gz', dependencies=TRUE, lib='${RLIBS}')"
Rscript -e "install.packages('https://cran.r-project.org/src/contrib/Archive/RcppEigen/RcppEigen_0.3.3.9.3.tar.gz', dependencies=TRUE, lib='${RLIBS}')"
Rscript -e "install.packages('https://cran.r-project.org/src/contrib/transport_0.14-6.tar.gz', dependencies=TRUE, lib='${RLIBS}')"






