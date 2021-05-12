#' Calculate regression credible interval overlap
#'
#' @param Ucomp Upper bound of the complete data
#' @param Umiss Upper bound of the missing data
#' @param Lcomp Lower bound of the complete data
#' @param Lmiss Lower bound of the missing data
#'
#' @return
#'
interval_overlap <- function(Ucomp, Umiss, Lcomp, Lmiss) {
  
  Jk <- 0.5 *
    ((pmin(Ucomp, Umiss) - pmax(Lcomp, Lmiss)) / (Ucomp - Lcomp) +
       (pmin(Ucomp, Umiss) - pmax(Lcomp, Lmiss)) / (Umiss - Lmiss))
  
  return(Jk)
  
}