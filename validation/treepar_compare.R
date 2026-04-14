#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

usage <- function(status = 1) {
  cat(
    "Usage:\n",
    "  treepar_compare.R p0 <t> <lambda> <mu> <rho>\n",
    "  treepar_compare.R p1 <t> <lambda> <mu> <rho>\n",
    "  treepar_compare.R LikConstant <lambda> <mu> <sampling> <x_csv> <root> <survival>\n",
    "  treepar_compare.R LikShiftsSTT <lambda> <mu> <times_csv> <ttype_csv> <sampling> <sprob> <root> <survival>\n",
    sep = "",
    file = stderr()
  )
  quit(save = "no", status = status)
}

fail <- function(...) {
  cat(sprintf(...), "\n", sep = "", file = stderr())
  usage(1)
}

require_nargs <- function(n, mode) {
  if (length(args) != n) {
    fail("Mode '%s' expects %d arguments including the mode; got %d.", mode, n, length(args))
  }
}

parse_number <- function(x, name) {
  val <- suppressWarnings(as.numeric(x))
  if (length(val) != 1 || is.na(val) || !is.finite(val)) {
    fail("Argument '%s' must be a finite number; got '%s'.", name, x)
  }
  val
}

parse_integer <- function(x, name) {
  val <- suppressWarnings(as.integer(x))
  if (length(val) != 1 || is.na(val)) {
    fail("Argument '%s' must be an integer; got '%s'.", name, x)
  }
  val
}

parse_num_vec <- function(x, name) {
  if (identical(x, "") || is.na(x)) return(numeric())
  vals <- suppressWarnings(as.numeric(strsplit(x, ",", fixed = TRUE)[[1]]))
  if (any(is.na(vals)) || any(!is.finite(vals))) {
    fail("Argument '%s' must be a comma-separated list of finite numbers; got '%s'.", name, x)
  }
  vals
}

parse_int_vec <- function(x, name) {
  if (identical(x, "") || is.na(x)) return(integer())
  vals <- suppressWarnings(as.integer(strsplit(x, ",", fixed = TRUE)[[1]]))
  if (any(is.na(vals))) {
    fail("Argument '%s' must be a comma-separated list of integers; got '%s'.", name, x)
  }
  vals
}

if (length(args) < 1 || args[[1]] %in% c("-h", "--help", "help")) {
  usage(if (length(args) < 1) 1 else 0)
}

ensure_treepar <- function() {
  if (!requireNamespace("TreePar", quietly = TRUE)) {
    cat("TreePar is not installed in this R environment. Install it before running external validation.\n",
        file = stderr())
    quit(save = "no", status = 2)
  }
}

treepar_fun <- function(name) {
  ensure_treepar()
  ns <- asNamespace("TreePar")
  if (!exists(name, envir = ns, mode = "function", inherits = FALSE)) {
    cat(sprintf("TreePar function '%s' was not found in the installed TreePar namespace.\n", name),
        file = stderr())
    quit(save = "no", status = 2)
  }
  get(name, envir = ns, mode = "function", inherits = FALSE)
}

mode <- args[[1]]

if (mode == "p0") {
  require_nargs(5, mode)
  t <- parse_number(args[[2]], "t")
  l <- parse_number(args[[3]], "lambda")
  m <- parse_number(args[[4]], "mu")
  rho <- parse_number(args[[5]], "rho")
  val <- treepar_fun("p0")(t, l, m, rho)
  cat(sprintf("%.17g\n", val))
  quit(save = "no")
}

if (mode == "p1") {
  require_nargs(5, mode)
  t <- parse_number(args[[2]], "t")
  l <- parse_number(args[[3]], "lambda")
  m <- parse_number(args[[4]], "mu")
  rho <- parse_number(args[[5]], "rho")
  val <- treepar_fun("p1")(t, l, m, rho)
  cat(sprintf("%.17g\n", val))
  quit(save = "no")
}

if (mode == "LikConstant") {
  require_nargs(7, mode)
  lambda <- parse_number(args[[2]], "lambda")
  mu <- parse_number(args[[3]], "mu")
  sampling <- parse_number(args[[4]], "sampling")
  x <- parse_num_vec(args[[5]], "x")
  root <- parse_integer(args[[6]], "root")
  survival <- parse_integer(args[[7]], "survival")

  ensure_treepar()
  val <- TreePar::LikConstant(
    lambda = lambda,
    mu = mu,
    sampling = sampling,
    x = x,
    root = root,
    survival = survival
  )
  cat(sprintf("%.17g\n", val))
  quit(save = "no")
}

if (mode == "LikShiftsSTT") {
  require_nargs(9, mode)
  lambda <- parse_number(args[[2]], "lambda")
  mu <- parse_number(args[[3]], "mu")
  times <- parse_num_vec(args[[4]], "times")
  ttype <- parse_int_vec(args[[5]], "ttype")
  sampling <- parse_number(args[[6]], "sampling")
  sprob <- parse_number(args[[7]], "sprob")
  root <- parse_integer(args[[8]], "root")
  survival <- parse_integer(args[[9]], "survival")

  ensure_treepar()
  val <- TreePar::LikShiftsSTT(
    par = c(lambda, mu),
    times = times,
    ttype = ttype,
    numbd = 1,
    tconst = -1,
    sampling = sampling,
    sprob = sprob,
    root = root,
    survival = survival
  )
  cat(sprintf("%.17g\n", val))
  quit(save = "no")
}

fail("Unknown mode: %s", mode)
