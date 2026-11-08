# Both dependencies ship with the pinned R distribution.
for (package in c("stats", "utils")) {
  if (!requireNamespace(package, quietly = TRUE)) {
    stop(sprintf("The R installation is missing the bundled package '%s'.", package))
  }
  cat(sprintf("Using %s %s\n", package, utils::packageVersion(package)))
}
