args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 2L) stop("Usage: Rscript model.R [data_dir] [output_dir]")
data_dir <- if (length(args) >= 1L) args[1L] else "/data"
output_dir <- if (length(args) >= 2L) args[2L] else "/output"
features <- c("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare")

read_passengers <- function(filename, required) {
  path <- file.path(data_dir, filename)
  if (!file.exists(path)) stop("Missing file: ", path)
  df <- utils::read.csv(path, colClasses = "character", na.strings = c("", "NA"),
                        strip.white = TRUE, check.names = FALSE)
  missing <- setdiff(c("PassengerId", required), names(df))
  if (length(missing)) stop(filename, " lacks columns: ", paste(missing, collapse = ", "))
  if (!nrow(df)) stop(filename, " is empty.")
  df$PassengerId <- trimws(df$PassengerId)
  if (anyNA(df$PassengerId) || any(df$PassengerId == "") || anyDuplicated(df$PassengerId)) {
    stop(filename, " requires unique, nonempty PassengerId values.")
  }
  df
}

binary_labels <- function(values, source) {
  if (!all(values %in% c("0", "1"))) stop(source, " requires a 0 or 1 for every Survived value.")
  as.integer(values)
}

encode_features <- function(df) {
  x <- df[features]
  sex <- tolower(trimws(x$Sex))
  if (any(!is.na(sex) & !sex %in% c("female", "male"))) stop("Unknown value in Sex.")
  x$Sex <- match(sex, c("male", "female")) - 1L
  for (name in setdiff(features, "Sex")) {
    values <- suppressWarnings(as.numeric(x[[name]]))
    if (any(!is.na(x[[name]]) & !is.finite(values))) stop("Invalid numeric value in ", name)
    x[[name]] <- values
  }
  x
}

fit_model <- function(x, y) {
  if (any(colSums(!is.na(x)) == 0L)) stop("A training feature has no observed values.")
  fill <- vapply(x, stats::median, numeric(1), na.rm = TRUE)
  # For binary Sex, replace missing entries with the most frequent training value.
  fill["Sex"] <- as.integer(mean(x$Sex, na.rm = TRUE) > 0.5)
  x <- impute(x, fill)
  model <- stats::glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare,
                      data = data.frame(Survived = y, x), family = stats::binomial())
  list(model = model, fill = fill)
}

impute <- function(x, fill) {
  for (name in names(fill)) x[[name]][is.na(x[[name]])] <- fill[[name]]
  x
}

predict_class <- function(fit, x) {
  probability <- stats::predict(fit$model, newdata = impute(x, fit$fill), type = "response")
  as.integer(probability >= 0.5)
}

report_accuracy <- function(name, actual, predicted) {
  correct <- sum(actual == predicted)
  cat(sprintf("%s accuracy: %.2f%% (%d/%d)\n", name, 100 * correct / length(actual),
              correct, length(actual)))
}

train <- read_passengers("train.csv", c(features, "Survived"))
test <- read_passengers("test.csv", features)
y <- binary_labels(train$Survived, "train.csv")
x <- encode_features(train)
test_x <- encode_features(test)
cat(sprintf("Loaded %d training passengers and %d test passengers.\n", nrow(train), nrow(test)))
cat(sprintf("Training survivors: %d/%d (%.1f%%)\n", sum(y), length(y), 100 * mean(y)))
for (group in sort(unique(stats::na.omit(train$Pclass)))) {
  subset <- y[which(train$Pclass == group)]
  cat(sprintf("  Class %s survival: %d/%d (%.1f%%)\n", group, sum(subset),
              length(subset), 100 * mean(subset)))
}
cat("Predictors: ordinal Pclass, Sex (female=1, male=0), Age, SibSp, Parch, Fare.\n")
cat("Missing predictor counts (before imputation):\n")
print(rbind(train = colSums(is.na(x)), test = colSums(is.na(test_x))))
cat("Missing numeric values use training medians; Sex uses the training mode.\n")

if (any(tabulate(y + 1L, nbins = 2L) < 2L)) {
  stop("The stratified split needs at least two passengers in each survival class.")
}
set.seed(42)
holdout <- unlist(lapply(split(seq_along(y), y), function(rows) {
  count <- max(1L, min(length(rows) - 1L, round(length(rows) * 0.20)))
  sample(rows, count)
}), use.names = FALSE)
cat(sprintf("Stratified split (seed 42): %d fit, %d held out.\n",
            nrow(x) - length(holdout), length(holdout)))
cat("Fitting logistic regression; survival is predicted at probability >= 0.5.\n")
fold <- fit_model(x[-holdout, ], y[-holdout])
report_accuracy("Holdout validation", y[holdout], predict_class(fold, x[holdout, ]))

fit <- fit_model(x, y)
cat("Refitted on all training rows. Final imputation values:\n")
print(fit$fill)
report_accuracy("Full training (in-sample)", y, predict_class(fit, x))
predictions <- predict_class(fit, test_x)
cat(sprintf("Test predictions: %d survived, %d did not.\n", sum(predictions),
            length(predictions) - sum(predictions)))

if (file.exists(file.path(data_dir, "test_labels.csv"))) {
  labels <- read_passengers("test_labels.csv", "Survived")
  if (!setequal(labels$PassengerId, test$PassengerId)) {
    stop("test_labels.csv must cover exactly the PassengerId values in test.csv.")
  }
  actual <- binary_labels(labels$Survived, "test_labels.csv")[match(test$PassengerId, labels$PassengerId)]
  report_accuracy("Test (test_labels.csv)", actual, predictions)
} else if ("Survived" %in% names(test)) {
  report_accuracy("Test (test.csv)", binary_labels(test$Survived, "test.csv"), predictions)
} else {
  cat("Test accuracy unavailable: Kaggle test.csv omits Survived. Submit the predictions\n",
      "to Kaggle for a leaderboard score, or supply genuine labels in test_labels.csv.\n", sep = "")
}

if (!dir.exists(output_dir) && !dir.create(output_dir, recursive = TRUE)) {
  stop("Could not create the output directory: ", output_dir)
}
path <- file.path(output_dir, "r_predictions.csv")
utils::write.csv(data.frame(PassengerId = test$PassengerId, Survived = predictions),
                 path, row.names = FALSE, quote = FALSE)
cat(sprintf("Wrote %d rows to %s, in test.csv order.\n", length(predictions), path))
