# emergence_paper_numbers.R
#
# The morphological-emergence model: does a run develop a productive inventory, as a
# function of model, communication budget (log scale) and their interaction?
#
# The paper describes a mixed-effects model with a random intercept for seed. That model is
# fitted here too, and it comes out SINGULAR with a seed variance of 0 -- meaning it is
# numerically identical to the plain logistic GLM. So the GLM is what everything below is
# read off, and the final section prints the two side by side, with the seed variance and
# the singularity flag, so that equivalence is visible rather than asserted.
#
# Source of the reported contrasts: pooled budget slope, and the Tukey table's pairwise
# model comparisons at mean budget.

suppressMessages({ library(lme4); library(emmeans) })

# Resolve results/csv relative to this script, so it runs from any directory.
.args <- commandArgs(trailingOnly = FALSE)
.self <- sub("^--file=", "", .args[grep("^--file=", .args)])
.here <- if (length(.self)) dirname(normalizePath(.self)) else getwd()
CSV <- file.path(.here, "..", "..", "..", "results", "csv")

d <- read.csv(file.path(CSV, "emergence_runs.csv"))
d$logb_c <- log2(d$budget)
d$seed   <- factor(d$seed)
d$model  <- relevel(factor(d$model), ref = "gpt-5.4")

g <- function(f) glm(f, family = binomial, data = d)

full <- g(nonflat ~ logb_c * model)
add  <- g(nonflat ~ logb_c + model)
nob  <- g(nonflat ~ model)
nol  <- g(nonflat ~ logb_c)

cat("=== FULL-model Wald coefficients (GPT-5.4 = reference) ===\n")
print(round(coef(summary(full)), 4))

cat("\n=== budget main effect (pooled slope) ===\n")
cat("pooled slope (additive model): "); print(round(coef(summary(add))["logb_c", ], 4))
r1 <- anova(nob, add, test = "LRT")
cat(sprintf("LRT: chi2(%d) = %.2f, p = %.4f\n",
            r1$Df[2], r1$Deviance[2], r1[2, "Pr(>Chi)"]))

cat("\n=== model main effects ===\n")
r2 <- anova(nol, add, test = "LRT")
cat(sprintf("LRT: chi2(%d) = %.2f, p = %.4f\n",
            r2$Df[2], r2$Deviance[2], r2[2, "Pr(>Chi)"]))

cat("\n=== LRT Test (interaction terms) ===\n")
cat("interaction terms (Opus x budget, Sonnet x budget)\n")
r3 <- anova(add, full, test = "LRT")
cat(sprintf("interaction LRT: chi2(%d) = %.2f, p = %.4f\n",
            r3$Df[2], r3$Deviance[2], r3[2, "Pr(>Chi)"]))

# --- per-budget contrasts vs GPT (levels: gpt-5.4, opus, sonnet) -------------
budgets <- sort(unique(d$budget))
emm <- emmeans(full, ~ model | logb_c, at = list(logb_c = log2(budgets)))
cons <- list("Opus - GPT"           = c(-1, 1, 0),
             "Sonnet - GPT"         = c(-1, 0, 1),
             "Opus - Sonnet"        = c(0, 1, -1),
             "Claude(pooled) - GPT" = c(-1, 0.5, 0.5))

cat("\n=== per-budget contrasts vs GPT (INDIVIDUAL models) ===\n")
for (nm in names(cons)) {
  tab <- as.data.frame(summary(contrast(emm, method = cons[nm])))
  tab$budget <- 2^tab$logb_c
  cat(sprintf("-- %s --\n", nm))
  print(tab[, c("budget", "estimate", "SE", "z.ratio", "p.value")], row.names = FALSE)
}

cat("\n=== Pairwise model contrasts at mean budget (Wald; Tukey-adjusted) ===\n")
print(pairs(emmeans(full, ~ model)))

# --- GLM vs GLMM side-by-side ----------------------------------------------
cat("\n=== GLM vs GLMM (1 | seed) fixed-effect comparison ===\n")
gm <- glmer(nonflat ~ logb_c * model + (1 | seed), family = binomial, data = d,
            control = glmerControl(optimizer = "bobyqa"))
cmp <- data.frame(
  glm_est   = round(coef(summary(full))[, "Estimate"], 4),
  glmm_est  = round(coef(summary(gm))[, "Estimate"], 4),
  glm_se    = round(coef(summary(full))[, "Std. Error"], 4),
  glmm_se   = round(coef(summary(gm))[, "Std. Error"], 4),
  glm_p     = round(coef(summary(full))[, 4], 4),
  glmm_p    = round(coef(summary(gm))[, 4], 4)
)
print(cmp)
cat(sprintf("\nmax |estimate| difference: %.2e | max |SE| difference: %.2e\n",
            max(abs(cmp$glm_est - cmp$glmm_est)),
            max(abs(cmp$glm_se  - cmp$glmm_se))))
cat("seed variance in GLMM:", format(as.data.frame(VarCorr(gm))$vcov[1], digits = 3),
    "| singular:", isSingular(gm), "\n")
cat(sprintf("AIC: GLM = %.2f | GLMM = %.2f\n", AIC(full), AIC(gm)))
