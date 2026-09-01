# decode_production.R
#
# Statistics for the decode/production paragraph (fig:morphology-b), on the
# grounded encode->decode battery items. Mixed-effects model per task:
#   - Model comparison:  outcome ~ model * attested + (1|run) + (1|cx)

# Decode outcome = slots-correct-out-of-2 (binomial, 2 trials). Production
# outcome = exact_match (binary). `attested` = tested construction already in
# the codebook (control) vs novel. cx = construction nested in run. Random
# slopes for `attested` were singular (see README) so random intercepts only;
# `model` is between-run so never enters the random structure. GPT-5.4 is the
# reference level.

suppressMessages({ library(lme4); library(emmeans) })
# Resolve results/csv relative to this script, so it runs from any directory.
.args <- commandArgs(trailingOnly = FALSE)
.self <- sub("^--file=", "", .args[grep("^--file=", .args)])
.here <- if (length(.self)) dirname(normalizePath(.self)) else getwd()
CSV <- file.path(.here, "..", "..", "..", "results", "csv")

d <- read.csv(file.path(CSV, "decode_production_items_full.csv"))
d$model    <- relevel(factor(d$model), ref = "gpt-5.4")
d$attested <- relevel(factor(d$attested), ref = "attested")
d$run <- factor(d$run); d$cx <- factor(d$cx)
ctrl <- glmerControl(optimizer = "bobyqa")
novel <- d[d$attested == "novel" & !is.na(d$native_cap) & d$native_cap >= 2, ]

pair <- function(emm) print(summary(contrast(emm, "pairwise", adjust = "tukey"),
                                     infer = TRUE, type = "link"))

cat("################  DECODE: model comparison  ################\n")
dm <- glmer(cbind(dec_succ, 2 - dec_succ) ~ model * attested + (1|run) + (1|cx),
            family = binomial, data = d, control = ctrl)
print(round(coef(summary(dm)), 4))
cat("\nmarginal decode rate by model (averaged over attested):\n")
emm_dm <- emmeans(dm, ~ model, type = "response"); print(emm_dm)
cat("\npairwise model contrasts (log-odds, Tukey):\n"); pair(emmeans(dm, ~ model))

cat("\n################  PRODUCTION: model comparison (exact_match)  ################\n")
dp <- d[d$exact %in% c(0,1), ]
pm <- glmer(exact ~ model * attested + (1|run) + (1|cx), family = binomial, data = dp, control = ctrl)
print(round(coef(summary(pm)), 4))
cat("\nmarginal production rate by model (averaged over attested):\n")
emm_pm <- emmeans(pm, ~ model, type = "response"); print(emm_pm)
cat("\npairwise model contrasts (log-odds, Tukey):\n"); pair(emmeans(pm, ~ model))
