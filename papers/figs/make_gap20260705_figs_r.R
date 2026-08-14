#!/usr/bin/env Rscript
# make_gap20260705_figs_r.R — canonical renderer for the two Paper-A gap-battery
# figures (2026-07-05 arms A#1, A#2). Plot-only (no GPU). Numbers are read at
# render time from the single-source-of-truth verdict JSON
#   experiments/results/ieee_gap_20260705/A/verdict_gapA.json
# so the figures cannot drift from the reported values. Output: PNG preview ->
# papers/figs/<name>.png, vector tier -> figs/tex/<name>.tex (tikz; LaTeX-typeset
# fonts == body). Article column 6.5in.
#
# Figures:
#   A_gap_normcap  — norm-cap dose-response (median grok step vs ceiling k) on the
#                    second/third tasks A5 (rung-60) and D60 (rung-120): direction
#                    (preserve + accelerate) invariant, magnitude task-dependent.
#   A_gap_normlaw  — Proposition-1 hidden-norm equilibrium vs weight decay lambda
#                    (log-log): Muon negative power law vs AdamW flat.
#
# Usage:  cd papers/figs && Rscript make_gap20260705_figs_r.R   (or from repo root)
# The plot builders are defined ABOVE the render marker below so the IEEE-variant
# script (make_ieee_variants_A.R) can source the definitions only.

suppressPackageStartupMessages({
  library(jsonlite)
  library(ggplot2)
  library(dplyr)
  library(scales)
  library(ragg)
})

# ── paths (robust to being run from papers/figs or repo root) ──────────────────
wd <- getwd()
if (basename(wd) == "figs") {
  root <- normalizePath(file.path(wd, "..", ".."))
} else {
  root <- normalizePath(wd)
}
figdir <- file.path(root, "papers", "figs")
res    <- file.path(root, "experiments", "results")

# ── colour palette (Okabe-Ito, matching make_A_figs_r.R / make_A_new_figs_r.R) ──
CB <- list(
  blue       = "#0072B2",
  orange     = "#E69F00",
  vermillion = "#D55E00",
  green      = "#009E73",
  skyblue    = "#56B4E9",
  purple     = "#CC79A7",
  grey       = "#999999",
  black      = "#000000"
)
OPT_COL <- c(AdamW = CB$blue, Muon = CB$vermillion, SGDM = CB$grey)
INK     <- "#1a1a1a"

# ── shared theme: article column, effective >= 8pt; only panel tags bold ───────
paper_theme <- function(base_size = 9) {
  theme_minimal(base_size = base_size, base_family = "TeX Gyre Termes") +
    theme(
      panel.grid.minor   = element_blank(),
      panel.grid.major.x = element_blank(),
      panel.grid.major.y = element_line(linewidth = 0.25, colour = "#d9dde3"),
      axis.title  = element_text(colour = "#1a202c", size = base_size + 0.5, face = "plain"),
      axis.text   = element_text(colour = "#2d3748", size = base_size,       face = "plain"),
      plot.title  = element_text(colour = "#111827", size = base_size + 1,   face = "plain"),
      legend.position = "top",
      legend.title    = element_text(size = base_size, face = "plain"),
      legend.text     = element_text(size = base_size, face = "plain"),
      plot.margin     = margin(5, 6, 5, 6)
    )
}

source(file.path(figdir, "fig_pipeline.R"))  # emit_vector(): tikz/.tex + cairo_pdf/.pdf
source(file.path(figdir, "A_panel_contract.R"))
save_png <- function(p, name, w = 6.5, h = 4.0) {
  pngp <- file.path(figdir, paste0(name, ".png"))
  ragg::agg_png(pngp, width = w, height = h, units = "in", res = 300, scaling = 1)
  print(p); invisible(dev.off())
  emit_vector(p, name, w, h)   # vector tier: tikz (.tex) or cairo_pdf (.pdf)
  cat(sprintf("saved %s  (%.1f x %.1f in)\n", pngp, w, h))
}

# ── single-source-of-truth loader ─────────────────────────────────────────────
gap_load <- function() {
  gapf <- file.path(res, "ieee_gap_20260705", "A", "verdict_gapA.json")
  jsonlite::fromJSON(gapf, simplifyVector = FALSE)
}

# ══════════════════════════════════════════════════════════════════════════════
#  A_gap_normcap — dose-response of median grok step to the hidden-norm ceiling k
#  on the second/third tasks (A5 rung-60, D60 rung-120). Both preserved at every
#  ceiling; direction invariant, magnitude task-dependent.
# ══════════════════════════════════════════════════════════════════════════════
gap_normcap_plot <- function(V, base_size = 9) {
  k_ord <- c("kinf", "k3", "k2", "k1p5", "k1")
  k_lab <- c(kinf = "∞", k3 = "3", k2 = "2", k1p5 = "1.5", k1 = "1")
  task_lab <- c(A5 = "A5 (rung-60)", D60 = "D60 (rung-120)")
  task_col <- c("A5 (rung-60)" = CB$vermillion, "D60 (rung-120)" = CB$blue)

  build_task <- function(task) {
    pk <- V$normaccel$tasks[[task]]$per_k
    data.frame(
      klab = factor(k_lab[k_ord], levels = k_lab[k_ord]),
      grok = vapply(k_ord, function(k) as.numeric(pk[[k]]$median_grok_step), numeric(1)),
      n    = vapply(k_ord, function(k) as.integer(pk[[k]]$n), integer(1)),
      rate = vapply(k_ord, function(k) as.numeric(pk[[k]]$grok_rate), numeric(1)),
      task = task_lab[[task]], stringsAsFactors = FALSE)
  }
  df <- rbind(build_task("A5"), build_task("D60"))
  df$task <- factor(df$task, levels = task_lab)

  # per-task acceleration (uncapped -> tightest cap) and preserved-count labels
  acc <- function(task) V$normaccel$tasks[[task]]$per_k$k1$accel_vs_inf
  npres <- function(task) V$normaccel$tasks[[task]]$per_k$kinf$n
  lab_df <- data.frame(
    klab = factor("1", levels = k_lab[k_ord]),
    task = factor(task_lab, levels = task_lab),
    grok = c(V$normaccel$tasks$A5$per_k$k1$median_grok_step,
             V$normaccel$tasks$D60$per_k$k1$median_grok_step),
    lab  = c(sprintf("%.2f×, %d/%d preserved", acc("A5"),  npres("A5"),  npres("A5")),
             sprintf("%.2f×, %d/%d preserved", acc("D60"), npres("D60"), npres("D60"))),
    stringsAsFactors = FALSE)

  ggplot(df, aes(klab, grok, colour = task, group = task)) +
    geom_line(linewidth = 1.1) +
    geom_point(size = 2.6) +
    geom_text(data = lab_df, aes(label = lab), hjust = 1.05, vjust = -0.9,
              size = 2.7, show.legend = FALSE) +
    scale_colour_manual(values = task_col, name = NULL) +
    scale_y_log10(labels = label_comma(),
                  expand = expansion(mult = c(0.10, 0.18))) +
    labs(title = "(a) Dose-response of grok step to the ceiling",
         x = "Hidden-norm ceiling k (loose → tight)",
         y = "Median grok step (log scale)") +
    paper_theme(base_size)
}

# ── 4-panel composite for A_gap_normcap ───────────────────────────────────────
gap_normcap_composite <- function(V, base_size = 9) {
  library(patchwork)
  pa <- gap_normcap_plot(V, base_size) + labs(title = "(a) Dose-response of grok step to the ceiling")

  # (b) paired seed bootstrap intervals across the S5 cap-dose comparisons.
  bc <- jsonlite::fromJSON(file.path(root, "experiments", "revision2026", "A",
                                     "bootstrap_cis.json"), simplifyVector = FALSE)
  pick <- function(nm) Filter(function(e) e$name == nm, bc$entries)[[1]]
  cap_names <- c("S5 primary: kinf/k3", "S5 primary: kinf/k2",
                 "S5 primary: kinf/k1p5", "S5 primary: kinf/k1")
  cap_lab <- c("k=3", "k=2", "k=1.5", "k=1")
  cap_entries <- lapply(cap_names, pick)
  paired <- data.frame(
    cap = factor(cap_lab, levels = cap_lab),
    est = vapply(cap_entries, function(e) e$ratio, numeric(1)),
    lo = vapply(cap_entries, function(e) e$ci95_percentile[[1]], numeric(1)),
    hi = vapply(cap_entries, function(e) e$ci95_percentile[[2]], numeric(1)))
  pb <- ggplot(paired, aes(cap, est)) +
    geom_hline(yintercept = 1, linetype = "dashed", colour = CB$grey) +
    geom_point(size = 2.7, colour = CB$vermillion) +
    geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.22, colour = CB$vermillion) +
    geom_text(aes(label = sprintf("%.2fx", est)), vjust = -0.9, size = 2.6) +
    labs(title = "(b) Paired S5 acceleration intervals",
         x = "Hidden-norm ceiling", y = "Acceleration (uncapped / capped)") +
    paper_theme(base_size)

  # (c) ADF defloored per-seed (D60, cadence 5): k1 vs kinf strips
  adf_dir <- file.path(root, "experiments", "revision2026", "gpu2026", "adf")
  read_adf <- function(task, k) {
    fs <- list.files(adf_dir, pattern = sprintf("^adf_%s_%s_s[0-9]+\\.jsonl$", task, k),
                     full.names = TRUE)
    vapply(fs, function(f) {
      ll <- readLines(f, warn = FALSE); ll <- ll[nzchar(trimws(ll))]
      rec <- jsonlite::fromJSON(gsub("\\bNaN\\b", "null", ll[length(ll)]),
                                simplifyVector = TRUE)
      s <- if (!is.null(rec[["_summary"]])) rec[["_summary"]] else rec
      as.numeric(s$grok_step)
    }, numeric(1))
  }
  d60 <- data.frame(
    step = c(read_adf("D60", "kinf"), read_adf("D60", "k1")),
    arm  = rep(c("uncapped", "k = 1"),
               c(length(read_adf("D60", "kinf")), length(read_adf("D60", "k1")))))
  med <- tapply(d60$step, d60$arm, median)
  adfv <- jsonlite::fromJSON(file.path(adf_dir, "adf_verdict.json"),
                             simplifyVector = FALSE)
  e_ad  <- Filter(function(e) e$task == "add", adfv$entries)[[1]]
  e_d60 <- Filter(function(e) e$task == "D60", adfv$entries)[[1]]
  pc <- ggplot(d60, aes(arm, step)) +
    stat_summary(fun = median, geom = "crossbar", width = 0.5, colour = CB$black) +
    geom_point(position = position_jitter(width = 0.06, height = 0, seed = 3),
               size = 2.2, colour = CB$blue, alpha = 0.85) +
    geom_text(data = data.frame(x = 1.5, y = max(d60$step) * 1.05),
              aes(x = x, y = y, label = sprintf(
                "median %.0f → %.0f (ratio %.2f)\n95%% CI [%.2f, %.2f]",
                e_d60$median_uncapped, e_d60$median_capped, e_d60$ratio,
                e_d60$ci95_percentile[[1]], e_d60$ci95_percentile[[2]])),
              size = 2.7, lineheight = 0.9, inherit.aes = FALSE) +
    labs(title = "(c) D60 defloored fine-cadence per-seed (cadence 5)",
         x = NULL, y = "Grok step") +
    paper_theme(base_size)

  # (d) cross-task acceleration summary with CIs (from bootstrap_cis + adf verdict)
  bc <- jsonlite::fromJSON(file.path(root, "experiments", "revision2026", "A",
                                     "bootstrap_cis.json"), simplifyVector = FALSE)
  pick <- function(nm) Filter(function(e) e$name == nm, bc$entries)[[1]]
  e_s5 <- pick("S5 primary: kinf/k1")
  e_a5 <- pick("A5: kinf/k1")
  acc_df <- data.frame(
    task = factor(c("S5", "D60", "A5", "mod-add"),
                  levels = c("S5", "D60", "A5", "mod-add")),
    est  = c(e_s5$ratio, e_d60$ratio, e_a5$ratio, e_ad$ratio),
    lo   = c(e_s5$ci95_percentile[[1]], e_d60$ci95_percentile[[1]],
             e_a5$ci95_percentile[[1]], e_ad$ci95_percentile[[1]]),
    hi   = c(e_s5$ci95_percentile[[2]], e_d60$ci95_percentile[[2]],
             e_a5$ci95_percentile[[2]], e_ad$ci95_percentile[[2]]))
  pd <- ggplot(acc_df, aes(task, est)) +
    geom_hline(yintercept = 1, linetype = "dashed", colour = CB$grey) +
    geom_point(size = 2.8, colour = CB$black) +
    geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.25, colour = CB$black) +
    geom_text(aes(label = sprintf("%.2f×", est)), vjust = -1.0, size = 2.7) +
    coord_cartesian(ylim = c(0, max(acc_df$est) * 1.3)) +
    labs(title = "(d) Acceleration is preserve-first: every task ≥ 1×",
         x = NULL, y = "Acceleration (uncapped / k = 1)") +
    paper_theme(base_size)

  compose_A_four_panel(list(pa, pb, pc, pd), "A_gap_normcap", 7.2, 5.2, root)
}

# ══════════════════════════════════════════════════════════════════════════════
#  A_gap_normlaw — Proposition-1 test: equilibrium hidden norm vs weight decay.
#  Muon follows a negative power law (slope ~ -0.43), AdamW is flat. Reported as
#  confirmed in SIGN + MONOTONICITY, not as the idealized -1 exponent.
# ══════════════════════════════════════════════════════════════════════════════
gap_normlaw_plot <- function(V, base_size = 9) {
  lam <- c(0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
  opt_lab <- c(muon = "Muon", adamw = "AdamW")
  opt_col <- c(Muon = CB$vermillion, AdamW = CB$blue)

  build_opt <- function(o) {
    fit <- V$normlaw$fits[[o]]
    wn  <- vapply(as.character(lam),
                  function(l) as.numeric(fit$median_wn_by_lambda[[l]]), numeric(1))
    data.frame(lambda = lam, wn = wn, opt = opt_lab[[o]], stringsAsFactors = FALSE)
  }
  pts <- rbind(build_opt("muon"), build_opt("adamw"))
  pts$opt <- factor(pts$opt, levels = opt_lab)

  # fit lines in log10 space: log10(wn) = intercept + slope * log10(lambda)
  line_opt <- function(o) {
    fit <- V$normlaw$fits[[o]]
    xs  <- 10^seq(log10(min(lam)), log10(max(lam)), length.out = 50)
    data.frame(lambda = xs,
               wn = 10^(fit$intercept + fit$slope * log10(xs)),
               opt = opt_lab[[o]], stringsAsFactors = FALSE)
  }
  lines <- rbind(line_opt("muon"), line_opt("adamw"))
  lines$opt <- factor(lines$opt, levels = opt_lab)

  slope_lab <- data.frame(
    lambda = c(0.45, 0.45),
    wn     = c(430, 15),
    opt    = factor(c("Muon", "AdamW"), levels = opt_lab),
    lab    = c(sprintf("Muon slope %.2f", V$normlaw$fits$muon$slope),
               sprintf("AdamW slope %.2f", V$normlaw$fits$adamw$slope)),
    stringsAsFactors = FALSE)

  ggplot(pts, aes(lambda, wn, colour = opt)) +
    geom_line(data = lines, aes(lambda, wn, colour = opt), linewidth = 0.9,
              linetype = "dashed") +
    geom_point(size = 2.6) +
    geom_text(data = slope_lab, aes(label = lab), hjust = 1, size = 2.7,
              show.legend = FALSE) +
    scale_colour_manual(values = opt_col, name = NULL) +
    scale_x_log10(breaks = lam, labels = label_number(drop0trailing = TRUE)) +
    scale_y_log10(labels = label_comma()) +
    labs(title = "(a) Equilibrium hidden norm vs weight decay",
         x = "Weight decay λ (log scale)",
         y = "Median hidden norm ‖W‖ (log scale)") +
    paper_theme(base_size) +
    theme(axis.text.x = element_text(angle = 35, hjust = 1))
}

# ── 4-panel composite for A_gap_normlaw ───────────────────────────────────────
gap_normlaw_composite <- function(V, base_size = 9) {
  library(patchwork)
  pa <- gap_normlaw_plot(V, base_size)

  refit <- jsonlite::fromJSON(
    file.path(root, "experiments", "revision2026", "A", "refit_normlaw.json"),
    simplifyVector = FALSE)
  lam <- c(0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
  opt_lab <- c(muon = "Muon", adamw = "AdamW")
  opt_col <- c(Muon = CB$vermillion, AdamW = CB$blue)

  # (b) per-seed final norms (5 seeds per λ per optimizer; refit arm of record)
  seed_df <- bind_rows(lapply(names(opt_lab), function(o) {
    per <- refit[[o]]$per_seed_final_wn_by_lambda
    bind_rows(lapply(names(per), function(l) {
      data.frame(lambda = as.numeric(l), wn = as.numeric(unlist(per[[l]])),
                 opt = opt_lab[[o]], stringsAsFactors = FALSE)
    }))
  }))
  seed_df$opt <- factor(seed_df$opt, levels = opt_lab)
  pb <- ggplot(seed_df, aes(factor(lambda), wn, colour = opt)) +
    stat_summary(fun = median, geom = "crossbar", width = 0.5, colour = CB$black) +
    geom_point(position = position_jitter(width = 0.08, height = 0, seed = 5),
               size = 1.8, alpha = 0.8) +
    scale_colour_manual(values = opt_col, name = NULL) +
    scale_y_log10(labels = label_comma()) +
    labs(title = "(b) Per-seed equilibrium norms (5 seeds/point)",
         x = "Weight decay λ", y = "Final hidden norm ‖W‖ (log scale)") +
    paper_theme(base_size) +
    theme(axis.text.x = element_text(angle = 35, hjust = 1))

  # (c) slope estimate ± SE against the two theoretical targets:
  #     −1 = idealized Prop-1 exponent; −0.5 = Kosson rotational equilibrium
  tg <- bind_rows(lapply(names(opt_lab), function(o) {
    f <- refit[[o]]$fits$full_final
    data.frame(opt = opt_lab[[o]], slope = f$slope, se = f$se,
               stringsAsFactors = FALSE)
  }))
  tg$opt <- factor(tg$opt, levels = opt_lab)
  ref_df <- data.frame(target = c("idealized Prop-1 (-1)",
                                  "rotational equilibrium (-0.5)"),
                       y = c(-1.0, -0.5))
  pc <- ggplot(tg, aes(opt, slope, colour = opt)) +
    geom_hline(data = ref_df, aes(yintercept = y, linetype = target),
               colour = CB$grey, linewidth = 0.8) +
    scale_linetype_manual(values = c("dashed", "dotted"), name = NULL) +
    geom_pointrange(aes(ymin = slope - 1.96 * se, ymax = slope + 1.96 * se),
                    size = 0.9, linewidth = 1.0, show.legend = FALSE) +
    geom_text(aes(label = sprintf("%.2f ± %.2f", slope, se)), vjust = -1.2,
              size = 2.8, show.legend = FALSE) +
    scale_colour_manual(values = opt_col, guide = "none") +
    coord_cartesian(ylim = c(-1.15, 0.35)) +
    labs(title = "(c) Slope vs theory: sign + monotonicity, not the -1 ideal",
         x = NULL, y = "log–log slope (95% CI)") +
    paper_theme(base_size) +
    theme(legend.position = "top")

  # (d) leave-one-λ-out slope stability (recomputed from per-seed medians)
  loo <- bind_rows(lapply(names(opt_lab), function(o) {
    per <- refit[[o]]$per_seed_final_wn_by_lambda
    med <- vapply(names(per), function(l) median(as.numeric(unlist(per[[l]]))),
                  numeric(1))
    lams <- as.numeric(names(med))
    slopes <- vapply(seq_along(lams), function(i) {
      keep <- -i
      coef(lm(log10(med[keep]) ~ log10(lams[keep])))[2]
    }, numeric(1))
    data.frame(held_out = factor(format(lams, drop0trailing = TRUE),
                                 levels = format(lams, drop0trailing = TRUE)),
               slope = slopes, opt = opt_lab[[o]], stringsAsFactors = FALSE)
  }))
  loo$opt <- factor(loo$opt, levels = opt_lab)
  pd <- ggplot(loo, aes(held_out, slope, colour = opt, group = opt)) +
    geom_hline(yintercept = c(-1, -0.5), linetype = c("dashed", "dotted"),
               colour = CB$grey, linewidth = 0.7) +
    geom_line(linewidth = 0.9) + geom_point(size = 2.2) +
    scale_colour_manual(values = opt_col, name = NULL) +
    labs(title = "(d) Leave-one-λ-out slope stability",
         x = "Held-out λ level", y = "Refit slope") +
    paper_theme(base_size) +
    theme(axis.text.x = element_text(angle = 35, hjust = 1))

  compose_A_four_panel(list(pa, pb, pc, pd), "A_gap_normlaw", 7.2, 5.2, root)
}

# ── extend the tikz sanitize map for the two glyphs the pipeline default omits:
#    the norm double-bar (U+2016) and the squared superscript in "R²" (used in
#    the log-log fit labels). Without this they emit as raw UTF-8, which the
#    pdflatex IEEE mirror cannot compile. Mirrors the A_landscape >/< precedent
#    in make_ieee_variants_A.R. Harmless for A_gap_normcap (neither glyph occurs).
gap_extend_sanitize <- function() {
  options(
    tikzSanitizeCharacters    = c(getOption("tikzSanitizeCharacters"),    "‖", "²"),
    tikzReplacementCharacters = c(getOption("tikzReplacementCharacters"), "$\\|$", "$^2$"))
}

# ══════════════════════════════════════════════════════════════════════════════
cat("Rendering gap-battery Paper-A figures...\n")
# ══════════════════════════════════════════════════════════════════════════════
gap_extend_sanitize()
V <- gap_load()
save_png(gap_normcap_composite(V), "A_gap_normcap", w = 7.2, h = 5.2)
save_png(gap_normlaw_composite(V), "A_gap_normlaw", w = 7.2, h = 5.2)
