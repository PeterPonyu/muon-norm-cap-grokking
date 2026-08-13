#!/usr/bin/env Rscript
# make_A_new_figs_r.R — Standalone renderer for NEW Paper-A figures built from
# already-measured-but-previously-unused data. Plot-only (no GPU).
# All plotted values are computed from the raw per-run logs in
# experiments/results/ at render time (numbers red line).
# Output: PNG -> papers/figs/<name>.png (300 dpi via ragg), article column 6.5in.
# Usage:  cd papers/figs && Rscript make_A_new_figs_r.R   (or from repo root)

suppressPackageStartupMessages({
  library(jsonlite)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(scales)
  library(patchwork)
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

# ── colour palette (Okabe-Ito colourblind-safe, matching make_A_figs_r.R) ──────
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
      strip.text      = element_text(colour = "#1a202c", size = base_size, face = "plain"),
      plot.margin     = margin(5, 6, 5, 6),
      plot.tag        = element_text(face = "bold", size = 11, family = "TeX Gyre Termes"),
      plot.tag.position = "topleft"
    )
}
tag_ann <- function() plot_annotation(tag_levels = "a", tag_prefix = "(", tag_suffix = ")")

source(file.path(figdir, "fig_pipeline.R"))  # emit_vector(): tikz/.tex + cairo_pdf/.pdf
source(file.path(figdir, "A_panel_contract.R"))
save_png <- function(p, name, w = 6.5, h = 4.0) {
  pngp <- file.path(figdir, paste0(name, ".png"))
  ragg::agg_png(pngp, width = w, height = h, units = "in", res = 300, scaling = 1)
  print(p); invisible(dev.off())
  emit_vector(p, name, w, h)   # vector tier: tikz (.tex) or cairo_pdf (.pdf)
  cat(sprintf("saved %s  (%.1f x %.1f in)\n", pngp, w, h))
}

# ── JSON helpers ──────────────────────────────────────────────────────────────
last_summary <- function(path) {
  lines <- readLines(path, warn = FALSE)
  lines <- lines[nzchar(trimws(lines))]
  for (i in rev(seq_along(lines))) {
    txt <- gsub("\\bNaN\\b", "null", lines[i])
    txt <- gsub("\\bInfinity\\b", "null", txt); txt <- gsub("\\b-Infinity\\b", "null", txt)
    rec <- tryCatch(jsonlite::fromJSON(txt, simplifyVector = TRUE), error = function(e) NULL)
    if (!is.null(rec) && !is.null(rec[["_summary"]])) return(rec[["_summary"]])
  }
  NULL
}
step_records <- function(path) {
  lines <- readLines(path, warn = FALSE)
  lines <- lines[nzchar(trimws(lines))]
  out <- list()
  for (ln in lines) {
    txt <- gsub("\\bNaN\\b", "null", ln)
    txt <- gsub("\\bInfinity\\b", "null", txt); txt <- gsub("\\b-Infinity\\b", "null", txt)
    rec <- tryCatch(jsonlite::fromJSON(txt, simplifyVector = TRUE), error = function(e) NULL)
    if (!is.null(rec) && !is.null(rec[["step"]])) out[[length(out) + 1]] <- rec
  }
  out
}
med <- function(x) stats::median(x, na.rm = TRUE)

# ══════════════════════════════════════════════════════════════════════════════
#  NEW-1 (A1) — Norm-cap time course: hidden norm + test acc vs step,
#  uncapped vs tightest cap. Median ± IQR over 8 seeds; grok step marked.
#  Source: s5_normctl/{kinf,k1}_s0..7.jsonl  (per-step records).
# ══════════════════════════════════════════════════════════════════════════════
make_A1 <- function() {
  conds <- c(uncapped = "kinf", capped = "k1")
  cond_lab <- c(uncapped = "Uncapped", capped = "Norm-capped")
  cond_col <- c(Uncapped = CB$grey, `Norm-capped` = CB$vermillion)

  rows <- list(); gmark <- list(); fracs <- list(); seedg <- list(); accs <- list()
  xmax <- 4000
  for (cl in names(conds)) {
    tag <- conds[[cl]]
    per_seed <- list(); grok_steps <- c()
    for (s in 0:7) {
      f <- file.path(res, "s5_normctl", sprintf("%s_s%d.jsonl", tag, s))
      if (!file.exists(f)) next
      recs <- step_records(f)
      df <- do.call(rbind, lapply(recs, function(r)
        data.frame(step = r$step, wn = r$wn_hidden, acc = r$test_acc)))
      per_seed[[length(per_seed) + 1]] <- df
      su <- last_summary(f); if (!is.null(su$grok_step)) grok_steps <- c(grok_steps, su$grok_step)
    }
    # union step grid up to xmax; forward-fill each seed (carry last recorded value forward)
    # so seeds that early-stop after grokking keep their post-grok value rather than dropping out
    all_steps <- sort(unique(unlist(lapply(per_seed, function(d) d$step))))
    all_steps <- all_steps[all_steps <= xmax]
    ffill <- function(v) { for (i in seq_along(v)) if (is.na(v[i]) && i > 1) v[i] <- v[i - 1]; v }
    wn_mat <- sapply(per_seed, function(d) ffill(d$wn[match(all_steps, d$step)]))
    agg <- data.frame(step = all_steps,
                      wn_med = apply(wn_mat, 1, med),
                      wn_lo  = apply(wn_mat, 1, function(x) quantile(x, .25, na.rm = TRUE)),
                      wn_hi  = apply(wn_mat, 1, function(x) quantile(x, .75, na.rm = TRUE)))
    agg$cond <- cond_lab[[cl]]
    rows[[cl]] <- agg
    gmark[[cl]] <- data.frame(cond = cond_lab[[cl]], grok = med(grok_steps))
    seedg[[cl]] <- data.frame(cond = cond_lab[[cl]], seed = seq_along(grok_steps),
                              grok_step = grok_steps)
    # panel (b): fraction of seeds grokked = empirical CDF of per-seed grok steps
    # (monotone; crosses 0.5 exactly at the median grok step marked by the dashed line)
    nseed <- length(per_seed)
    fr <- sapply(all_steps, function(t) sum(grok_steps <= t, na.rm = TRUE) / nseed)
    fracs[[cl]] <- data.frame(step = all_steps, frac = fr, cond = cond_lab[[cl]])
    # panel (d): median test-acc trajectory
    acc_mat <- sapply(per_seed, function(d) ffill(d$acc[match(all_steps, d$step)]))
    accs[[cl]] <- data.frame(step = all_steps,
                             acc_med = apply(acc_mat, 1, med),
                             acc_lo  = apply(acc_mat, 1, function(x) quantile(x, .25, na.rm = TRUE)),
                             acc_hi  = apply(acc_mat, 1, function(x) quantile(x, .75, na.rm = TRUE)),
                             cond = cond_lab[[cl]])
  }
  D <- bind_rows(rows); G <- bind_rows(gmark); Fr <- bind_rows(fracs)
  Sg <- bind_rows(seedg); Ac <- bind_rows(accs)
  D$cond <- factor(D$cond, levels = c("Uncapped", "Norm-capped"))
  G$cond <- factor(G$cond, levels = c("Uncapped", "Norm-capped"))
  Fr$cond <- factor(Fr$cond, levels = c("Uncapped", "Norm-capped"))
  Sg$cond <- factor(Sg$cond, levels = c("Uncapped", "Norm-capped"))
  Ac$cond <- factor(Ac$cond, levels = c("Uncapped", "Norm-capped"))

  pa <- ggplot(D, aes(step, wn_med, colour = cond, fill = cond)) +
    geom_ribbon(aes(ymin = wn_lo, ymax = wn_hi), alpha = 0.18, colour = NA) +
    geom_line(linewidth = 0.7) +
    geom_vline(data = G, aes(xintercept = grok, colour = cond),
               linetype = "dashed", linewidth = 0.45, show.legend = FALSE) +
    scale_colour_manual(values = cond_col, name = NULL) +
    scale_fill_manual(values = cond_col, name = NULL) +
    coord_cartesian(xlim = c(0, xmax)) +
    labs(title = "Hidden-weight norm during training",
         x = "Training step", y = "Hidden-weight norm") +
    paper_theme()

  pb <- ggplot(Fr, aes(step, frac, colour = cond)) +
    geom_step(linewidth = 0.7, direction = "hv") +
    geom_vline(data = G, aes(xintercept = grok, colour = cond),
               linetype = "dashed", linewidth = 0.45, show.legend = FALSE) +
    geom_hline(yintercept = 0.5, linetype = "dotted", linewidth = 0.35, colour = INK) +
    scale_colour_manual(values = cond_col, name = NULL) +
    coord_cartesian(xlim = c(0, xmax), ylim = c(0, 1.02)) +
    labs(title = "Fraction of seeds grokked",
         x = "Training step", y = "Fraction grokked (val acc ≥ 0.95)") +
    paper_theme()

  # (c) per-seed grok steps, both conditions
  pc <- ggplot(Sg, aes(cond, grok_step, colour = cond)) +
    stat_summary(fun = med, geom = "crossbar", width = 0.45, colour = INK) +
    geom_point(position = position_jitter(width = 0.06, height = 0, seed = 9),
               size = 2.2, alpha = 0.85, show.legend = FALSE) +
    scale_colour_manual(values = cond_col, name = NULL) +
    scale_y_log10(labels = label_comma()) +
    labs(title = "Per-seed grok steps",
         x = NULL, y = "Grok step (log scale)") +
    paper_theme()

  # (d) test-acc trajectories (median ± IQR)
  pd <- ggplot(Ac, aes(step, acc_med, colour = cond, fill = cond)) +
    geom_ribbon(aes(ymin = acc_lo, ymax = acc_hi), alpha = 0.18, colour = NA) +
    geom_line(linewidth = 0.7) +
    geom_vline(data = G, aes(xintercept = grok, colour = cond),
               linetype = "dashed", linewidth = 0.45, show.legend = FALSE) +
    scale_colour_manual(values = cond_col, name = NULL) +
    scale_fill_manual(values = cond_col, name = NULL) +
    coord_cartesian(xlim = c(0, xmax), ylim = c(0, 1.02)) +
    labs(title = "Validation accuracy during training",
         x = "Training step", y = "Validation accuracy") +
    paper_theme()

  p <- compose_A_four_panel(list(pa, pb, pc, pd), "A_normctl_timecourse", 7.2, 5.2, root) &
    theme(legend.position = "top")
  save_png(p, "A_normctl_timecourse", w = 7.2, h = 5.2)
  cat(sprintf("  [A1] grok step median uncapped=%.0f capped=%.0f\n",
              G$grok[G$cond == "Uncapped"], G$grok[G$cond == "Norm-capped"]))
}

# ══════════════════════════════════════════════════════════════════════════════
#  NEW-2 (A2) — Three-route norm-growth discriminator: norm_growth_ratio
#  by group ladder x optimizer. HONEST: optimizer-family discriminator, NOT a
#  monotone complexity law (AdamW non-monotone on one group).
#  Source: group_complexity/ladder_<grp>_<opt>_d256_s0..4.jsonl
# ══════════════════════════════════════════════════════════════════════════════
make_A2 <- function() {
  grp_ord <- c("Z60", "Z120", "D30", "D60", "A5", "S5")
  grp_lab <- c(Z60 = "Z60", Z120 = "Z120", D30 = "D30", D60 = "D60",
               A5 = "A5", S5 = "S5")
  opts <- c(AdamW = "adamw", Muon = "muon", SGDM = "sgdm")
  rows <- list(); seeds_rows <- list()
  for (g in grp_ord) for (ol in names(opts)) {
    o <- opts[[ol]]
    for (s in 0:4) {
      f <- file.path(res, "group_complexity", sprintf("ladder_%s_%s_d256_s%d.jsonl", g, o, s))
      if (!file.exists(f)) next
      su <- last_summary(f)
      grokked <- !is.null(su$final_test_acc) && !is.na(su$final_test_acc) && su$final_test_acc >= 0.95
      seeds_rows[[length(seeds_rows) + 1]] <- data.frame(
        grp = grp_lab[[g]], opt = ol, seed = s,
        ngr = su$norm_growth_ratio, grokked = grokked)
    }
  }
  S <- bind_rows(seeds_rows)
  D <- S %>% group_by(grp, opt) %>%
    summarise(ngr_med = med(ngr), ngr_lo = quantile(ngr, .25, na.rm = TRUE),
              ngr_hi = quantile(ngr, .75, na.rm = TRUE),
              grok_rate = mean(grokked), n = n(), .groups = "drop")
  D$grp <- factor(D$grp, levels = grp_lab[grp_ord])
  D$opt <- factor(D$opt, levels = c("AdamW", "Muon", "SGDM"))
  S$grp <- factor(S$grp, levels = grp_lab[grp_ord])
  S$opt <- factor(S$opt, levels = c("AdamW", "Muon", "SGDM"))

  # SGDM never groks (0/5 on every rung); its norm-growth ratio collapses to
  # ~0 so the bars are near-invisible. Tag each SGDM slot so a measured zero is
  # not mistaken for missing data. Mirror the A_floor "0/5" pattern: keep all
  # optimizers in the label frame (blank for AdamW/Muon) so position_dodge lands
  # the text on the correct (3rd) sub-bar.
  zero_lab <- D %>% mutate(lab = ifelse(opt == "SGDM", "0/5", ""))
  pa <- ggplot(D, aes(grp, ngr_med, fill = opt)) +
    geom_col(position = position_dodge(width = 0.78), width = 0.72, colour = NA) +
    geom_errorbar(aes(ymin = ngr_lo, ymax = ngr_hi),
                  position = position_dodge(width = 0.78), width = 0.28, linewidth = 0.45,
                  colour = "#3a3a3a") +
    geom_hline(yintercept = 1.0, linetype = "dashed", linewidth = 0.45, colour = INK) +
    geom_text(data = zero_lab, aes(grp, 0.08, label = lab, group = opt),
              position = position_dodge(width = 0.78), vjust = 0, size = 2.4,
              colour = "black", show.legend = FALSE) +
    annotate("text", x = 0.6, y = 1.05, label = "no net growth",
             hjust = 0, vjust = 0, size = 2.7, colour = "black", family = "TeX Gyre Termes") +
    scale_fill_manual(values = OPT_COL, name = NULL) +
    labs(title = "Norm growth across the group ladder",
         x = "Group (abelian → non-abelian)", y = "Norm-growth ratio (grok / init)") +
    paper_theme() +
    theme(panel.grid.major.x = element_blank())

  # (b) per-seed norm-growth ratios (the distribution behind the IQR bars)
  pb <- ggplot(S, aes(grp, ngr, colour = opt)) +
    geom_hline(yintercept = 1.0, linetype = "dashed", linewidth = 0.45, colour = INK) +
    geom_point(position = position_jitterdodge(jitter.width = 0.12, dodge.width = 0.78,
                                               seed = 4),
               size = 1.7, alpha = 0.85) +
    scale_colour_manual(values = OPT_COL, name = NULL) +
    labs(title = "Per-seed norm-growth ratios (5 seeds/cell)",
         x = "Group (abelian → non-abelian)", y = "Norm-growth ratio") +
    paper_theme() +
    theme(panel.grid.major.x = element_blank(), legend.position = "none")

  # (c) grok rate by rung: orthogonalization is NOT necessary (AdamW 5/5 except A5 3/5)
  pc <- ggplot(D, aes(grp, grok_rate, fill = opt)) +
    geom_col(position = position_dodge(width = 0.78), width = 0.72, colour = NA) +
    geom_text(aes(label = paste0(round(grok_rate * 5), "/5"), group = opt),
              position = position_dodge(width = 0.78), vjust = 1.25, size = 2.0,
              colour = "white", show.legend = FALSE) +
    scale_fill_manual(values = OPT_COL, name = NULL, guide = "none") +
    coord_cartesian(ylim = c(0, 1.15)) +
    labs(title = "Grok rate: only SGDM floors (0/5 everywhere)",
         x = "Group (abelian → non-abelian)", y = "Grok rate (5 seeds)") +
    paper_theme() +
    theme(panel.grid.major.x = element_blank(), legend.position = "none")

  # (d) family discriminator, not a complexity law: Muon ratio vs ladder position
  #     is non-monotone (Spearman, recomputed); AdamW rises above 1 only on A5.
  mu <- D$ngr_med[D$opt == "Muon"]; ad <- D$ngr_med[D$opt == "AdamW"]
  xpos <- seq_along(grp_ord)
  sp_mu <- suppressWarnings(cor.test(xpos, mu, method = "spearman", exact = FALSE))
  pd <- ggplot(D %>% filter(opt != "SGDM"), aes(grp, ngr_med, colour = opt, group = opt)) +
    geom_hline(yintercept = 1.0, linetype = "dashed", linewidth = 0.45, colour = INK) +
    geom_line(linewidth = 0.8) + geom_point(size = 2.2) +
    geom_text(data = data.frame(grp = factor("A5", levels = grp_lab[grp_ord]),
                                ngr_med = D$ngr_med[D$opt == "AdamW" & D$grp == "A5"],
                                opt = factor("AdamW", levels = c("AdamW", "Muon"))),
              aes(label = "A5: AdamW > 1, groks 3/5"), vjust = -1.1, size = 2.4,
              colour = OPT_COL[["AdamW"]], show.legend = FALSE) +
    scale_colour_manual(values = OPT_COL, name = NULL) +
    labs(title = sprintf("Not a complexity law (Muon Spearman rho = %.2f, p = %.2f)",
                         unname(sp_mu$estimate), sp_mu$p.value),
         x = "Group (abelian → non-abelian)", y = "Median norm-growth ratio") +
    paper_theme() +
    theme(panel.grid.major.x = element_blank(), legend.position = "none")

  p <- compose_A_four_panel(list(pa, pb, pc, pd), "A_norm_discriminator", 7.2, 5.4, root) &
    theme(legend.position = "top")
  save_png(p, "A_norm_discriminator", w = 7.2, h = 5.4)
  cat("  [A2] muon ngr range ",
      sprintf("%.2f-%.2f", min(D$ngr_med[D$opt=="Muon"]), max(D$ngr_med[D$opt=="Muon"])),
      "| adamw ",
      sprintf("%.2f-%.2f", min(D$ngr_med[D$opt=="AdamW"]), max(D$ngr_med[D$opt=="AdamW"])),
      "| sgdm max ", sprintf("%.3f", max(D$ngr_med[D$opt=="SGDM"])), "\n")
}

# ══════════════════════════════════════════════════════════════════════════════
#  NEW-3 (A3, appendix) — Activation spike vs depth. Existence/spread (n=3).
#  Source: sink_triad/depth_<opt>_pre_{L1,L3}_s{0,1,2}.jsonl
# ══════════════════════════════════════════════════════════════════════════════
make_A3 <- function() {
  opts <- c(AdamW = "adamw", Muon = "muon", SGDM = "sgdm")
  Ls   <- c(L1 = 1, L3 = 3)
  rows <- list(); traj <- list()
  for (ol in names(opts)) for (Ln in names(Ls)) for (s in 0:2) {
    f <- file.path(res, "sink_triad", sprintf("depth_%s_pre_%s_s%d.jsonl", opts[[ol]], Ln, s))
    if (!file.exists(f)) next
    su <- last_summary(f)
    rows[[length(rows) + 1]] <- data.frame(opt = ol, depth = Ls[[Ln]],
                                           spike = su$final_spike_magnitude, seed = s)
    recs <- step_records(f)
    if (length(recs)) {
      td <- do.call(rbind, lapply(recs, function(r)
        data.frame(step = r$step,
                   spike = max(c(r$spike_l0, r$spike_l1), na.rm = TRUE))))
      td$opt <- ol; td$depth <- Ls[[Ln]]; td$seed <- s
      traj[[length(traj) + 1]] <- td
    }
  }
  D <- bind_rows(rows)
  D$opt <- factor(D$opt, levels = c("AdamW", "Muon", "SGDM"))
  Dm <- D %>% group_by(opt, depth) %>% summarise(spike_med = med(spike), .groups = "drop")
  Tj <- bind_rows(traj)
  Tj$opt <- factor(Tj$opt, levels = c("AdamW", "Muon", "SGDM"))

  pa <- ggplot(D, aes(depth, spike, colour = opt)) +
    geom_line(data = Dm, aes(depth, spike_med, colour = opt), linewidth = 0.7) +
    geom_point(size = 1.6, alpha = 0.75) +
    scale_colour_manual(values = OPT_COL, name = NULL) +
    scale_x_continuous(breaks = c(1, 3), labels = c("1 layer", "3 layers")) +
    scale_y_log10(labels = label_comma()) +
    labs(title = "Peak activation magnitude vs depth",
         x = "Network depth", y = "Peak activation magnitude (log scale)") +
    paper_theme()

  # (b) spike formation dynamics: median spike vs step by depth arm
  Tm <- Tj %>% group_by(opt, depth, step) %>%
    summarise(spike_med = med(spike), .groups = "drop") %>%
    mutate(depth_lab = factor(paste0(depth, "L"), levels = c("1L", "3L")))
  pb <- ggplot(Tm, aes(step, spike_med, colour = opt, linetype = depth_lab)) +
    geom_line(linewidth = 0.65) +
    scale_colour_manual(values = OPT_COL, name = NULL) +
    scale_linetype_manual(values = c("solid", "dashed"), name = NULL) +
    scale_y_log10(labels = label_comma()) +
    labs(title = "Spike formation during training (median of 3 seeds)",
         x = "Training step", y = "Peak activation (log scale)") +
    paper_theme() +
    theme(legend.position = "top")

  # (c) lr robustness: final spike vs learning rate (sink_triad_lr arm)
  lr_rows <- list()
  lr_files <- list.files(file.path(res, "sink_triad_lr"), pattern = "\\.jsonl$",
                         full.names = TRUE)
  for (f in lr_files) {
    b <- basename(f)
    m <- regmatches(b, regexec("^(adamw|muon|sgdm)_lr([0-9.]+)_pre_s([0-9])", b))[[1]]
    if (length(m) == 0) next
    su <- last_summary(f)
    lr_rows[[length(lr_rows) + 1]] <- data.frame(
      opt = c(adamw = "AdamW", muon = "Muon", sgdm = "SGDM")[[m[2]]],
      lr = as.numeric(m[3]), spike = su$final_spike_magnitude)
  }
  Lr <- bind_rows(lr_rows)
  Lr$opt <- factor(Lr$opt, levels = c("AdamW", "Muon", "SGDM"))
  pc <- ggplot(Lr, aes(factor(lr), spike, colour = opt)) +
    stat_summary(fun = med, geom = "crossbar", width = 0.4, colour = INK) +
    geom_point(position = position_jitter(width = 0.06, height = 0, seed = 6),
               size = 1.7, alpha = 0.85) +
    facet_wrap(~ opt, nrow = 1) +
    scale_colour_manual(values = OPT_COL, guide = "none") +
    scale_y_log10(labels = label_comma()) +
    labs(title = "Learning-rate robustness of the spike",
         x = "Learning rate", y = "Peak activation (log scale)") +
    paper_theme() +
    theme(axis.text.x = element_text(angle = 35, hjust = 1, size = 6.5))

  # (d) per-seed spread at each depth (the n=3 honesty panel)
  pd <- ggplot(D, aes(factor(depth), spike, colour = opt)) +
    stat_summary(fun = med, geom = "crossbar", width = 0.4, colour = INK) +
    geom_point(position = position_jitter(width = 0.05, height = 0, seed = 6),
               size = 1.9, alpha = 0.85) +
    facet_wrap(~ opt, nrow = 1) +
    scale_colour_manual(values = OPT_COL, guide = "none") +
    scale_y_log10(labels = label_comma()) +
    labs(title = "Per-seed spread (n = 3)",
         x = "Depth (layers)", y = "Peak activation (log scale)") +
    paper_theme()

  p <- (pa | pb) / (pc | pd) + tag_ann() & theme(legend.position = "top")
  save_png(p, "A_spike_depth", w = 7.2, h = 5.4)
  cat("  [A3] muon spike L1->L3 median ",
      sprintf("%.0f -> %.0f", Dm$spike_med[Dm$opt=="Muon"&Dm$depth==1], Dm$spike_med[Dm$opt=="Muon"&Dm$depth==3]),
      "\n")
}

# ══════════════════════════════════════════════════════════════════════════════
#  NEW-4 (A4, appendix) — S5 rescue coverage map (2 seeds/cell): which
#  optimizer x lr x weight-decay cells recover grokking. Coverage/existence map.
#  Source: s5_rescue/<opt>_lr<lr>_wd<wd>_s{0,1}.jsonl
# ══════════════════════════════════════════════════════════════════════════════
make_A4 <- function() {
  files <- list.files(file.path(res, "s5_rescue"), pattern = "\\.jsonl$", full.names = TRUE)
  rows <- list()
  for (f in files) {
    b <- basename(f)
    m <- regmatches(b, regexec("^(adamw|sgdm)_lr([0-9.]+)_wd([0-9.]+)_s([0-9])", b))[[1]]
    if (length(m) == 0) next
    opt <- m[2]; lr <- m[3]; wd <- m[4]
    su <- last_summary(f)
    rows[[length(rows) + 1]] <- data.frame(
      opt = opt, lr = lr, wd = wd,
      grok = !is.null(su$grok_step) && !is.na(su$grok_step),
      grok_step = if (!is.null(su$grok_step) && !is.na(su$grok_step)) su$grok_step else NA_real_)
  }
  R <- bind_rows(rows)
  D <- R %>%
    group_by(opt, lr, wd) %>%
    summarise(n_grok = sum(grok), n = n(), .groups = "drop")
  opt_lab <- c(adamw = "AdamW", sgdm = "SGDM")
  D$opt <- factor(opt_lab[D$opt], levels = c("AdamW", "SGDM"))
  R$opt <- factor(opt_lab[R$opt], levels = c("AdamW", "SGDM"))
  # order wd / lr by numeric value but keep clean string labels (no "NA")
  wd_lv <- unique(D$wd); wd_lv <- wd_lv[order(as.numeric(wd_lv))]
  lr_lv <- unique(D$lr); lr_lv <- lr_lv[order(as.numeric(lr_lv))]
  D$wd  <- factor(D$wd, levels = wd_lv)
  D$lr  <- factor(D$lr, levels = lr_lv)
  D$cov <- D$n_grok / D$n
  D$lab <- sprintf("%d/%d", D$n_grok, D$n)

  pa <- ggplot(D, aes(wd, lr, fill = cov)) +
    geom_tile(colour = "white", linewidth = 0.7) +
    geom_text(aes(label = lab), size = 2.9, colour = "#111827", family = "TeX Gyre Termes") +
    facet_wrap(~ opt, nrow = 1, scales = "free_y") +
    scale_fill_gradient(low = "#f3f4f6", high = CB$green, limits = c(0, 1),
                        name = "Grok\nfraction") +
    labs(title = "Recovery of generalization across hyperparameters",
         x = "Weight decay", y = "Learning rate") +
    paper_theme() +
    theme(panel.grid.major.y = element_blank(), legend.position = "right")

  # (b) per-lr marginal coverage
  M <- D %>% group_by(opt, lr) %>%
    summarise(cov = sum(n_grok) / sum(n), .groups = "drop")
  pb <- ggplot(M, aes(lr, cov, fill = opt)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.68, colour = NA) +
    scale_fill_manual(values = c(AdamW = OPT_COL[["AdamW"]], SGDM = OPT_COL[["SGDM"]]),
                      name = NULL) +
    coord_cartesian(ylim = c(0, 1.05)) +
    labs(title = "Marginal coverage by learning rate",
         x = "Learning rate", y = "Grok fraction (all wd)") +
    paper_theme() +
    theme(legend.position = "top")

  # (c) time-to-grok distribution in rescued cells
  Gt <- R %>% filter(grok)
  pc <- ggplot(Gt, aes(opt, grok_step, colour = opt)) +
    stat_summary(fun = med, geom = "crossbar", width = 0.45, colour = INK) +
    geom_point(position = position_jitter(width = 0.07, height = 0, seed = 8),
               size = 1.7, alpha = 0.8, show.legend = FALSE) +
    scale_colour_manual(values = c(AdamW = OPT_COL[["AdamW"]], SGDM = OPT_COL[["SGDM"]]),
                        guide = "none") +
    scale_y_log10(labels = label_comma()) +
    labs(title = "Time-to-grok in rescued cells",
         x = NULL, y = "Grok step (log scale)") +
    paper_theme()

  # (d) seed-pair agreement: 2/2 vs 1/2 vs 0/2 cells
  A <- D %>% mutate(cls = factor(paste0(n_grok, "/", n), levels = c("2/2", "1/2", "0/2"))) %>%
    count(opt, cls)
  pd <- ggplot(A, aes(cls, n, fill = opt)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.68, colour = NA) +
    geom_text(aes(label = n, group = opt), position = position_dodge(width = 0.75),
              vjust = -0.4, size = 2.6, colour = "#111827", show.legend = FALSE) +
    scale_fill_manual(values = c(AdamW = OPT_COL[["AdamW"]], SGDM = OPT_COL[["SGDM"]]),
                      name = NULL) +
    scale_y_continuous(expand = expansion(mult = c(0.02, 0.14))) +
    labs(title = "Cell verdict agreement (2-seed cells)",
         x = "Seeds grokking", y = "Cells") +
    paper_theme() +
    theme(legend.position = "none")

  p <- (pa | pb) / (pc | pd) + tag_ann() & theme(legend.position = "top")
  save_png(p, "A_rescue_coverage", w = 7.2, h = 5.4)
  ag <- sum(D$n_grok[D$opt == "AdamW"]); sg <- sum(D$n_grok[D$opt == "SGDM"])
  cat(sprintf("  [A4] AdamW grok cells total %d/%d ; SGDM %d/%d\n",
              ag, sum(D$n[D$opt=="AdamW"]), sg, sum(D$n[D$opt=="SGDM"])))
}

# ══════════════════════════════════════════════════════════════════════════════
#  NEW-5 (A5, appendix) — Feature effective-rank decay across continual tasks
#  on permuted-MNIST, by optimizer and learning rate. Median +/- band.
#  Source: muon_plasticity_mnist/{adamw,muon,sgdm}_s{0,1,2}.jsonl  (per-task probes)
#          muon_plasticity_mnist_lrctl/muon_mlr*_s*.jsonl
# ══════════════════════════════════════════════════════════════════════════════
make_A5 <- function() {
  read_arm <- function(subdir, fname) {
    f <- file.path(res, subdir, fname)
    if (!file.exists(f)) return(NULL)
    recs <- step_records_task(f)
    if (length(recs) == 0) return(NULL)
    do.call(rbind, lapply(recs, function(r)
      data.frame(task = r$task, fr = r$probes$feat_eff_rank)))
  }
  # task-record reader (records keyed by "task" not "step")
  arms <- list(
    list(lab = "AdamW",            col = OPT_COL[["AdamW"]], dir = "muon_plasticity_mnist",       pat = "adamw_s%d.jsonl", seeds = 0:2),
    list(lab = "Muon",             col = OPT_COL[["Muon"]],  dir = "muon_plasticity_mnist",       pat = "muon_s%d.jsonl",  seeds = 0:2),
    list(lab = "SGDM",             col = OPT_COL[["SGDM"]],  dir = "muon_plasticity_mnist",       pat = "sgdm_s%d.jsonl",  seeds = 0:2),
    list(lab = "Muon (scaled)", col = CB$green,           dir = "muon_plasticity_mnist_lrctl", pat = "muon_mlr0p005_s%d.jsonl", seeds = 0:4)
  )
  rows <- list()
  for (a in arms) {
    per <- list()
    for (s in a$seeds) {
      df <- read_arm(a$dir, sprintf(a$pat, s))
      if (!is.null(df)) per[[length(per) + 1]] <- df
    }
    if (length(per) == 0) next
    tasks <- Reduce(intersect, lapply(per, function(d) d$task))
    agg <- do.call(rbind, lapply(tasks, function(t) {
      fr <- sapply(per, function(d) d$fr[match(t, d$task)])
      data.frame(task = t, fr_med = med(fr),
                 fr_lo = quantile(fr, .25, na.rm = TRUE), fr_hi = quantile(fr, .75, na.rm = TRUE))
    }))
    agg$arm <- a$lab; agg$col <- a$col
    rows[[length(rows) + 1]] <- agg
  }
  D <- bind_rows(rows)
  arm_levels <- c("AdamW", "Muon", "SGDM", "Muon (scaled lr)")
  arm_cols   <- c(AdamW = OPT_COL[["AdamW"]], Muon = OPT_COL[["Muon"]],
                  SGDM = OPT_COL[["SGDM"]], `Muon (scaled lr)` = CB$green)
  D$arm <- factor(D$arm, levels = arm_levels)

  # per-seed endpoints + rank-halving task for panels (b)-(d)
  end_rows <- list(); half_rows <- list(); init_rows <- list()
  for (a in arms) {
    per <- list()
    for (s in a$seeds) {
      df <- read_arm(a$dir, sprintf(a$pat, s))
      if (!is.null(df)) per[[length(per) + 1]] <- df
    }
    if (length(per) == 0) next
    for (i in seq_along(per)) {
      d <- per[[i]]
      end_rows[[length(end_rows) + 1]] <- data.frame(
        arm = a$lab, seed = i - 1, fr_end = d$fr[which.max(d$task)],
        fr_init = d$fr[which.min(d$task)])
      half <- d$fr[which.min(d$task)] / 2
      ht <- suppressWarnings(min(d$task[d$fr <= half]))
      half_rows[[length(half_rows) + 1]] <- data.frame(
        arm = a$lab, seed = i - 1, half_task = if (is.finite(ht)) ht else NA_real_)
    }
  }
  E <- bind_rows(end_rows); H <- bind_rows(half_rows)
  E$arm <- factor(E$arm, levels = arm_levels)
  H$arm <- factor(H$arm, levels = arm_levels)

  pa <- ggplot(D, aes(task, fr_med, colour = arm, fill = arm)) +
    geom_ribbon(aes(ymin = fr_lo, ymax = fr_hi), alpha = 0.14, colour = NA) +
    geom_line(linewidth = 0.65) +
    scale_colour_manual(values = arm_cols, name = NULL) +
    scale_fill_manual(values = arm_cols, name = NULL) +
    labs(title = "Feature effective rank across a continual-learning stream",
         x = "Task index", y = "Feature effective rank") +
    paper_theme() +
    theme(legend.position = "top")

  # (b) endpoint effective rank per seed (150th task)
  pb <- ggplot(E, aes(arm, fr_end, colour = arm)) +
    stat_summary(fun = med, geom = "crossbar", width = 0.45, colour = INK) +
    geom_point(position = position_jitter(width = 0.06, height = 0, seed = 2),
               size = 2.0, alpha = 0.85, show.legend = FALSE) +
    scale_colour_manual(values = arm_cols, guide = "none") +
    coord_cartesian(ylim = c(0, NA)) +
    labs(title = "Endpoint rank per seed (final task)",
         x = NULL, y = "Feature effective rank") +
    paper_theme() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))

  # (c) decay speed: task index at which rank first halves (per seed)
  pc <- ggplot(H, aes(arm, half_task, colour = arm)) +
    stat_summary(fun = function(x) med(x), geom = "crossbar", width = 0.45,
                 colour = INK) +
    geom_point(position = position_jitter(width = 0.06, height = 0, seed = 2),
               size = 2.0, alpha = 0.85, show.legend = FALSE) +
    scale_colour_manual(values = arm_cols, guide = "none") +
    labs(title = "Decay speed: first task with rank ≤ initial/2",
         x = NULL, y = "Task index") +
    paper_theme() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))

  # (d) rank retained: endpoint / initial per seed
  E2 <- E %>% mutate(retained = fr_end / fr_init)
  pd <- ggplot(E2, aes(arm, retained, colour = arm)) +
    stat_summary(fun = med, geom = "crossbar", width = 0.45, colour = INK) +
    geom_point(position = position_jitter(width = 0.06, height = 0, seed = 2),
               size = 2.0, alpha = 0.85, show.legend = FALSE) +
    scale_colour_manual(values = arm_cols, guide = "none") +
    coord_cartesian(ylim = c(0, 1.02)) +
    labs(title = "Rank retained at stream end",
         x = NULL, y = "Endpoint / initial rank") +
    paper_theme() +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))

  p <- (pa | pb) / (pc | pd) + tag_ann() & theme(legend.position = "top")
  save_png(p, "A_feature_rank_decay", w = 7.2, h = 5.4)
  last <- D %>% group_by(arm) %>% filter(task == max(task)) %>% ungroup()
  cat("  [A5] final feature rank: ",
      paste(sprintf("%s=%.0f", last$arm, last$fr_med), collapse = " "), "\n")
}

# task-keyed record reader used by A5
step_records_task <- function(path) {
  lines <- readLines(path, warn = FALSE)
  lines <- lines[nzchar(trimws(lines))]
  out <- list()
  for (ln in lines) {
    txt <- gsub("\\bNaN\\b", "null", ln)
    txt <- gsub("\\bInfinity\\b", "null", txt); txt <- gsub("\\b-Infinity\\b", "null", txt)
    rec <- tryCatch(jsonlite::fromJSON(txt, simplifyVector = TRUE), error = function(e) NULL)
    if (!is.null(rec) && !is.null(rec[["task"]]) && !is.null(rec[["probes"]])) out[[length(out) + 1]] <- rec
  }
  out
}

# ── run all ───────────────────────────────────────────────────────────────────
cat("Rendering new Paper-A figures...\n")
make_A1()
make_A2()
make_A3()
make_A4()
make_A5()
cat("done.\n")
