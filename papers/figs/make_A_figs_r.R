#!/usr/bin/env Rscript
# make_A_figs_r.R — Professional R/ggplot2 renderer for all 11 Paper-A figures.
# Reads the SAME underlying data as the matplotlib generators.
# Output:  PNG → papers/figs/<name>.png   (overwrites; 300 dpi via ragg)
#          SVG → papers/figs/evidence_r/<name>.svg
# Style:   Okabe-Ito colourblind-safe palette, paper_theme() matching the
#          shared stats-figures template.
# Usage:   Rscript papers/figs/make_A_figs_r.R   (from repo root)

suppressPackageStartupMessages({
  library(jsonlite)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(scales)
  library(patchwork)
  library(ragg)
  library(svglite)
})

# ── paths ─────────────────────────────────────────────────────────────────────
root   <- normalizePath(file.path(getwd()))
figdir <- file.path(root, "papers", "figs")
evdir  <- file.path(figdir, "evidence_r")
res    <- file.path(root, "experiments", "results")
dir.create(evdir, showWarnings = FALSE, recursive = TRUE)
source(file.path(figdir, "fig_pipeline.R"))  # emit_vector(): tikz/.tex + cairo_pdf/.pdf
source(file.path(figdir, "A_panel_contract.R"))

# ── colour palette (Okabe-Ito, matching figstyle.py) ─────────────────────────
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

# ── shared theme ──────────────────────────────────────────────────────────────
paper_theme <- function(base_size = 8.5) {
  theme_minimal(base_size = base_size, base_family = "TeX Gyre Termes") +
    theme(
      panel.grid.minor     = element_blank(),
      panel.grid.major.x   = element_blank(),
      panel.grid.major.y   = element_line(linewidth = 0.25, colour = "#d9dde3"),
      axis.title           = element_text(colour = "#1a202c", size = base_size + 0.5,
                                          face = "plain", family = "TeX Gyre Termes"),
      axis.text            = element_text(colour = "#2d3748", size = base_size,
                                          face = "plain", family = "TeX Gyre Termes"),
      plot.title           = element_text(face = "plain", colour = "#111827",
                                          size = base_size + 1.5, family = "TeX Gyre Termes"),
      plot.subtitle        = element_text(colour = "#1a1a1a", size = base_size,
                                          face = "plain", family = "TeX Gyre Termes"),
      legend.position      = "top",
      legend.title         = element_text(size = base_size, face = "plain",
                                          family = "TeX Gyre Termes"),
      legend.text          = element_text(size = base_size, face = "plain",
                                          family = "TeX Gyre Termes"),
      strip.text           = element_text(face = "plain", colour = "#1a202c",
                                          size = base_size, family = "TeX Gyre Termes"),
      plot.margin          = margin(5.5, 6, 5.5, 6),
      plot.tag             = element_text(face = "bold", size = 11,
                                          family = "TeX Gyre Termes"),
      plot.tag.position    = "topleft"
    )
}

# ── patchwork tag annotation helper ───────────────────────────────────────────
tag_annotation <- function() {
  plot_annotation(tag_levels = "a", tag_prefix = "(", tag_suffix = ")")
}

# ── save helper ───────────────────────────────────────────────────────────────
save_both <- function(p, name, w = 7.2, h = 4.2) {
  pngp <- file.path(figdir, paste0(name, ".png"))
  svgp <- file.path(evdir,  paste0(name, ".svg"))
  ragg::agg_png(pngp, width = w, height = h, units = "in", res = 300, scaling = 1)
  print(p); dev.off()
  svglite::svglite(svgp, width = w, height = h)
  print(p); dev.off()
  emit_vector(p, name, w, h)   # vector tier: tikz (.tex) or cairo_pdf (.pdf)
  cat(sprintf("saved %s  (%s)\n", pngp, svgp))
}

# ── JSON helpers ──────────────────────────────────────────────────────────────
read_json_safe <- function(path, sv = TRUE) {
  txt <- paste(readLines(path, warn = FALSE), collapse = "\n")
  txt <- gsub("\\bInfinity\\b",   "1e308",  txt)
  txt <- gsub("\\b-Infinity\\b",  "-1e308", txt)
  txt <- gsub("\\bNaN\\b",        "null",   txt)
  jsonlite::fromJSON(txt, simplifyVector = sv)
}

last_summary <- function(path) {
  lines <- readLines(path, warn = FALSE)
  lines <- lines[nzchar(trimws(lines))]
  txt   <- lines[length(lines)]
  txt   <- gsub("\\bNaN\\b",        "null", txt)
  txt   <- gsub("\\bInfinity\\b",   "null", txt)
  txt   <- gsub("\\b-Infinity\\b",  "null", txt)
  rec   <- jsonlite::fromJSON(txt, simplifyVector = TRUE)
  if (!is.null(rec[["_summary"]])) rec[["_summary"]] else rec
}

# ══════════════════════════════════════════════════════════════════════════════
#  1. A_normctl — Norm-control causal figure (§3.2)
#     Data: experiments/results/figures-021/normctl_verdict.json
# ══════════════════════════════════════════════════════════════════════════════
make_A_normctl <- function() {
  v      <- read_json_safe(file.path(res, "figures-021", "normctl_verdict.json"))
  k_ord  <- c("kinf", "k3", "k2", "k1p5", "k1")
  k_lab  <- c(kinf = "∞", k3 = "3", k2 = "2", k1p5 = "1.5", k1 = "1")
  k_num  <- c(kinf = 5.5, k3 = 3.0, k2 = 2.0, k1p5 = 1.5, k1 = 1.0)  # x positions

  extract_arm <- function(arm) {
    bind_rows(lapply(k_ord, function(k) {
      r <- arm$rows[[k]]
      data.frame(k = k, k_x = k_num[[k]], k_label = k_lab[[k]],
                 grok_med = r$grok_step_med, wn_med = r$wn_hidden_med,
                 n_grok = r$n_grok, n = r$n, stringsAsFactors = FALSE)
    }))
  }
  df_s5  <- extract_arm(v$s5)  %>% mutate(task = "S5 (non-abelian)")
  df_add <- extract_arm(v$add) %>% mutate(task = "Modular addition")
  df     <- bind_rows(df_s5, df_add) %>%
    mutate(k_label = factor(k_label, levels = rev(k_lab)),
           grok_text = paste0(n_grok, "/", n))

  task_col <- c("S5 (non-abelian)" = CB$vermillion, "Modular addition" = CB$blue)
  task_shp <- c("S5 (non-abelian)" = 16,            "Modular addition" = 15)

  p1 <- ggplot(df, aes(k_label, grok_med, colour = task, shape = task, group = task)) +
    geom_line(linewidth = 1.2) +
    geom_point(size = 3.5, stroke = 0.8) +
    geom_text(data = filter(df, task == "S5 (non-abelian)"),
              aes(label = grok_text), vjust = -1.1, size = 2.7,
              colour = CB$vermillion, show.legend = FALSE) +
    scale_y_log10(labels = label_comma(),
                  expand = expansion(mult = c(0.05, 0.18))) +
    coord_cartesian(clip = "off") +
    scale_colour_manual(values = task_col, name = NULL) +
    # shape is redundant with colour (same task variable): drop its guide so
    # the collected legend renders ONE merged-glyph block, not twin boxes
    scale_shape_manual(values = task_shp, name = NULL, guide = "none") +
    labs(title = "Grok step vs ceiling",
         x = expression("Hidden-norm ceiling " * k * "  (" * group("||", W, "||") <= k %.% group("||", W, "||")[init] * ")"),
         y = "Median grok step (log scale)") +
    paper_theme() +
    theme(legend.position = "top")

  p2 <- ggplot(df, aes(k_label, n_grok / n, colour = task, shape = task,
                       group = task)) +
    geom_hline(yintercept = 1, linetype = "dashed", colour = CB$grey) +
    geom_line(linewidth = 1.1) +
    geom_point(size = 3.2, stroke = 0.8) +
    geom_text(aes(label = grok_text), vjust = -0.9, size = 2.5,
              show.legend = FALSE) +
    # composite-level theme() overrides per-panel legend.position, so the
    # duplicate (b) legend must be killed at the scale level instead
    scale_colour_manual(values = task_col, name = NULL, guide = "none") +
    scale_shape_manual(values = task_shp, name = NULL, guide = "none") +
    coord_cartesian(ylim = c(0, 1.12)) +
    labs(title = "Grokking preservation rate",
         x = expression("Hidden-norm ceiling " * k),
         y = "Fraction grokked") +
    paper_theme() +
    theme(legend.position = "none")

  # ── per-seed S5 grok steps from raw jsonl (panels c/d; numbers from disk) ──
  s5_dir <- file.path(res, "s5_normctl")
  seeds_df <- bind_rows(lapply(k_ord, function(k) {
    bind_rows(lapply(0:7, function(s) {
      f <- file.path(s5_dir, sprintf("%s_s%d.jsonl", k, s))
      if (!file.exists(f)) return(NULL)
      rec <- last_summary(f)
      data.frame(k = k, seed = s, grok_step = rec$grok_step,
                 stringsAsFactors = FALSE)
    }))
  })) %>% mutate(k_label = factor(k_lab[k], levels = rev(k_lab)))

  # Paired per-seed acceleration kinf -> k1. The interval is read from the
  # persisted bootstrap summary; points show the eight underlying paired ratios.
  wide <- seeds_df %>% filter(k %in% c("kinf", "k1")) %>%
    select(k, seed, grok_step) %>% pivot_wider(names_from = k, values_from = grok_step) %>%
    mutate(ratio = kinf / k1)
  bc <- read_json_safe(file.path(root, "experiments", "revision2026", "A",
                                 "bootstrap_cis.json"), sv = FALSE)
  pick_boot <- function(name) Filter(function(entry) entry$name == name, bc$entries)[[1]]
  s5_ci <- pick_boot("S5 primary: kinf/k1")
  p3 <- ggplot(wide, aes("kinf -> k1", ratio)) +
    geom_hline(yintercept = 1, linetype = "dashed", colour = CB$grey) +
    stat_summary(fun = median, geom = "crossbar", width = 0.4, colour = CB$black) +
    geom_point(position = position_jitter(width = 0.05, height = 0, seed = 11),
               size = 2.2, colour = CB$vermillion, alpha = 0.85) +
    geom_errorbar(data = data.frame(x = 1, lo = s5_ci$ci95_percentile[[1]],
                                    hi = s5_ci$ci95_percentile[[2]]),
                  aes(x = x, ymin = lo, ymax = hi), width = 0.18,
                  colour = CB$black, inherit.aes = FALSE) +
    geom_text(data = data.frame(x = 1, y = s5_ci$ci95_percentile[[2]] * 1.08),
              aes(x = x, y = y, label = sprintf("%.2fx [%.2f, %.2f]",
                  s5_ci$ratio, s5_ci$ci95_percentile[[1]],
                  s5_ci$ci95_percentile[[2]])),
              size = 2.7, inherit.aes = FALSE) +
    labs(title = "S5 paired acceleration interval",
         x = NULL, y = "Grok-step ratio (uncapped / k = 1)") +
    paper_theme()

  # Cross-task boundary: the cap preserves acceleration direction but available
  # headroom varies strongly across tasks and evaluation cadence.
  adf <- read_json_safe(file.path(root, "experiments", "revision2026", "gpu2026",
                                  "adf", "adf_verdict.json"), sv = FALSE)
  adf_pick <- function(task) Filter(function(entry) entry$task == task, adf$entries)[[1]]
  boundary <- data.frame(
    task = factor(c("S5", "D60", "A5", "mod-add"),
                  levels = c("S5", "D60", "A5", "mod-add")),
    est = c(s5_ci$ratio, adf_pick("D60")$ratio,
            pick_boot("A5: kinf/k1")$ratio, adf_pick("add")$ratio),
    lo = c(s5_ci$ci95_percentile[[1]], adf_pick("D60")$ci95_percentile[[1]],
           pick_boot("A5: kinf/k1")$ci95_percentile[[1]],
           adf_pick("add")$ci95_percentile[[1]]),
    hi = c(s5_ci$ci95_percentile[[2]], adf_pick("D60")$ci95_percentile[[2]],
           pick_boot("A5: kinf/k1")$ci95_percentile[[2]],
           adf_pick("add")$ci95_percentile[[2]]))
  p4 <- ggplot(boundary, aes(task, est)) +
    geom_hline(yintercept = 1, linetype = "dashed", colour = CB$grey) +
    geom_point(size = 2.6, colour = CB$black) +
    geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.2, colour = CB$black) +
    geom_text(aes(label = sprintf("%.2fx", est)), vjust = -0.9, size = 2.5) +
    labs(title = "Task boundary: effect size, not direction",
         x = NULL, y = "Acceleration (uncapped / k = 1)") +
    paper_theme()

  save_both(
    compose_A_four_panel(list(p1, p2, p3, p4), "A_normctl", 7.2, 5.2, root) &
      theme(legend.position = "top"),
    "A_normctl", w = 7.2, h = 5.2
  )
}

# ══════════════════════════════════════════════════════════════════════════════
#  2. A_lmc — LMC barrier trajectories (§3.4)
#     Data: experiments/results/lmc_instability/{adamw,muon}_s{0..4}.jsonl
# ══════════════════════════════════════════════════════════════════════════════
make_A_lmc <- function() {
  LOCK <- 0.1
  lmc_dir <- file.path(res, "lmc_instability")

  collect_opt <- function(opt) {
    bind_rows(lapply(0:4, function(i) {
      p <- file.path(lmc_dir, sprintf("%s_s%d.jsonl", opt, i))
      if (!file.exists(p)) return(NULL)
      s   <- last_summary(p)
      bbs <- s[["barrier_by_spawn"]]
      if (is.null(bbs)) return(NULL)
      steps   <- as.integer(names(bbs))
      barriers <- as.numeric(unlist(bbs))
      data.frame(optimizer = opt, seed = i, spawn_step = steps,
                 barrier = barriers, stringsAsFactors = FALSE)
    }))
  }

  df_aw <- collect_opt("adamw") %>% mutate(opt_label = "AdamW")
  df_mu <- collect_opt("muon")  %>% mutate(opt_label = "Muon")
  df    <- bind_rows(df_aw, df_mu)

  # median trajectory per optimizer
  med_df <- df %>%
    group_by(opt_label, spawn_step) %>%
    summarise(barrier = median(barrier), .groups = "drop")

  # barrier at last spawn step per seed
  b8_df <- df %>%
    group_by(opt_label, seed) %>%
    slice_max(spawn_step, n = 1) %>%
    ungroup()

  p1 <- ggplot(df, aes(spawn_step, barrier, colour = opt_label, group = interaction(opt_label, seed))) +
    geom_line(linewidth = 0.6, alpha = 0.35) +
    geom_line(data = med_df, aes(group = opt_label), linewidth = 1.6) +
    geom_point(data = med_df, aes(group = opt_label), size = 2.5) +
    geom_hline(yintercept = LOCK, linetype = "dotted", colour = CB$grey, linewidth = 0.8) +
    scale_x_continuous(trans = scales::pseudo_log_trans(base = 10, sigma = 50),
                       breaks = c(0, 100, 1000, 8000),
                       labels = label_comma()) +
    scale_colour_manual(values = OPT_COL, name = NULL) +
    labs(title = "Barrier trajectories",
         x = "Spawn step (symlog)", y = "Inter-child loss barrier") +
    paper_theme() +
    theme(legend.position = "top")

  med8_df <- b8_df %>%
    group_by(opt_label) %>%
    summarise(med_barrier = median(barrier), .groups = "drop")

  opt_levels <- sort(unique(b8_df$opt_label))
  med8_df <- med8_df %>%
    mutate(opt_x = as.numeric(factor(opt_label, levels = opt_levels)))

  set.seed(42)
  p2 <- ggplot(b8_df, aes(opt_label, barrier, colour = opt_label)) +
    geom_jitter(size = 3, width = 0.06, height = 0) +
    geom_segment(data = med8_df,
                 aes(x = opt_x - 0.15, xend = opt_x + 0.15,
                     y = med_barrier, yend = med_barrier),
                 colour = "#1a1a1a", linewidth = 0.7, inherit.aes = FALSE) +
    geom_hline(yintercept = LOCK, linetype = "dotted", colour = CB$grey, linewidth = 0.8) +
    annotate("text", x = 2, y = LOCK + 0.55, label = "locked\n(barrier < 0.1)",
             hjust = 1, vjust = 0, size = 2.3, colour = "#1a1a1a") +
    scale_colour_manual(values = OPT_COL, guide = "none") +
    labs(title = "Barrier at spawn 8000",
         x = NULL, y = "Inter-child loss barrier") +
    paper_theme()

  # (c) extended Muon forks out to spawn 32000: the barrier only approaches
  #     the no-merge threshold there (near-lock 0.16 in one seed).
  ext_dir <- file.path(res, "lmc_instability_ext")
  ext_df <- bind_rows(lapply(list.files(ext_dir, pattern = "\\.jsonl$",
                                        full.names = TRUE), function(f) {
    su <- last_summary(f)
    bbs <- su[["barrier_by_spawn"]]
    if (is.null(bbs)) return(NULL)
    data.frame(run = gsub("\\.jsonl$", "", basename(f)),
               spawn_step = as.integer(names(bbs)),
               barrier = as.numeric(unlist(bbs)), stringsAsFactors = FALSE)
  }))
  p3 <- ggplot(ext_df, aes(spawn_step, barrier, group = run, colour = run)) +
    geom_line(linewidth = 0.9) + geom_point(size = 2.2) +
    geom_hline(yintercept = LOCK, linetype = "dotted", colour = CB$grey, linewidth = 0.8) +
    scale_colour_manual(values = c(CB$purple, CB$orange, CB$skyblue), guide = "none") +
    scale_x_continuous(labels = label_comma()) +
    scale_y_log10(labels = label_number()) +
    labs(title = "Extended Muon forks (spawn to 32,000)",
         x = "Spawn step", y = "Inter-child loss barrier (log)") +
    paper_theme()

  # (d) five-run short-fork audit. Zero observed k* hits is shown with its
  #     finite-sample upper bound, not as evidence of a zero population rate.
  audit_path <- file.path(res, "figures-deepcheck", "a_lmc_deepcheck_summary.json")
  audit <- read_json_safe(audit_path, sv = FALSE)
  audit_df <- data.frame(
    outcome = factor(c("Observed k*", "95% upper bound"),
                     levels = c("Observed k*", "95% upper bound")),
    rate = c(audit$n_kstar_present / audit$n_combined, audit$zero_success_upper95)
  )
  p4 <- ggplot(audit_df, aes(outcome, rate, fill = outcome)) +
    geom_col(width = 0.58) +
    geom_text(aes(label = c(sprintf("%d/%d", audit$n_kstar_present, audit$n_combined),
                            sprintf("%.3f", audit$zero_success_upper95))),
              vjust = -0.45, size = 2.8, colour = "#1a1a1a") +
    scale_fill_manual(values = c("Observed k*" = CB$vermillion,
                                 "95% upper bound" = CB$grey), guide = "none") +
    scale_y_continuous(limits = c(0, 0.55), labels = label_percent()) +
    labs(title = "Five-run Muon robustness audit",
         subtitle = "0/5 observed; uncertainty remains wide",
         x = NULL, y = "k* hit rate") +
    paper_theme()

  save_both(
    compose_A_four_panel(list(p1, p2, p3, p4), "A_lmc", 7.2, 5.2, root) &
      theme(legend.position = "top"),
    "A_lmc", w = 7.2, h = 5.2
  )
}

# ══════════════════════════════════════════════════════════════════════════════
#  3. A_sink — Sink-triad factorial (§3.3)
#     Data: experiments/results/figures-008/sink_verdicts.json
# ══════════════════════════════════════════════════════════════════════════════
make_A_sink <- function() {
  sv <- read_json_safe(file.path(res, "figures-008", "sink_verdicts.json"))
  opt_files <- c(AdamW = "adamw", Muon = "muon", SGDM = "sgdm")

  read_sink_steps <- function(path) {
    rows <- lapply(readLines(path, warn = FALSE), function(line) {
      rec <- tryCatch(jsonlite::fromJSON(line, simplifyVector = TRUE),
                      error = function(e) NULL)
      if (is.null(rec) || is.null(rec$step)) return(NULL)
      data.frame(step = rec$step, sink_ratio = rec$sink_ratio,
                 stringsAsFactors = FALSE)
    })
    bind_rows(Filter(Negate(is.null), rows))
  }

  # (a) median sink-ratio trajectories across the five pre-norm seeds.
  trajectory_rows <- list()
  onset_rows <- list()
  for (label in names(opt_files)) for (seed in 0:4) {
    path <- file.path(res, "sink_triad",
                      sprintf("%s_pre_s%d.jsonl", opt_files[[label]], seed))
    if (!file.exists(path)) next
    steps <- read_sink_steps(path)
    steps$optimizer <- label
    steps$seed <- seed
    trajectory_rows[[length(trajectory_rows) + 1]] <- steps
    summary <- last_summary(path)
    onset_rows[[length(onset_rows) + 1]] <- data.frame(
      optimizer = label, seed = seed,
      onset = if (is.null(summary$sink_formation_step)) NA_real_ else summary$sink_formation_step,
      stringsAsFactors = FALSE)
  }
  trajectories <- bind_rows(trajectory_rows)
  trajectories$optimizer <- factor(trajectories$optimizer,
                                   levels = c("AdamW", "Muon", "SGDM"))
  trajectory_med <- trajectories %>% group_by(optimizer, step) %>%
    summarise(sink = median(sink_ratio),
              lo = quantile(sink_ratio, 0.25),
              hi = quantile(sink_ratio, 0.75), .groups = "drop")
  p1 <- ggplot(trajectory_med, aes(step, sink, colour = optimizer, fill = optimizer)) +
    geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.16, colour = NA) +
    geom_line(linewidth = 0.85) +
    geom_hline(yintercept = 0.5, linetype = "dotted", colour = CB$grey) +
    scale_colour_manual(values = OPT_COL, name = NULL) +
    scale_fill_manual(values = OPT_COL, name = NULL) +
    coord_cartesian(ylim = c(0, 1.02)) +
    labs(title = "Sink-ratio trajectories (median + IQR)",
         x = "Training step", y = "Sink ratio") +
    paper_theme()

  # (b) per-seed onset; missing threshold crossings remain censored markers.
  onsets <- bind_rows(onset_rows)
  onsets$optimizer <- factor(onsets$optimizer, levels = c("AdamW", "Muon", "SGDM"))
  onsets <- onsets %>% mutate(
    status = ifelse(is.na(onset), "not formed", "formed"),
    plot_onset = ifelse(is.na(onset), 12000, onset))
  p2 <- ggplot(onsets, aes(optimizer, plot_onset, colour = optimizer, shape = status)) +
    stat_summary(data = filter(onsets, status == "formed"), fun = median,
                 geom = "crossbar", width = 0.45, colour = "#1a1a1a") +
    geom_point(position = position_jitter(width = 0.06, height = 0, seed = 12),
               size = 2.3) +
    scale_colour_manual(values = OPT_COL, guide = "none") +
    scale_shape_manual(values = c("formed" = 16, "not formed" = 17), name = NULL) +
    scale_y_log10(labels = label_comma()) +
    labs(title = "Per-seed sink-formation onset",
         subtitle = "Triangles: no threshold crossing by 12,000",
         x = NULL, y = "Formation step (log scale)") +
    paper_theme()

  # (c) learning-rate control. Use the actual sink_triad_lr family.
  lr_dir <- file.path(res, "sink_triad_lr")
  lr_rows <- bind_rows(lapply(list.files(lr_dir, pattern = "\\.jsonl$",
                                        full.names = TRUE), function(path) {
    summary <- last_summary(path)
    optimizer <- ifelse(summary$optimizer == "muon", "Muon", "AdamW")
    eta <- ifelse(summary$optimizer == "muon", summary$muon_lr, summary$lr)
    data.frame(optimizer = optimizer, eta = eta,
               onset = if (is.null(summary$sink_formation_step)) NA_real_ else summary$sink_formation_step,
               final_spike = summary$final_spike_magnitude,
               stringsAsFactors = FALSE)
  }))
  lr_rows$optimizer <- factor(lr_rows$optimizer, levels = c("AdamW", "Muon"))
  p3 <- ggplot(lr_rows, aes(eta, final_spike, colour = optimizer, shape = optimizer)) +
    stat_summary(fun = median, geom = "line", aes(group = optimizer), linewidth = 0.8) +
    geom_point(size = 2.2, alpha = 0.85) +
    scale_x_log10(labels = label_number()) +
    scale_y_log10(labels = label_comma()) +
    # optimizer legend already owned by (a): suppress the (c)/(d) duplicates so
    # the collected legend stays a compact two-row band
    scale_colour_manual(values = OPT_COL, name = NULL, guide = "none") +
    scale_shape_manual(values = c(AdamW = 16, Muon = 17), name = NULL, guide = "none") +
    labs(title = "Learning-rate control",
         x = "Optimizer learning rate", y = "Final spike magnitude (log)") +
    paper_theme()

  # (d) depth localization from the persisted L1/L3 factorial summary.
  depth_metric <- sv$depth$final_spike_magnitude
  depth_rows <- bind_rows(lapply(names(depth_metric), function(key) {
    parts <- strsplit(key, "_")[[1]]
    data.frame(
      optimizer = c(adamw = "AdamW", muon = "Muon", sgdm = "SGDM")[[parts[[1]]]],
      depth = as.integer(sub("L", "", parts[[2]])),
      mean = depth_metric[[key]][[1]], sd = depth_metric[[key]][[2]],
      n = depth_metric[[key]][[3]], stringsAsFactors = FALSE)
  }))
  depth_rows$optimizer <- factor(depth_rows$optimizer,
                                 levels = c("AdamW", "Muon", "SGDM"))
  p4 <- ggplot(depth_rows, aes(factor(depth), mean, colour = optimizer,
                               group = optimizer)) +
    geom_line(linewidth = 0.8) + geom_point(size = 2.4) +
    geom_errorbar(aes(ymin = pmax(mean - sd, 1e-3), ymax = mean + sd),
                  width = 0.15, linewidth = 0.45) +
    scale_colour_manual(values = OPT_COL, name = NULL, guide = "none") +
    scale_y_log10(labels = label_comma()) +
    labs(title = "Activation localization by depth",
         x = "Transformer depth", y = "Final spike magnitude (mean +/- SD)") +
    paper_theme()

  save_both(
    compose_A_four_panel(list(p1, p2, p3, p4), "A_sink", 7.2, 5.4, root) &
      # legend.box="vertical" stacks each collected guide onto its own row so
      # the combined row never exceeds the figure width and clips.
      theme(legend.position = "top", legend.justification = "left",
            legend.box = "vertical"),
    "A_sink", w = 7.2, h = 5.4
  )
}

# ══════════════════════════════════════════════════════════════════════════════
#  4. A_synth — Synthesis composite 5-panel (§4)
#     Data: figures-002 s5_norm_ratio_table.json, figures-021 normctl_verdict.json,
#           figures-012 lmc_kstar_verdict.json, figures-008 sink_verdicts.json,
#           figures-005 plasticity_verdicts.json
# ══════════════════════════════════════════════════════════════════════════════
make_A_synth <- function() {
  s5norm  <- read_json_safe(file.path(res, "figures-002", "s5_norm_ratio_table.json"))
  normctl <- read_json_safe(file.path(res, "figures-021", "normctl_verdict.json"))
  lmc     <- read_json_safe(file.path(res, "figures-012", "lmc_kstar_verdict.json"))
  sv      <- read_json_safe(file.path(res, "figures-008", "sink_verdicts.json"))
  plast   <- read_json_safe(file.path(res, "figures-005", "plasticity_verdicts.json"))

  # Panel 1: Growth signature
  gm  <- s5norm$muon_sc1.0$norm_ratio_mean
  gsd <- s5norm$muon_sc1.0$norm_ratio_std
  df1 <- data.frame(
    optimizer = c("Muon", "AdamW/SGDM"),
    mean = c(gm, 0), sd = c(gsd, 0),
    stringsAsFactors = FALSE
  ) %>% mutate(optimizer = factor(optimizer, levels = c("Muon", "AdamW/SGDM")))
  p1 <- ggplot(df1, aes(optimizer, mean, fill = optimizer)) +
    geom_col(width = 0.55) +
    geom_errorbar(aes(ymin = mean - sd, ymax = mean + sd), width = 0.15,
                  linewidth = 0.45, colour = "#333333") +
    annotate("text", x = 2, y = 1.2, label = "0/20\nevents", size = 2.8,
             colour = "#1a1a1a", vjust = 0) +
    scale_fill_manual(values = c(Muon = CB$vermillion, "AdamW/SGDM" = CB$grey),
                      guide = "none") +
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.25))) +
    labs(title = "S5 growth signature", x = NULL,
         y = "S5 norm ratio\n(grok/init)") +
    paper_theme(8)

  # Panel 2: Causal cap
  k_ord <- c("kinf", "k3", "k2", "k1p5", "k1")
  k_lab <- c(kinf = "∞", k3 = "3", k2 = "2", k1p5 = "1.5", k1 = "1")
  df2 <- data.frame(
    k_idx  = seq_along(k_ord),
    k_lab  = k_lab[k_ord],
    grok   = sapply(k_ord, function(k) normctl$s5$rows[[k]]$grok_step_med),
    stringsAsFactors = FALSE
  ) %>% mutate(k_lab = factor(k_lab, levels = rev(k_lab)))  # invert axis
  p2 <- ggplot(df2, aes(k_lab, grok, group = 1)) +
    geom_line(colour = CB$vermillion, linewidth = 1.3) +
    geom_point(colour = CB$vermillion, size = 3.0) +
    scale_y_log10(labels = label_comma()) +
    labs(title = "Causal cap (S5)",
         x = "Ceiling k", y = "Median grok step (log scale)") +
    paper_theme(8)

  # Panel 3: Basin commitment
  lock_aw <- lmc$by_optimizer$adamw$locked_by_8000_rate
  lock_mu <- lmc$by_optimizer$muon$locked_by_8000_rate
  df3 <- data.frame(optimizer = c("AdamW", "Muon"),
                    rate = c(lock_aw, lock_mu))
  p3 <- ggplot(df3, aes(optimizer, rate, fill = optimizer)) +
    geom_col(width = 0.55) +
    scale_fill_manual(values = OPT_COL, guide = "none") +
    scale_y_continuous(limits = c(0, 1), labels = label_percent()) +
    labs(title = "Basin lock rate", x = NULL,
         y = "Lock-by-8000 rate") +
    paper_theme(8) +
    theme(axis.text.x = element_text(angle = 35, hjust = 1))

  # Panel 4: two bounding diagnostics in one faceted panel. Both are unitless
  # proportions on [0,1], but retain separate x categories and interpretation.
  sr <- sv$main$final_sink_ratio
  wret <- plast$muon_label_refit$weight_retention[1]
  fret <- plast$muon_label_refit$feat_retention[1]
  boundary <- bind_rows(
    data.frame(
      diagnostic = "Sink ratio",
      category = c("AdamW", "Muon", "SGDM"),
      value = c(sr$adamw_pre[1], sr$muon_pre[1], sr$sgdm_pre[1])),
    data.frame(
      diagnostic = "Muon plasticity retention",
      category = c("Weight spectrum", "Feature rank"),
      value = c(wret, fret)))
  boundary$diagnostic <- factor(
    boundary$diagnostic,
    levels = c("Sink ratio", "Muon plasticity retention"))
  boundary$category <- factor(
    boundary$category,
    levels = c("AdamW", "Muon", "SGDM", "Weight spectrum", "Feature rank"))
  boundary_cols <- c(
    AdamW = OPT_COL[["AdamW"]], Muon = OPT_COL[["Muon"]],
    SGDM = OPT_COL[["SGDM"]], `Weight spectrum` = CB$grey,
    `Feature rank` = CB$green)
  p4 <- ggplot(boundary, aes(category, value, fill = category)) +
    geom_col(width = 0.58) +
    geom_text(aes(label = sprintf("%.3f", value)), vjust = -0.35,
              size = 2.4, colour = "#1a1a1a") +
    facet_wrap(~ diagnostic, scales = "free_x", nrow = 1) +
    scale_fill_manual(values = boundary_cols, guide = "none") +
    scale_y_continuous(limits = c(0, 1), expand = expansion(mult = c(0.02, 0.10))) +
    labs(title = "Diagnostic boundary regimes",
         x = NULL, y = "Unitless proportion") +
    paper_theme(8) +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))

  save_both(
    compose_A_four_panel(list(p1, p2, p3, p4), "A_synth", 7.2, 5.4, root),
    "A_synth", w = 7.2, h = 5.4
  )
}

# ══════════════════════════════════════════════════════════════════════════════
#  5. A_case — One S5 seed uncapped vs capped: TIME COURSE (§3.2 case study)
#     Data: experiments/results/s5_normctl/kinf_s0.jsonl, k1_s0.jsonl
#     Single-seed (seed 0) trajectories — no IQR band; cf. the 8-seed median
#     in A_normctl_timecourse. Per-step records: wn_hidden + test_acc vs step.
# ══════════════════════════════════════════════════════════════════════════════
make_A_case <- function() {
  s_dir   <- file.path(res, "s5_normctl")
  GROK_TH <- 0.95   # grok threshold (matches run config grok_thresh)

  # Per-step trajectory for one run (drops _meta / _summary records).
  load_steps <- function(fname) {
    lines <- readLines(file.path(s_dir, fname), warn = FALSE)
    lines <- lines[nzchar(trimws(lines))]
    rows  <- lapply(lines, function(l) {
      txt <- gsub("\\bNaN\\b", "null", l)
      tryCatch(jsonlite::fromJSON(txt, simplifyVector = TRUE), error = function(e) NULL)
    })
    rows <- Filter(function(r) !is.null(r) && is.null(r[["_meta"]]) &&
                              is.null(r[["_summary"]]), rows)
    data.frame(
      step     = sapply(rows, function(r) r[["step"]]),
      test_acc = sapply(rows, function(r) r[["test_acc"]]),
      train_acc = sapply(rows, function(r) if (is.null(r[["train_acc"]])) NA_real_ else r[["train_acc"]]),
      wn_hidden = sapply(rows, function(r) r[["wn_hidden"]]),
      stringsAsFactors = FALSE
    )
  }

  unc <- load_steps("kinf_s0.jsonl") %>% mutate(condition = "Uncapped (k=∞)")
  cap <- load_steps("k1_s0.jsonl")   %>% mutate(condition = "Capped (k=1)")

  # Grok step = first eval where test_acc crosses the threshold (computed
  # from raw records; must reproduce the cited 9750 / 350).
  grok_of <- function(d) {
    hit <- which(d$test_acc >= GROK_TH)
    if (length(hit) == 0) NA_integer_ else d$step[hit[1]]
  }
  unc_grok <- grok_of(unc); cap_grok <- grok_of(cap)
  unc_norm <- tail(unc$wn_hidden, 1); cap_norm <- tail(cap$wn_hidden, 1)
  cat(sprintf("A_case: uncapped grok=%s final_norm=%.4f | capped grok=%s final_norm=%.4f\n",
              unc_grok, unc_norm, cap_grok, cap_norm))

  traj <- bind_rows(unc, cap) %>%
    mutate(condition = factor(condition,
                              levels = c("Uncapped (k=∞)", "Capped (k=1)")))
  cond_col <- c("Uncapped (k=∞)" = CB$grey, "Capped (k=1)" = CB$vermillion)

  grok_df <- data.frame(
    condition = factor(c("Uncapped (k=∞)", "Capped (k=1)"),
                       levels = c("Uncapped (k=∞)", "Capped (k=1)")),
    grok = c(unc_grok, cap_grok)
  )

  # Symlog x-axis so both the early (350) and late (9750) grok read clearly.
  x_scale <- scale_x_continuous(
    trans  = scales::pseudo_log_trans(base = 10, sigma = 80),
    breaks = c(0, 100, 350, 1000, 9750),
    labels = c('0', '100', '350', '1k', '9.75k'),
    expand = expansion(mult = c(0.01, 0.04))
  )

  # (a) hidden norm vs step — uncapped climbs; capped pinned at init scale.
  # Place endpoint value labels in guaranteed-empty zones (above the uncapped
  # peak; above the pinned capped line) so they never sit on a curve.
  peak_unc <- max(unc$wn_hidden, na.rm = TRUE)
  end_lab <- data.frame(
    condition = grok_df$condition,
    step = c(tail(unc$step, 1), tail(cap$step, 1)),
    norm = c(peak_unc + 30, cap_norm + 22),
    lab  = sprintf("%.1f", c(unc_norm, cap_norm))
  )
  p1 <- ggplot(traj, aes(step, wn_hidden, colour = condition)) +
    geom_vline(data = grok_df, aes(xintercept = grok, colour = condition),
               linetype = "dashed", linewidth = 0.6, show.legend = FALSE) +
    geom_line(linewidth = 1.0) +
    geom_text(data = end_lab, aes(step, norm, label = lab),
              hjust = 1.0, size = 2.7, colour = "black", show.legend = FALSE) +
    x_scale +
    scale_y_continuous(expand = expansion(mult = c(0.02, 0.10))) +
    scale_colour_manual(values = cond_col, name = NULL) +
    labs(title = "Hidden-weight norm",
         x = "Training step (symlog)",
         y = expression(group("||", W[hidden], "||"))) +
    paper_theme() +
    theme(legend.position = "top")

  # (b) validation accuracy vs step — capped groks early, uncapped late.
  p2 <- ggplot(traj, aes(step, test_acc, colour = condition)) +
    geom_hline(yintercept = GROK_TH, linetype = "dotted",
               colour = CB$grey, linewidth = 0.7) +
    geom_vline(data = grok_df, aes(xintercept = grok, colour = condition),
               linetype = "dashed", linewidth = 0.6, show.legend = FALSE) +
    geom_line(linewidth = 1.0) +
    annotate("text", x = cap_grok, y = 0.40,
             label = paste0("groks @ ", scales::comma(cap_grok)),
             hjust = -0.10, size = 2.6, colour = "black") +
    annotate("text", x = unc_grok, y = 0.28,
             label = paste0("groks @ ", scales::comma(unc_grok)),
             hjust = 1.12, size = 2.6, colour = "black") +
    annotate("text", x = 0, y = GROK_TH + 0.045,
             label = "grok threshold 0.95",
             hjust = 0, vjust = 0, size = 2.4, colour = "black") +
    x_scale +
    scale_y_continuous(limits = c(0, 1.05),
                       expand = expansion(mult = c(0.02, 0.04))) +
    scale_colour_manual(values = cond_col, name = NULL) +
    labs(title = "Validation accuracy",
         x = "Training step (symlog)",
         y = "Validation accuracy") +
    paper_theme() +
    theme(legend.position = "top")

  # (c) grok-point zooms: the transition shape in a window around each grok step
  zoom_of <- function(d, g, half) d %>% filter(step >= max(0, g - half), step <= g + half)
  zcap <- zoom_of(cap, cap_grok, 400) %>% mutate(pane = sprintf("Capped: groks @ %s", scales::comma(cap_grok)))
  zunc <- zoom_of(unc, unc_grok, 2500) %>% mutate(pane = sprintf("Uncapped: groks @ %s", scales::comma(unc_grok)))
  Z <- bind_rows(zcap, zunc)
  Z$pane <- factor(Z$pane, levels = c(sprintf("Capped: groks @ %s", scales::comma(cap_grok)),
                                      sprintf("Uncapped: groks @ %s", scales::comma(unc_grok))))
  p3 <- ggplot(Z, aes(step, test_acc, colour = condition)) +
    geom_hline(yintercept = GROK_TH, linetype = "dotted", colour = CB$grey, linewidth = 0.6) +
    geom_line(linewidth = 0.9) +
    facet_wrap(~ pane, scales = "free_x", nrow = 1) +
    scale_colour_manual(values = cond_col, guide = "none") +
    coord_cartesian(ylim = c(0, 1.02)) +
    labs(title = "Transition shape at the grok point",
         x = "Training step", y = "Validation accuracy") +
    paper_theme() +
    theme(axis.text.x = element_text(size = 7, angle = 35, hjust = 1))

  # (d) generalization gap: train minus validation accuracy over training
  G <- traj %>% mutate(gap = train_acc - test_acc) %>% filter(!is.na(gap))
  p4 <- ggplot(G, aes(step, gap, colour = condition)) +
    geom_hline(yintercept = 0, linetype = "dotted", colour = CB$grey, linewidth = 0.6) +
    geom_vline(data = grok_df, aes(xintercept = grok, colour = condition),
               linetype = "dashed", linewidth = 0.6, show.legend = FALSE) +
    geom_line(linewidth = 0.9) +
    x_scale +
    scale_colour_manual(values = cond_col, name = NULL) +
    labs(title = "Generalization gap (train − validation)",
         x = "Training step (symlog)", y = "Accuracy gap") +
    paper_theme() +
    theme(legend.position = "top")

  save_both(
    ((p1 | p2) / (p3 | p4) + plot_layout(guides = "collect") & theme(legend.position = "top")) +
      tag_annotation(),
    "A_case", w = 7.2, h = 5.2
  )
}

# ══════════════════════════════════════════════════════════════════════════════
#  6. A_floor — Optimizer floor composite (§3.6)
#     Data: hardcoded 019 Stage-B table + sgdm_cliff/*.jsonl
# ══════════════════════════════════════════════════════════════════════════════
make_A_floor <- function() {
  groups  <- c("Z60", "D30", "A5", "Z120", "D60", "S5")

  # Read the complete 019 Stage-B ladder. Panel (a) is derived from these rows;
  # no published count is duplicated as a hard-coded table.
  gc_dir <- file.path(res, "group_complexity")
  gc_rows <- list()
  for (g in groups) for (o in c("adamw", "muon", "sgdm")) {
    for (s in 0:4) {
      f <- file.path(gc_dir, sprintf("ladder_%s_%s_d256_s%d.jsonl", g, o, s))
      if (!file.exists(f)) next
      su <- last_summary(f)
      gc_rows[[length(gc_rows) + 1]] <- data.frame(
        group = g, opt = c(adamw = "AdamW", muon = "Muon", sgdm = "SGDM")[[o]],
        seed = s, acc = su$final_test_acc,
        grok_step = if (!is.null(su$grok_step) && !is.na(su$grok_step)) su$grok_step else NA_real_,
        stringsAsFactors = FALSE)
    }
  }
  GC <- bind_rows(gc_rows)
  GC$group <- factor(GC$group, levels = groups)
  GC$opt <- factor(GC$opt, levels = c("AdamW", "Muon", "SGDM"))
  df_lad <- GC %>% group_by(group, opt) %>%
    summarise(rate = mean(acc >= 0.95), .groups = "drop") %>%
    rename(optimizer = opt)

  # SGDM groks 0/5 on every rung, so its bars have zero height and are
  # visually indistinguishable from missing data. Tag each SGDM slot with an
  # explicit "0/5" so a measured-zero never reads as a gap in the sweep.
  zero_lab <- df_lad %>% mutate(lab = ifelse(optimizer == "SGDM", "0/5", ""))
  p_lad <- ggplot(df_lad, aes(group, rate, fill = optimizer)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7) +
    geom_text(data = zero_lab, aes(group, 0.045, label = lab, group = optimizer),
              position = position_dodge(width = 0.8), vjust = 0, size = 1.9,
              colour = CB$grey, show.legend = FALSE) +
    geom_vline(xintercept = 3.5, colour = "#cccccc", linetype = "dotted") +
    scale_fill_manual(values = OPT_COL, name = NULL) +
    scale_y_continuous(labels = label_percent(), limits = c(0, 1.05)) +
    labs(title = "De-confounded group ladder",
         subtitle = "Muon 5/5 on every rung; AdamW 5/5 except A5 (3/5); SGDM 0/5 everywhere (floor)",
         x = NULL, y = "Grok rate (of 5 seeds)") +
    paper_theme(8) +
    theme(legend.position = "top",
          plot.subtitle = element_text(size = 6.8, colour = "#444444"))

  # (b) per-seed final accuracy: the raw distribution behind the rates in (a)
  p_b <- ggplot(GC, aes(group, acc, colour = opt)) +
    geom_hline(yintercept = 0.95, linetype = "dotted", colour = CB$grey, linewidth = 0.6) +
    geom_point(position = position_jitterdodge(jitter.width = 0.10, dodge.width = 0.8,
                                               seed = 4),
               size = 1.7, alpha = 0.85) +
    scale_colour_manual(values = OPT_COL, name = NULL, guide = "none") +
    coord_cartesian(ylim = c(0, 1.05)) +
    labs(title = "Per-seed final accuracy (5 seeds/cell)",
         x = NULL, y = "Final test accuracy") +
    paper_theme(8) +
    theme(legend.position = "none")

  # (c) time-to-grok by rung: the ladder orders difficulty, the optimizer orders speed
  Tg <- GC %>% filter(!is.na(grok_step))
  p_c <- ggplot(Tg, aes(group, grok_step, colour = opt)) +
    stat_summary(fun = median, geom = "crossbar", width = 0.55,
                 position = position_dodge(width = 0.8), colour = "black") +
    geom_point(position = position_jitterdodge(jitter.width = 0.10, dodge.width = 0.8,
                                               seed = 4),
               size = 1.5, alpha = 0.8) +
    scale_colour_manual(values = OPT_COL, name = NULL, guide = "none") +
    scale_y_log10(labels = label_comma()) +
    labs(title = "Time-to-grok by rung (median + seeds)",
         x = NULL, y = "Grok step (log scale)") +
    paper_theme(8) +
    theme(legend.position = "none")

  # (d) no-decay AdamW control from the actual nowd_<group>_* file family.
  nd_dir <- file.path(res, "group_complexity_nowd")
  nd_rows <- list()
  for (g in groups) {
    fs <- list.files(nd_dir, pattern = sprintf("^nowd_%s_.*\\.jsonl$", g),
                     full.names = TRUE)
    n_grok <- 0L
    for (f in fs) {
      su <- last_summary(f)
      if (!is.null(su$final_test_acc) && !is.na(su$final_test_acc) &&
          su$final_test_acc >= 0.95) n_grok <- n_grok + 1L
    }
    if (length(fs)) nd_rows[[length(nd_rows) + 1]] <- data.frame(
      group = g, rate = n_grok / length(fs), n = length(fs), stringsAsFactors = FALSE)
  }
  ND <- bind_rows(nd_rows)
  ND$group <- factor(ND$group, levels = groups)
  aw_rate <- df_lad %>% filter(optimizer == "AdamW") %>% select(group, rate)
  D4 <- ND %>% mutate(arm = "AdamW, wd = 0") %>%
    bind_rows(aw_rate %>% mutate(arm = "AdamW (wd = 0.01)"))
  p_d <- ggplot(D4, aes(group, rate, fill = arm)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.68) +
    geom_vline(xintercept = 3.5, colour = "#cccccc", linetype = "dotted") +
    scale_fill_manual(values = c("AdamW (wd = 0.01)" = CB$blue,
                                 "AdamW, wd = 0" = CB$skyblue), name = NULL) +
    scale_y_continuous(labels = label_percent(), limits = c(0, 1.05)) +
    labs(title = "No-decay AdamW: A5/S5 fail, other rungs 5/5",
         x = NULL, y = "Grok rate") +
    paper_theme(8) +
    theme(legend.position = "top")

  save_both(
    compose_A_four_panel(list(p_lad, p_b, p_c, p_d), "A_floor", 7.2, 5.4, root),
    "A_floor", w = 7.2, h = 5.4
  )
}


# ══════════════════════════════════════════════════════════════════════════════
#  7. A_plasticity — Plasticity dissociation scatter (§3.5)
#     Data: experiments/results/figures-005/plasticity_verdicts.json
# ══════════════════════════════════════════════════════════════════════════════
make_A_plasticity <- function() {
  pv <- read_json_safe(file.path(res, "figures-005", "plasticity_verdicts.json"))

  opt_disp <- c(adamw = "AdamW", muon = "Muon", sgdm = "SGDM")
  df <- bind_rows(lapply(names(pv), function(nm) {
    parts <- strsplit(nm, "_")[[1]]
    opt   <- ifelse(parts[1] %in% names(opt_disp), opt_disp[parts[1]],
                    tools::toTitleCase(parts[1]))
    arm   <- paste(parts[-1], collapse = "_")
    data.frame(optimizer = opt, arm = arm,
               weight_ret = pv[[nm]]$weight_retention[1],
               feat_ret   = pv[[nm]]$feat_retention[1],
               n_fit      = pv[[nm]]$n_fit[1],
               stringsAsFactors = FALSE)
  })) %>%
    mutate(marker = ifelse(arm == "proj_shift", "proj_shift", "label_refit"),
           pt_size = 2.5 + 8 * n_fit / 100)

  p <- ggplot(df, aes(weight_ret, feat_ret, colour = optimizer, shape = marker)) +
    geom_point(aes(size = n_fit), alpha = 0.78, stroke = 0.6) +
    scale_colour_manual(values = OPT_COL, name = "Optimizer") +
    scale_shape_manual(values = c(proj_shift = 16, label_refit = 17),
                       labels = c(proj_shift = "Projection shift",
                                  label_refit = "Label refit"),
                       name = "Arm") +
    scale_size_continuous(range = c(2, 7), name = "# fittable") +
    guides(size = guide_legend(nrow = 2, byrow = TRUE, order = 3),
           colour = guide_legend(order = 1),
           shape = guide_legend(order = 2)) +
    scale_x_continuous(limits = c(0.85, 1.02)) +
    scale_y_continuous(limits = c(-0.02, 0.07)) +
    labs(title = "Weight spectrum vs feature rank retention",
         x = "Weight eff-rank retention (last/first)",
         y = "Feature eff-rank retention (last/first)") +
    paper_theme() +
    theme(legend.position = "right",
          legend.text = element_text(size = 7))

  # (b) retention gap (weight − feature) by optimizer: the dissociation as a
  #     per-arm quantity rather than a scatter position
  dg <- df %>% mutate(gap = weight_ret - feat_ret,
                      arm_lab = paste(optimizer, marker, sep = "\n"))
  p2 <- ggplot(dg, aes(optimizer, gap, colour = optimizer)) +
    geom_hline(yintercept = 0, linetype = "dotted", colour = CB$grey) +
    stat_summary(fun = median, geom = "crossbar", width = 0.45, colour = "#1a1a1a") +
    geom_point(position = position_jitter(width = 0.07, height = 0, seed = 3),
               size = 2.2, alpha = 0.85, show.legend = FALSE) +
    scale_colour_manual(values = OPT_COL, guide = "none") +
    labs(title = "Retention gap (weight - feature)",
         x = NULL, y = "Retention gap") +
    paper_theme()

  # (c) number of fittable tasks per arm (sample-size honesty: which cells the
  #     dissociation actually covers)
  p3 <- ggplot(df, aes(optimizer, n_fit, fill = optimizer)) +
    geom_col(position = position_dodge(width = 0.75), width = 0.68,
             aes(group = marker), colour = NA) +
    geom_text(aes(label = n_fit, group = marker),
              position = position_dodge(width = 0.75), vjust = -0.4, size = 2.5,
              colour = "#1a1a1a", show.legend = FALSE) +
    scale_fill_manual(values = OPT_COL, guide = "none") +
    labs(title = "Fittable tasks per arm",
         x = NULL, y = "# tasks fittable") +
    scale_y_continuous(expand = expansion(mult = c(0.05, 0.15))) +
    paper_theme()

  # (d) benchmark boundary: on permuted-MNIST the synthetic plasticity ordering
  #     does not transfer (endpoint feature rank, 150-task stream)
  read_arm_end <- function(subdir, pat, seeds) {
    vals <- c()
    for (s in seeds) {
      f <- file.path(res, subdir, sprintf(pat, s))
      if (!file.exists(f)) next
      lines <- readLines(f, warn = FALSE); lines <- lines[nzchar(trimws(lines))]
      tasks <- Filter(Negate(is.null), lapply(lines, function(l)
        tryCatch(jsonlite::fromJSON(l, simplifyVector = TRUE), error = function(e) NULL)))
      tasks <- Filter(function(r) !is.null(r$task) && !is.null(r$probes), tasks)
      if (length(tasks)) vals <- c(vals, tasks[[length(tasks)]]$probes$feat_eff_rank)
    }
    vals
  }
  mnist <- bind_rows(
    data.frame(opt = "AdamW", fr = read_arm_end("muon_plasticity_mnist", "adamw_s%d.jsonl", 0:2)),
    data.frame(opt = "Muon",  fr = read_arm_end("muon_plasticity_mnist", "muon_s%d.jsonl", 0:2)),
    data.frame(opt = "SGDM",  fr = read_arm_end("muon_plasticity_mnist", "sgdm_s%d.jsonl", 0:2)),
    data.frame(opt = "Muon (scaled lr)",
               fr = c(read_arm_end("muon_plasticity_mnist_lrctl", "muon_mlr0p005_s%d.jsonl", 0:4),
                      read_arm_end("muon_plasticity_perm_mnist_ext", "muon_mlr0p005_s%d.jsonl", 0:4))))
  mnist$opt <- factor(mnist$opt, levels = c("AdamW", "Muon", "SGDM", "Muon (scaled lr)"))
  mnist_cols <- c(AdamW = OPT_COL[["AdamW"]], Muon = OPT_COL[["Muon"]],
                  SGDM = OPT_COL[["SGDM"]], `Muon (scaled lr)` = CB$green)
  p4 <- ggplot(mnist, aes(opt, fr, colour = opt)) +
    stat_summary(fun = median, geom = "crossbar", width = 0.45, colour = "#1a1a1a") +
    geom_point(position = position_jitter(width = 0.06, height = 0, seed = 3),
               size = 2.2, alpha = 0.85, show.legend = FALSE) +
    scale_colour_manual(values = mnist_cols, guide = "none") +
    labs(title = "Boundary: permuted-MNIST endpoint rank",
         x = NULL, y = "Feature effective rank (final task)") +
    paper_theme() +
    theme(axis.text.x = element_text(angle = 20, hjust = 1))

  save_both(
    compose_A_four_panel(list(p, p2, p3, p4), "A_plasticity", 7.2, 5.4, root) &
      theme(legend.position = "top"),
    "A_plasticity", w = 7.2, h = 5.4
  )
}

# ══════════════════════════════════════════════════════════════════════════════
#  8. A_plasticity_scale — Perm-MNIST plasticity scale figure (§3.5)
#     Data: experiments/results/muon_plasticity_mnist/**/*.jsonl
# ══════════════════════════════════════════════════════════════════════════════
make_A_plasticity_scale <- function() {
  # same glob logic as make_A_plasticity_scale.py
  trend <- function(fp) {
    lines <- readLines(fp, warn = FALSE)
    lines <- lines[nzchar(trimws(lines))]
    tasks <- lapply(lines[grepl('"task"', lines)], function(l) {
      tryCatch(jsonlite::fromJSON(l, simplifyVector = TRUE), error = function(e) NULL)
    })
    tasks <- Filter(Negate(is.null), tasks)
    if (length(tasks) == 0) return(NULL)
    fits  <- sapply(tasks, function(r) {
      v <- r[["steps_to_threshold"]]
      if (is.null(v)) NA_real_ else as.numeric(v)
    })
    first <- median(fits[1:min(10, length(fits))], na.rm = TRUE)
    last_v <- fits[max(1, length(fits) - 9):length(fits)]
    last  <- median(last_v, na.rm = TRUE)
    last_r <- tasks[[length(tasks)]]
    list(first = first, last = last,
         frL = last_r$probes$feat_eff_rank,
         dL  = last_r$probes$dead_frac)
  }

  collect <- function(glob_pat) {
    files <- Sys.glob(glob_pat)
    Filter(Negate(is.null), lapply(files, trend))
  }

  groups_raw <- list(
    "AdamW\n(lr 1e-3)"   = collect(file.path(res, "muon_plasticity_mnist", "adamw_s*.jsonl")),
    "SGDM\n(lr 0.02)"    = collect(file.path(res, "muon_plasticity_mnist", "sgdm_s*.jsonl")),
    "Muon\n(lr 0.005)"   = c(
      collect(file.path(res, "muon_plasticity_mnist_lrctl", "muon_mlr0p005_s*.jsonl")),
      collect(file.path(res, "muon_plasticity_perm_mnist_ext", "muon_mlr0p005_s*.jsonl"))
    ),
    "Muon\n(lr 0.01)"    = c(
      collect(file.path(res, "muon_plasticity_mnist_lrctl", "muon_mlr0p01_s*.jsonl")),
      collect(file.path(res, "muon_plasticity_perm_mnist_ext", "muon_mlr0p01_s*.jsonl"))
    ),
    "Muon\n(lr 0.02)"    = collect(file.path(res, "muon_plasticity_mnist", "muon_s*.jsonl"))
  )

  order <- c("AdamW\n(lr 1e-3)", "SGDM\n(lr 0.02)",
             "Muon\n(lr 0.005)", "Muon\n(lr 0.01)", "Muon\n(lr 0.02)")

  agg_df <- bind_rows(lapply(order, function(k) {
    v <- groups_raw[[k]]
    if (length(v) == 0) return(NULL)
    slow <- median(sapply(v, function(x) x$last / x$first), na.rm = TRUE)
    rank <- median(sapply(v, function(x) x$frL), na.rm = TRUE)
    dead <- median(sapply(v, function(x) x$dL),  na.rm = TRUE)
    data.frame(group = k, slowdown = slow, feat_rank = rank, dead_frac = dead,
               n = length(v), stringsAsFactors = FALSE)
  })) %>% mutate(group = factor(group, levels = order),
                 ref_slow = first(slowdown))  # AdamW reference

  # Colours: blue, grey, then vermillion gradient for 3 Muon lrs
  bar_cols <- c(CB$blue, CB$grey, "#f0a050", "#e0701f", CB$vermillion)
  names(bar_cols) <- order

  adamw_slow <- agg_df$slowdown[1]

  p1 <- ggplot(agg_df, aes(group, slowdown, fill = group)) +
    geom_col(width = 0.7) +
    geom_hline(yintercept = adamw_slow, linetype = "dotted",
               colour = CB$blue, linewidth = 0.8) +
    annotate("text", x = 5.5, y = adamw_slow, label = "AdamW baseline",
             hjust = 1, vjust = -0.5, size = 2.4, colour = "black") +
    scale_fill_manual(values = bar_cols, guide = "none") +
    labs(title = "Fit-speed slowdown",
         x = NULL, y = "Slowdown (last 10 vs first 10 tasks)") +
    paper_theme(8) +
    theme(axis.text.x = element_text(angle = 35, hjust = 1))

  p2 <- ggplot(agg_df, aes(group, feat_rank, fill = group)) +
    geom_col(width = 0.7) +
    geom_text(aes(y = feat_rank, label = sprintf("%.0f%% dead", 100 * dead_frac)),
              vjust = -0.4, size = 2.3, colour = "#1a1a1a") +
    scale_fill_manual(values = bar_cols, guide = "none") +
    labs(title = "Final feature effective rank",
         x = NULL, y = "Feature effective rank") +
    paper_theme(8) +
    theme(axis.text.x = element_text(angle = 35, hjust = 1))

  # per-seed frame for panels (c)-(d)
  seed_df <- bind_rows(lapply(order, function(k) {
    v <- groups_raw[[k]]
    if (length(v) == 0) return(NULL)
    bind_rows(lapply(seq_along(v), function(i) {
      data.frame(group = k, seed = i - 1,
                 slowdown = v[[i]]$last / v[[i]]$first,
                 feat_rank = v[[i]]$frL, dead_frac = v[[i]]$dL,
                 stringsAsFactors = FALSE)
    }))
  })) %>% mutate(group = factor(group, levels = order))

  p3 <- ggplot(seed_df, aes(group, slowdown, colour = group)) +
    geom_hline(yintercept = adamw_slow, linetype = "dotted",
               colour = CB$blue, linewidth = 0.7) +
    stat_summary(fun = median, geom = "crossbar", width = 0.45, colour = "#1a1a1a") +
    geom_point(position = position_jitter(width = 0.07, height = 0, seed = 12),
               size = 2.0, alpha = 0.85, show.legend = FALSE) +
    scale_colour_manual(values = bar_cols, guide = "none") +
    labs(title = "Per-seed slowdown",
         x = NULL, y = "Slowdown (last 10 / first 10)") +
    paper_theme(8) +
    theme(axis.text.x = element_text(angle = 35, hjust = 1))

  p4 <- ggplot(seed_df, aes(group, dead_frac, colour = group)) +
    stat_summary(fun = median, geom = "crossbar", width = 0.45, colour = "#1a1a1a") +
    geom_point(position = position_jitter(width = 0.07, height = 0, seed = 12),
               size = 2.0, alpha = 0.85, show.legend = FALSE) +
    scale_colour_manual(values = bar_cols, guide = "none") +
    scale_y_continuous(labels = label_percent()) +
    labs(title = "Dead-unit fraction per seed",
         x = NULL, y = "Dead fraction (final task)") +
    paper_theme(8) +
    theme(axis.text.x = element_text(angle = 35, hjust = 1))

  save_both(
    (p1 | p2) / (p3 | p4) + tag_annotation() & theme(legend.position = "top"),
    "A_plasticity_scale", w = 7.2, h = 5.4
  )
}

# ══════════════════════════════════════════════════════════════════════════════
#  9. A_two_route — Two-route model fit (§3.1)
#     Data: experiments/results/wd_sweep/adamw_wd*.jsonl (fitted in R)
# ══════════════════════════════════════════════════════════════════════════════
make_A_two_route <- function() {
  files <- list.files(file.path(res, "wd_sweep"), pattern = "^adamw_wd.*\\.jsonl$",
                      full.names = TRUE)
  pts <- bind_rows(lapply(files, function(fp) {
    lines <- readLines(fp, warn = FALSE)
    lines <- lines[nzchar(trimws(lines))]
    rows  <- lapply(lines, function(l)
      tryCatch(jsonlite::fromJSON(l, simplifyVector = TRUE), error = function(e) NULL))
    s     <- Filter(function(r) !is.null(r[["_summary"]]), rows)
    if (length(s) == 0) return(NULL)
    sm <- s[[length(s)]][["_summary"]]
    if (is.null(sm$grok_step) || is.null(sm$memorize_step)) return(NULL)
    data.frame(lam = sm$weight_decay,
               T   = sm$grok_step - sm$memorize_step,
               stringsAsFactors = FALSE)
  })) %>% filter(!is.na(T), T > 0)

  cat(sprintf("A_two_route: n=%d AdamW pts; lambda range [%.4f, %.2f]\n",
              nrow(pts), min(pts$lam), max(pts$lam)))

  pos <- pts %>% filter(lam > 0)

  # M1: single power law (lambda>0 only)
  m1_fit <- tryCatch(nls(log(T) ~ log(A) - alpha * log(lam), data = pos,
                         start = list(A = 100, alpha = 0.5)), error = function(e) NULL)
  m1_A     <- if (!is.null(m1_fit)) coef(m1_fit)[["A"]]     else NA
  m1_alpha <- if (!is.null(m1_fit)) coef(m1_fit)[["alpha"]] else NA

  # M2: two-route (all pts incl lambda=0)
  m2_fit <- tryCatch(nls(T ~ 1 / (1/D0 + lam/C), data = pts,
                         start = list(D0 = 5000, C = 500),
                         control = nls.control(maxiter = 500)), error = function(e) NULL)
  m2_D0 <- if (!is.null(m2_fit)) coef(m2_fit)[["D0"]] else NA
  m2_C  <- if (!is.null(m2_fit)) coef(m2_fit)[["C"]]  else NA

  # Plot range: show lambda=0 as small positive placeholder near the smallest
  # positive lambda so the log-x spacing reads evenly (data value unchanged).
  plot_pts <- pts %>% mutate(lam_plot = ifelse(lam == 0, 5e-4, lam))
  lam_seq  <- 10^seq(log10(5e-4), 0, length.out = 200)

  curve_df <- bind_rows(
    if (!is.null(m1_fit))
      data.frame(lam = lam_seq,
                 T   = m1_A * lam_seq^(-m1_alpha),
                 model = sprintf("M1 power law (α=%.2f)", m1_alpha)) else NULL,
    if (!is.null(m2_fit))
      data.frame(lam = lam_seq,
                 T   = 1 / (1/m2_D0 + lam_seq/m2_C),
                 model = sprintf("M2 two-route (D₀=%.0f, C=%.0f)", m2_D0, m2_C)) else NULL
  )

  model_cols <- c()
  if (!is.null(m1_fit)) model_cols[sprintf("M1 power law (α=%.2f)", m1_alpha)] <- "#333333"
  if (!is.null(m2_fit)) model_cols[sprintf("M2 two-route (D₀=%.0f, C=%.0f)", m2_D0, m2_C)] <- CB$vermillion

  p <- ggplot() +
    geom_point(data = plot_pts, aes(lam_plot, T, shape = "per-run delay"),
               colour = CB$blue, alpha = 0.65, size = 2.2) +
    {if (nrow(curve_df) > 0)
      geom_line(data = curve_df, aes(lam, T, colour = model, linetype = model),
                linewidth = 0.9)} +
    scale_x_log10(labels = label_scientific()) +
    scale_y_log10(labels = label_comma()) +
    scale_colour_manual(values = model_cols, name = NULL) +
    scale_linetype_manual(values = c("dotted", "solid"), name = NULL) +
    scale_shape_manual(values = c("per-run delay" = 16), name = NULL) +
    guides(shape    = guide_legend(order = 1,
                                   override.aes = list(colour = CB$blue, alpha = 1)),
           colour   = guide_legend(order = 2, nrow = 2),
           linetype = guide_legend(order = 2, nrow = 2)) +
    labs(x = expression("Weight decay " * lambda * "  (" * lambda * "=0 at left edge)"),
         y = expression("Realized delay " * T[grok] - T[mem] * " (steps)")) +
    paper_theme() +
    theme(legend.position = "bottom")

  # (b) T_mem / T_grok decomposition: weight decay moves the memorization
  #     endpoint little and the grokking endpoint a lot (the delay is grok-side)
  decomp <- bind_rows(lapply(files, function(fp) {
    lines <- readLines(fp, warn = FALSE)
    lines <- lines[nzchar(trimws(lines))]
    rows  <- lapply(lines, function(l)
      tryCatch(jsonlite::fromJSON(l, simplifyVector = TRUE), error = function(e) NULL))
    s     <- Filter(function(r) !is.null(r[["_summary"]]), rows)
    if (length(s) == 0) return(NULL)
    sm <- s[[length(s)]][["_summary"]]
    if (is.null(sm$grok_step) || is.null(sm$memorize_step)) return(NULL)
    data.frame(lam = sm$weight_decay, T_mem = sm$memorize_step,
               T_grok = sm$grok_step, stringsAsFactors = FALSE)
  }))
  dec_long <- decomp %>%
    mutate(lam_plot = ifelse(lam == 0, 5e-4, lam)) %>%
    pivot_longer(c(T_mem, T_grok), names_to = "qty", values_to = "step")
  p2 <- ggplot(dec_long, aes(lam_plot, step, colour = qty)) +
    geom_point(alpha = 0.6, size = 1.8) +
    scale_colour_manual(values = c(T_mem = CB$grey, T_grok = CB$blue),
                        labels = c(T_mem = expression(T[mem]), T_grok = expression(T[grok])),
                        name = NULL) +
    scale_x_log10(labels = label_scientific()) +
    scale_y_log10(labels = label_comma()) +
    labs(title = "Decomposition: decay moves the grok endpoint",
         x = expression("Weight decay " * lambda),
         y = "Step (log scale)") +
    paper_theme() +
    theme(legend.position = "top")

  # (c) model residuals: observed / fitted for M1 and M2
  res_df <- bind_rows(
    if (!is.null(m1_fit))
      pos %>% mutate(fit = m1_A * lam^(-m1_alpha), model = "M1 power law") else NULL,
    if (!is.null(m2_fit))
      pts %>% mutate(fit = 1 / (1/m2_D0 + lam/m2_C), model = "M2 two-route") else NULL
  ) %>% mutate(ratio = T / fit)
  p3 <- ggplot(res_df, aes(lam, ratio, colour = model)) +
    geom_hline(yintercept = 1, linetype = "dotted", colour = CB$grey) +
    geom_point(alpha = 0.65, size = 1.8) +
    scale_colour_manual(values = c("M1 power law" = "#333333",
                                   "M2 two-route" = CB$vermillion), name = NULL) +
    scale_x_log10(breaks = c(1e-3, 1e-2, 1e-1, 1),
                  labels = c("1e-03", "1e-02", "1e-01", "1e+00")) +
    scale_y_log10() +
    labs(title = "Fit residuals (observed / fitted)",
         x = expression("Weight decay " * lambda), y = "Ratio (log scale)") +
    paper_theme() +
    theme(legend.position = "top",
          axis.text.x = element_text(size = 7, angle = 25, hjust = 1),
          plot.margin = margin(5.5, 8, 12, 8))

  # (d) the other route: Muon on the same sweep shows no such delay law
  mfiles <- list.files(file.path(res, "wd_sweep"), pattern = "^muon_wd.*\\.jsonl$",
                       full.names = TRUE)
  mpts <- bind_rows(lapply(mfiles, function(fp) {
    lines <- readLines(fp, warn = FALSE)
    lines <- lines[nzchar(trimws(lines))]
    rows  <- lapply(lines, function(l)
      tryCatch(jsonlite::fromJSON(l, simplifyVector = TRUE), error = function(e) NULL))
    s     <- Filter(function(r) !is.null(r[["_summary"]]), rows)
    if (length(s) == 0) return(NULL)
    sm <- s[[length(s)]][["_summary"]]
    if (is.null(sm$grok_step) || is.null(sm$memorize_step)) return(NULL)
    data.frame(lam = sm$weight_decay,
               T   = sm$grok_step - sm$memorize_step, stringsAsFactors = FALSE)
  })) %>% filter(!is.na(T))
  p4 <- ggplot(mpts %>% mutate(lam_plot = ifelse(lam == 0, 5e-4, lam)),
               aes(lam_plot, T)) +
    geom_point(colour = CB$vermillion, alpha = 0.65, size = 1.9) +
    scale_x_log10(labels = label_scientific()) +
    scale_y_log10(labels = label_comma(), expand = expansion(mult = c(0.28, 0.08))) +
    labs(title = "Muon on the same sweep: no delay law",
         x = expression("Weight decay " * lambda),
         y = expression(T[grok] - T[mem] * " (steps, log)")) +
    paper_theme() +
    theme(plot.margin = margin(5.5, 8, 8, 8))

  save_both(
    (p | p2) / (p3 | p4) + tag_annotation() & theme(legend.position = "top"),
    "A_two_route", w = 7.2, h = 5.4
  )
}

# ══════════════════════════════════════════════════════════════════════════════
# 10. A_lmc_extension — Red-team LMC fork audits
#     Data: experiments/results/ultragoal_20260619_redteam/a_lmc_short/summaries.json
#           experiments/results/ultragoal_20260619_fullattack/a_lmc_extension/summaries.json
#           experiments/results/figures-redteam/redteam_a_lmc_stats.json
# ══════════════════════════════════════════════════════════════════════════════
make_A_lmc_extension <- function() {
  red_dir  <- file.path(res, "ultragoal_20260619_redteam")
  full_dir <- file.path(res, "ultragoal_20260619_fullattack")
  evid_dir <- file.path(root, "experiments","results","figures-redteam")

  red_data  <- read_json_safe(file.path(red_dir, "a_lmc_short", "summaries.json"),
                               sv = FALSE)
  full_path <- file.path(full_dir, "a_lmc_extension", "summaries.json")
  full_data <- if (file.exists(full_path))
    read_json_safe(full_path, sv = FALSE) else list()
  # unwrap "completed" wrapper if present
  if (!is.null(full_data$completed)) full_data <- full_data$completed

  rows <- c(
    lapply(full_data, function(x) c(list(source = "extension sweep"), x)),
    lapply(red_data,  function(x) c(list(source = "short audit"),     x))
  )

  df <- bind_rows(lapply(rows, function(x) {
    bb <- x$barrier_by_spawn
    if (is.null(bb)) return(NULL)
    tibble(
      source     = x$source,
      seed       = as.integer(x$seed),
      spawn_step = as.integer(names(bb)),
      barrier    = as.numeric(unlist(bb)),
      k_star     = if (is.null(x$k_star)) NA_real_ else as.numeric(x$k_star)
    )
  }))

  st    <- read_json_safe(file.path(evid_dir, "redteam_a_lmc_stats.json"))
  upper <- st$combined$zero_kstar_upper95
  n_tot <- nrow(distinct(df, source, seed))
  absent <- nrow(filter(distinct(df, source, seed, k_star), is.na(k_star)))

  df <- mutate(df, run_lab = sprintf("%s, seed %d", source, seed))
  p1 <- ggplot(df, aes(spawn_step, barrier,
                        colour = run_lab, linetype = run_lab,
                        group = run_lab)) +
    geom_line(linewidth = 0.85) +
    geom_point(size = 2.2) +
    annotate("text", x = max(df$spawn_step), y = 5.05,
             label = "all barriers ≫ 0.1 (no-merge threshold)",
             hjust = 1, size = 2.6, colour = "black") +
    scale_colour_manual(values = c("#2563eb", "#7c3aed", "#0e7490", "#b45309"),
                        name = NULL) +
    scale_linetype_manual(values = c("solid", "22", "42", "13"), name = NULL) +
    scale_y_continuous(limits = c(5.0, 5.6),
                       expand = expansion(mult = c(0.02, 0.05))) +
    # One legend entry per row: the long run labels ("extension sweep, seed 10")
    # collided with the next key glyph in the collected top legend (typesetter
    # feedback 2026-07-16). Extra right-margin on legend text as belt-and-braces.
    guides(colour   = guide_legend(nrow = 2, byrow = TRUE),
           linetype = guide_legend(nrow = 2, byrow = TRUE)) +
    labs(title = "Fork barriers (no k* merge observed)",
         x = "Spawn step", y = "Max LMC barrier") +
    paper_theme() +
    theme(legend.text = element_text(margin = margin(l = 5, r = 10)))

  bound_df <- tibble(
    metric = factor(c(sprintf("no merges\n(%d/%d seeds)", absent, n_tot),
                      "Clopper-Pearson\n95% upper"),
                    levels = c(sprintf("no merges\n(%d/%d seeds)", absent, n_tot),
                               "Clopper-Pearson\n95% upper")),
    value  = c(absent / n_tot, upper),
    label  = c(sprintf("%d/%d", absent, n_tot), sprintf("%.3f", upper))
  )

  p2 <- ggplot(bound_df, aes(metric, value, fill = metric)) +
    geom_col(width = 0.55, colour = "#1f2937", linewidth = 0.25, show.legend = FALSE) +
    geom_text(aes(label = label), vjust = -0.45, size = 3) +
    scale_fill_manual(values = c("#2563eb", "#f97316")) +
    scale_y_continuous(labels = label_percent(accuracy = 1), limits = c(0, 1.1),
                       expand = expansion(mult = c(0, 0.02))) +
    labs(title = "Boundary-grade non-hit",
         x = NULL, y = "Proportion") +
    paper_theme()

  save_both(
    ((p1 | p2) + plot_layout(guides = "collect") & theme(legend.position = "top")) +
      tag_annotation(),
    "A_lmc_extension", w = 5.0, h = 3.4
  )
}

# ══════════════════════════════════════════════════════════════════════════════
# 11. A_s5_route — S5 route map (§3.1)
#     Source: experiments/results/figures-002/fig_s5_route_map.png
#             (identical copy; data requires GPU-run s5_mech trajectory
#              processing not achievable from summary JSONs alone)
#     Strategy: Re-read raw s5_mech/*.jsonl + mech/*.jsonl and reproduce
#               the route map scatter in R.
# ══════════════════════════════════════════════════════════════════════════════
make_A_s5_route <- function() {
  s5_dir  <- file.path(res, "s5_mech")
  add_dir <- file.path(res, "mech")

  load_route <- function(dir_path, label) {
    files <- list.files(dir_path, pattern = "\\.jsonl$", full.names = TRUE)
    bind_rows(lapply(files, function(fp) {
      lines <- readLines(fp, warn = FALSE)
      lines <- lines[nzchar(trimws(lines))]
      parsed <- lapply(lines, function(l)
        tryCatch(jsonlite::fromJSON(l, simplifyVector = TRUE), error = function(e) NULL))
      parsed <- Filter(Negate(is.null), parsed)

      meta <- Filter(function(r) !is.null(r[["_meta"]]),    parsed)
      summ <- Filter(function(r) !is.null(r[["_summary"]]), parsed)
      hist <- Filter(function(r)  is.null(r[["_meta"]]) && is.null(r[["_summary"]]), parsed)

      if (length(meta) == 0 || length(summ) == 0 || length(hist) == 0) return(NULL)
      mt <- meta[[1]][["_meta"]]; sm <- summ[[length(summ)]][["_summary"]]
      T_mem <- sm[["memorize_step"]]; T_grok <- sm[["grok_step"]]
      if (is.null(T_mem) || is.null(T_grok) || is.na(T_mem) || is.na(T_grok)) return(NULL)

      # find hist record closest to T_grok
      steps   <- sapply(hist, function(h) h[["step"]])
      h_grok  <- hist[[which.min(abs(steps - T_grok))]]
      h_mem   <- hist[[which.min(abs(steps - T_mem))]]

      cosm <- h_grok[["cos_mem"]]
      wn_m <- h_mem[["wn_hidden"]];  wn_g <- h_grok[["wn_hidden"]]
      if (is.null(cosm) || is.null(wn_m) || is.null(wn_g) ||
          is.na(cosm) || is.na(wn_m) || is.na(wn_g) || wn_m == 0) return(NULL)

      cosm  <- max(-1, min(1, cosm))
      angle <- acos(cosm) * 180 / pi
      dlog  <- log(wn_g / wn_m)
      sc    <- mt[["init_scale"]] %||% 1.0

      data.frame(
        dataset   = label,
        optimizer = tools::toTitleCase(mt[["optimizer"]]),
        init_scale = sc,
        dlog_norm = dlog,
        angle_deg = angle,
        stringsAsFactors = FALSE
      )
    }))
  }

  `%||%` <- function(x, y) if (is.null(x)) y else x

  df_s5  <- load_route(s5_dir,  "S5")
  df_add <- load_route(add_dir, "mod-add")

  if (nrow(df_s5) == 0) {
    warning("A_s5_route: no s5_mech data parsed — restoring original")
    file.copy(file.path(figdir, "backup_matplotlib", "A_s5_route.png"),
              file.path(figdir, "A_s5_route.png"), overwrite = TRUE)
    cat("A_s5_route: BLOCKED (no parseable trajectory data) — original restored\n")
    return(invisible(NULL))
  }

  # Encode every distinguishing aesthetic through a scale so it is legended:
  #   colour = optimizer (AdamW / Muon / SGDM, incl. the faded grey SGDM pts)
  #   size   = dataset   (S5 large vs modular-addition small/faded background)
  #   shape  = init scale (1.0 circle vs 3.0 triangle, the lone triangle in S5)
  #   alpha  = dataset    (decorative, redundant with size → no separate legend)
  df <- bind_rows(
    df_add %>% mutate(dataset = "Modular addition"),
    df_s5  %>% mutate(dataset = "S5")
  )
  df$optimizer <- dplyr::recode(tolower(df$optimizer),
                                adamw = "AdamW", muon = "Muon", sgdm = "SGDM")
  df$optimizer <- factor(df$optimizer, levels = c("AdamW", "Muon", "SGDM"))
  df$dataset   <- factor(df$dataset, levels = c("S5", "Modular addition"))
  df$init_lab  <- factor(ifelse(df$init_scale == 3.0, "3.0", "1.0"),
                         levels = c("1.0", "3.0"))
  # Draw the faded modular-addition background first, S5 markers on top.
  df <- bind_rows(filter(df, dataset == "Modular addition"),
                  filter(df, dataset == "S5"))

  p <- ggplot(df, aes(dlog_norm, angle_deg, colour = optimizer,
                      shape = init_lab, size = dataset, alpha = dataset)) +
    geom_vline(xintercept = 0, colour = "#999999", linewidth = 0.5) +
    geom_point(stroke = 0.6) +
    scale_colour_manual(values = OPT_COL, name = "Optimizer", drop = FALSE) +
    scale_shape_manual(values = c("1.0" = 16, "3.0" = 17),
                       name = "Init scale") +
    scale_size_manual(values = c("S5" = 3.0, "Modular addition" = 1.5),
                      name = "Dataset") +
    scale_alpha_manual(values = c("S5" = 0.85, "Modular addition" = 0.22),
                       guide = "none") +
    guides(
      colour = guide_legend(order = 1,
                            override.aes = list(size = 3, alpha = 1, shape = 16)),
      size   = guide_legend(order = 2, override.aes = list(alpha = 1, shape = 16)),
      shape  = guide_legend(order = 3, override.aes = list(size = 3, alpha = 1))
    ) +
    labs(
      title = "S5 vs modular addition route map",
      x        = expression(Delta * "log hidden norm  (" * T[mem] * "→" * T[grok] * ")"),
      y        = expression("Angular distance (deg, " * T[mem] * "→" * T[grok] * ")")
    ) +
    paper_theme() +
    # Panel (a) carries three horizontal top legends (Optimizer/Dataset/Init
    # scale). The compose step below forces legend.position="top" on every
    # panel; centred over this half-width panel the legend row is wider than
    # the panel and overflows the figure's LEFT edge, clipping "Ada" off
    # "AdamW". Left-anchor the legend block so it grows rightward into the
    # canvas (panel-b has no legend) instead of off the left edge.
    theme(legend.position = "top", legend.justification = "left")

  # (b)-(c) marginals on S5: the route separation is bivariate — no single
  #         marginal alone separates the optimizer families
  s5_only <- df %>% filter(dataset == "S5")
  p_b <- ggplot(s5_only, aes(optimizer, dlog_norm, colour = optimizer)) +
    geom_hline(yintercept = 0, linetype = "dotted", colour = CB$grey) +
    stat_summary(fun = median, geom = "crossbar", width = 0.45, colour = "#1a1a1a") +
    geom_point(position = position_jitter(width = 0.07, height = 0, seed = 14),
               size = 2.2, alpha = 0.85, show.legend = FALSE) +
    scale_colour_manual(values = OPT_COL, guide = "none") +
    labs(title = "S5 marginal: norm route",
         x = NULL, y = expression(Delta * "log hidden norm")) +
    paper_theme()

  p_c <- ggplot(s5_only, aes(optimizer, angle_deg, colour = optimizer)) +
    stat_summary(fun = median, geom = "crossbar", width = 0.45, colour = "#1a1a1a") +
    geom_point(position = position_jitter(width = 0.07, height = 0, seed = 14),
               size = 2.2, alpha = 0.85, show.legend = FALSE) +
    scale_colour_manual(values = OPT_COL, guide = "none") +
    labs(title = "S5 marginal: angular route",
         x = NULL, y = "Angular distance (deg)") +
    paper_theme()

  # (d) update-geometry control: on S5 only spectrum flattening groks;
  #     on mod-add every structured update groks and geometry only sets speed
  ag3 <- read_json_safe(file.path(root, "experiments", "revision2026", "gpu2026",
                                  "ag3", "ag3_verdict.json"))
  ag3_rows <- bind_rows(lapply(names(ag3$per_op), function(task) {
    bind_rows(lapply(names(ag3$per_op[[task]]), function(arm) {
      x <- ag3$per_op[[task]][[arm]]
      data.frame(task = task, arm = arm, rate = x$n_grok / x$n,
                 stringsAsFactors = FALSE)
    }))
  }))
  arm_lv <- c("muon", "specflat", "randorth", "specinv", "adamw")
  ag3_rows$arm <- factor(ag3_rows$arm, levels = arm_lv[arm_lv %in% ag3_rows$arm])
  arm_cols <- c(muon = CB$vermillion, specflat = CB$orange, randorth = CB$skyblue,
                specinv = CB$purple, adamw = CB$blue)
  p_d <- ggplot(ag3_rows, aes(arm, rate, fill = arm)) +
    geom_col(width = 0.68) +
    geom_text(aes(label = paste0(round(rate * 5), "/5")), vjust = -0.4, size = 2.6,
              colour = "#1a1a1a") +
    facet_wrap(~ task, nrow = 1) +
    scale_fill_manual(values = arm_cols, guide = "none") +
    coord_cartesian(ylim = c(0, 1.1)) +
    labs(title = "Geometry control: only flattening groks S5",
         x = NULL, y = "Grok rate") +
    paper_theme() +
    theme(axis.text.x = element_text(angle = 25, hjust = 1))

  save_both(
    (p | p_b) / (p_c | p_d) + tag_annotation() &
      theme(legend.position = "top", legend.justification = "left",
            legend.box = "vertical"),
    "A_s5_route", w = 7.2, h = 5.6
  )
}

# ══════════════════════════════════════════════════════════════════════════════
#  Run all figures
# ══════════════════════════════════════════════════════════════════════════════
cat("=== figures: R/ggplot2 render ===\n\n")

run_fig <- function(name, fn) {
  cat(sprintf("--- %s ---\n", name))
  tryCatch(fn(), error = function(e) {
    cat(sprintf("ERROR in %s: %s\n  Restoring original\n", name, conditionMessage(e)))
    src <- file.path(figdir, "backup_matplotlib", paste0(name, ".png"))
    dst <- file.path(figdir, paste0(name, ".png"))
    if (file.exists(src)) file.copy(src, dst, overwrite = TRUE)
  })
}

run_fig("A_normctl",               make_A_normctl)
run_fig("A_lmc",                   make_A_lmc)
run_fig("A_sink",                  make_A_sink)
run_fig("A_synth",                 make_A_synth)
run_fig("A_case",                  make_A_case)
run_fig("A_floor",                 make_A_floor)
run_fig("A_plasticity",            make_A_plasticity)
run_fig("A_plasticity_scale",      make_A_plasticity_scale)
run_fig("A_two_route",             make_A_two_route)
run_fig("A_lmc_extension", make_A_lmc_extension)
run_fig("A_s5_route",              make_A_s5_route)

cat("\n=== done ===\n")
cat(sprintf("PNGs in %s\n", figdir))
cat(sprintf("SVGs in %s\n", evdir))
