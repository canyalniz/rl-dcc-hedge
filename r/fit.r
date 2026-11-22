library(rugarch)
library(WeightedPortTest)
library(FinTS)
library(rmgarch)
library(xts)
library(checkmate)
library(parallel)
library(fs)
library(arrow)

source('portfolio_modeling_functions.r')

fit_to_seed <- function(
    weekly_returns,
    modeling_end_date='2022-01-01',
    subjects=NULL,
    model_save_path='fitted_model.RData'
){
    train_boundaries <- get_train_boundaries(cutoff = modeling_end_date, weekly_returns = weekly_returns)

    fit_start = train_boundaries[1]
    fit_end = train_boundaries[2]

    ugarch_configs <- list(
        new_ugarch_config(
            model='eGARCH',
            ARCH_lags=1,
            GARCH_lags=1,
            AR_lags=0,
            MA_lags=0,
            dist='sstd'
        ),
        new_ugarch_config(
            model='eGARCH',
            ARCH_lags=1,
            GARCH_lags=1,
            AR_lags=0,
            MA_lags=0,
            dist='sstd'
        )
    )

    # Univariate Specs
    uspecs <- list(
        get_spec(ugarch_configs[[1]]),
        get_spec(ugarch_configs[[2]])
    )

    model_config <- new_mgarch_config(
        uspecs=uspecs,
        model='DCC',
        dccOrder=c(1,1),
        distribution='mvt',
        solver=c("hybrid", "solnp"),
        subjects=subjects
    )

    model <- fit_model(model_config=model_config, weekly_returns=weekly_returns, fit_start = fit_start, fit_end = fit_end, probe_lags = c(8,13,26))

    if(!is.null(model_save_path)){
        save(model, file=model_save_path)
    }

    return(model)
}

generate_trajectories <- function(model, starting_prices, sim_steps=209, m.sim=100, n.start=52, seed=42, startMethod='unconditional', mc.cores=4){
    subjects <- colnames(model$z)

    if(length(seed) == 1){
        rseed <- seed:(seed + m.sim - 1)
    } else if (length(seed) == m.sim) {
        rseed <- seed
    } else {
        stop(paste0(seed, " is not a valid seed for m.sim=", m.sim))
    }

    n.sim <- sim_steps + 1
    sim <- dccsim(
        fitORspec = model$fit,
        n.sim = n.sim,
        m.sim = m.sim,
        n.start = n.start,
        startMethod = startMethod,
        rseed = rseed
    )

    spec <- get_fixed_param_dccspec(model)

    modeldata <- model$fit@model$modeldata$data
    T_model <- dim(modeldata)[1]

    results <- mclapply(
        1:(m.sim),
        FUN = function(i){
            simX <- sim@msim$simX[[i]]
            colnames(simX) <- subjects
            rownames(simX) <- NULL # delete row names to save memory

            filtdata <- rbind(modeldata, simX)

            filt <- dccfilter(
                spec=spec,
                data=filtdata,
                filter.control = list(n.old=T_model)
            )

            returns <- filtdata[(T_model):(T_model+sim_steps),] # (sim_steps + 1) elements, p_t <- p_{t-1}*exp(r_t)
            colnames(returns) <- lapply(
                seq_along(subjects),
                function(j){
                    paste0(i-1,"_",j-1,"_returns")
                }
            )

            innovations <- filt@mfilter$stdresid
            innovations <- innovations[(T_model):(T_model+sim_steps),] # (sim_steps + 1) elements, aligned to returns, r_t <- mu + H_t * z_t
            colnames(innovations) <- lapply(
                seq_along(subjects),
                function(j){
                    paste0(i-1,"_",j-1,"_innov")
                }
            )

            cond_std <- coredata(sigma(filt))
            cond_std <- cond_std[(T_model+1):(T_model+sim_steps+1),] # (sim_steps + 1) elements, shifted to align as predictions
            colnames(cond_std) <- lapply(
                seq_along(subjects),
                function(j){
                    paste0(i-1,"_",j-1,"_condstd")
                }
            )

            cond_corr <- as.numeric(rcor(filt)[1,2,])
            cond_corr <- cond_corr[(T_model+1):(T_model+sim_steps+1)] # (sim_steps + 1) elements, shifted to align as predictions

            prices <- matrix(nrow = sim_steps + 1, ncol = length(subjects))
            for(j in seq_along(subjects)){
                subject <- subjects[[j]]
                prices[1,j] = starting_prices[[subject]]
                prices[2:(sim_steps+1),j] = exp(cumsum(simX[1:sim_steps,subject])) * starting_prices[[subject]]
            }
            colnames(prices) <- lapply(
                seq_along(subjects),
                function(j){
                    paste0(i-1,"_",j-1,"_prices")
                }
            )

            result_matrix <- cbind(returns, prices, innovations, cond_std, cond_corr)
            colnames(result_matrix)[
                length(colnames(result_matrix))
            ] <- paste0(i-1,"_condcorr") # set the cond_corr column name

            return(result_matrix)
        },
        mc.cores = mc.cores
    )
    
    results <- do.call(
        cbind,
        results
    ) # concatenate the results into a matrix

    return(results)
}


create_test_partition <- function(
    model,
    starting_prices,
    weekly_returns,
    test_start_date,
    subjects,
    save_path
){
    filtdata <- weekly_returns[subjects]

    T_model = sum(weekly_returns$date < test_start_date)
    T_test = length(weekly_returns$date) - T_model

    spec <- get_fixed_param_dccspec(model)

    filt <- dccfilter(
        spec=spec,
        data=filtdata,
        filter.control = list(n.old=T_model)
    )

    test_returns <- filtdata[(T_model):(T_model+T_test),] # (T_test + 1) elements
    colnames(test_returns) <- lapply(
        subjects,
        function(subject){
            paste0(subject,"_returns")
        }
    )

    innovations <- filt@mfilter$stdresid
    innovations <- innovations[(T_model):(T_model+T_test),] # (T_test + 1) elements, aligned to returns, r_t <- mu + H_t * z_t
    colnames(innovations) <- lapply(
        subjects,
        function(subject){
            paste0(subject,"_innov")
        }
    )

    cond_std <- matrix(nrow = T_test+1, ncol = length(subjects))
    filt_cond_std <- coredata(sigma(filt))
    cond_std[1:T_test,] <- filt_cond_std[(T_model+1):(T_model+T_test),] # (T_test) elements, shifted to align as predictions
    cond_std[T_test+1] <- -1 # last value will not be used by the environment or the agent, here as a placeholder
    colnames(cond_std) <- lapply(
        subjects,
        function(subject){
            paste0(subject,"_condstd")
        }
    )

    cond_corr <- numeric(T_test+1)
    filt_cond_corr <- as.numeric(rcor(filt)[1,2,])
    cond_corr[1:T_test] <- filt_cond_corr[(T_model+1):(T_model+T_test)] # (T_test) elements, shifted to align as predictions
    cond_corr[T_test+1] <- -1 # last value will not be used by the environment or the agent, here as a placeholder


    prices <- matrix(nrow = T_test + 1, ncol = length(subjects))
    for(j in seq_along(subjects)){
        subject <- subjects[[j]]
        prices[1,j] = starting_prices[[subject]]
        prices[2:(T_test+1),j] = exp(cumsum(filtdata[(T_model+1):(T_model+T_test),subject])) * starting_prices[[subject]]
    }
    colnames(prices) <- lapply(
        subjects,
        function(subject){
            paste0(subject,"_prices")
        }
    )

    dates <- weekly_returns[(T_model):(T_model+T_test),"date"]

    result_matrix <- cbind(test_returns, prices, innovations, cond_std, cond_corr, dates)
    colnames(result_matrix)[
        length(colnames(result_matrix)) - 1
    ] <- "condcorr" # set the cond_corr column name
    colnames(result_matrix)[
        length(colnames(result_matrix))
    ] <- "dates" # set the cond_corr column name
    result_df <- as.data.frame(result_matrix)

    write_parquet(
        result_df,
        save_path
    )
}

create_simulation_data <- function(
    model,
    starting_prices,
    sim_steps,
    m.sim,
    n.start,
    seed,
    startMethod,
    mc.cores,
    save_path
){
    simulation_results <- generate_trajectories(
        model=model,
        starting_prices=starting_prices,
        sim_steps=sim_steps,
        m.sim=m.sim,
        n.start=n.start,
        seed=seed,
        startMethod=startMethod,
        mc.cores=mc.cores
    )

    df <- as.data.frame(simulation_results)
    
    save_dir <- dirname(save_path) # get save directory
    filename <- tools::file_path_sans_ext(basename(save_path)) # get filename without extension
    i <- 1 # duplicate cound
    fexists <- file.exists(save_path)
    while(fexists){
        print(paste("WARNING:", save_path, "exists in the filesystem. Saving as next up duplicate."))
        dupl_fname <- paste0(filename, "_", i)
        save_path <- path(save_dir, dupl_fname, ext = 'parquet') # construct new path
        fexists <- file.exists(save_path) # check if new path exists
        i <- i + 1
    }
    write_parquet(df, save_path)
}

process_seed_df <- function(args){
    weekly_returns <- read_weekly_return_data(path=args[['seed_df_path']], mod=TRUE, format=args[['seed_df_format']])

    subjects <- args[['subjects']]
    if(is.null(subjects)){
        cn <- colnames(weekly_returns)
        subjects <- cn[cn!='date']
    } else {
        subjects <- as.character(subjects)
    }
    model <- fit_to_seed(
        weekly_returns=weekly_returns,
        modeling_end_date=args[['modeling_end_date']],
        subjects=subjects,
        model_save_path=args[['model_save_path']]
    )
    buffer <- create_test_partition(
        model=model,
        starting_prices=args[['starting_prices']],
        weekly_returns=weekly_returns,
        test_start_date=args[['testing_start_date']],
        subjects=subjects,
        save_path=args[['test_partition_save_path']]
    )
    # training
    buffer <- create_simulation_data(
        model=model,
        starting_prices=args[['starting_prices']],
        sim_steps=args[['sim_steps']],
        m.sim=args[['m_sim']],
        n.start=args[['n_start']],
        seed=args[['train_sim_seed']],
        startMethod=args[['startMethod']],
        mc.cores=args[['mc_cores']],
        save_path=args[['train_sim_save_path']]
    )
    # eval
    buffer <- create_simulation_data(
        model=model,
        starting_prices=args[['starting_prices']],
        sim_steps=args[['eval_sim_steps']],
        m.sim=args[['eval_m_sim']],
        n.start=args[['eval_n_start']],
        seed=args[['eval_sim_seed']],
        startMethod=args[['startMethod']],
        mc.cores=args[['mc_cores']],
        save_path=args[['eval_sim_save_path']]
    )
    # synth test
    buffer <- create_simulation_data(
        model=model,
        starting_prices=args[['starting_prices']],
        sim_steps=args[['test_sim_steps']],
        m.sim=args[['test_m_sim']],
        n.start=args[['test_n_start']],
        seed=args[['test_sim_seed']],
        startMethod=args[['startMethod']],
        mc.cores=args[['mc_cores']],
        save_path=args[['test_sim_save_path']]
    )
}

generate_training_data <- function(args){
    weekly_returns <- read_weekly_return_data(path=args[['seed_df_path']], mod=TRUE, format=args[['seed_df_format']])

    subjects <- args[['subjects']]
    if(is.null(subjects)){
        cn <- colnames(weekly_returns)
        subjects <- cn[cn!='date']
    } else {
        subjects <- as.character(subjects)
    }
    model <- fit_to_seed(
        weekly_returns=weekly_returns,
        modeling_end_date=args[['modeling_end_date']],
        subjects=subjects,
        model_save_path=args[['model_save_path']]
    )
    # training
    buffer <- create_simulation_data(
        model=model,
        starting_prices=args[['starting_prices']],
        sim_steps=args[['sim_steps']],
        m.sim=args[['m_sim']],
        n.start=args[['n_start']],
        seed=args[['train_sim_seed']],
        startMethod=args[['startMethod']],
        mc.cores=args[['mc_cores']],
        save_path=args[['train_sim_save_path']]
    )
}

generate_evaluation_data <- function(args){
    weekly_returns <- read_weekly_return_data(path=args[['seed_df_path']], mod=TRUE, format=args[['seed_df_format']])

    subjects <- args[['subjects']]
    if(is.null(subjects)){
        cn <- colnames(weekly_returns)
        subjects <- cn[cn!='date']
    } else {
        subjects <- as.character(subjects)
    }
    model <- fit_to_seed(
        weekly_returns=weekly_returns,
        modeling_end_date=args[['modeling_end_date']],
        subjects=subjects,
        model_save_path=args[['model_save_path']]
    )
    # eval
    buffer <- create_simulation_data(
        model=model,
        starting_prices=args[['starting_prices']],
        sim_steps=args[['eval_sim_steps']],
        m.sim=args[['eval_m_sim']],
        n.start=args[['eval_n_start']],
        seed=args[['eval_sim_seed']],
        startMethod=args[['startMethod']],
        mc.cores=args[['mc_cores']],
        save_path=args[['eval_sim_save_path']]
    )
}

generate_synth_test_data <- function(args){
    weekly_returns <- read_weekly_return_data(path=args[['seed_df_path']], mod=TRUE, format=args[['seed_df_format']])

    subjects <- args[['subjects']]
    if(is.null(subjects)){
        cn <- colnames(weekly_returns)
        subjects <- cn[cn!='date']
    } else {
        subjects <- as.character(subjects)
    }
    model <- fit_to_seed(
        weekly_returns=weekly_returns,
        modeling_end_date=args[['modeling_end_date']],
        subjects=subjects,
        model_save_path=args[['model_save_path']]
    )
    # synth test
    buffer <- create_simulation_data(
        model=model,
        starting_prices=args[['starting_prices']],
        sim_steps=args[['test_sim_steps']],
        m.sim=args[['test_m_sim']],
        n.start=args[['test_n_start']],
        seed=args[['test_sim_seed']],
        startMethod=args[['startMethod']],
        mc.cores=args[['mc_cores']],
        save_path=args[['test_sim_save_path']]
    )
}

generate_testing_partition <- function(args){
    weekly_returns <- read_weekly_return_data(path=args[['seed_df_path']], mod=TRUE, format=args[['seed_df_format']])

    subjects <- args[['subjects']]
    if(is.null(subjects)){
        cn <- colnames(weekly_returns)
        subjects <- cn[cn!='date']
    } else {
        subjects <- as.character(subjects)
    }
    model <- fit_to_seed(
        weekly_returns=weekly_returns,
        modeling_end_date=args[['modeling_end_date']],
        subjects=subjects,
        model_save_path=args[['model_save_path']]
    )
    buffer <- create_test_partition(
        model=model,
        starting_prices=args[['starting_prices']],
        weekly_returns=weekly_returns,
        test_start_date=args[['testing_start_date']],
        subjects=subjects,
        save_path=args[['test_partition_save_path']]
    )
}

## Bootstrap guardrails
bootstrap_thresholds <- function(args){
    # guardrail thresholds of the bootstrap pipeline, args[['b_*']] override the defaults
    # (defaults mirror configs/data_config.py SimDataConfig)
    thr <- list(
        b_max_attempts = 64,           # bootstrap samples drawn at most to get num_bootstrapped_models accepted models
        b_max_traj_retries = 10,       # resimulations of a rejected trajectory before the model is rejected
        b_max_abs_return_sds = 20,     # A, F: max |weekly log-return| in seed return sds
        b_min_sd_ratio = 1/3,          # A: bootstrap sample return sd / seed return sd
        b_max_sd_ratio = 3,
        b_bound_tol = 1e-4,            # B: parameter within b_bound_tol * max(1, |bound|) of an optimizer bound
        b_max_beta = 0.995,            # C: eGARCH |beta1|
        b_min_shape = 3,               # C: univariate sstd shape
        b_min_skew = 0.5,              # C: univariate sstd skew
        b_max_skew = 2,
        b_max_dcc_persistence = 0.995, # C: dcca1 + dccb1
        b_min_dcca = 1e-4,             # C: dcca1, ~0 means constant correlation and unidentified dccb1
        b_min_mshape = 4.1,            # C: joint mvt shape
        b_max_unc_corr = 0.99,         # C: |unconditional correlation|
        b_min_uncvol_ratio = 0.5,      # D: unconditional vol / seed unconditional vol
        b_max_uncvol_ratio = 2,
        b_max_unc_corr_diff = 0.4,     # D: |unconditional correlation - seed unconditional correlation|
        b_resid_pval = 0.01,           # E: residual test p-value below which a model is flagged
        b_max_condstd_ratio = 5,       # F: max condstd / max seed in-sample condstd
        b_max_cum_return_sds = 8,      # F: |cumulative log-return - seed drift| in seed sds * sqrt(T)
        b_min_innov_sd = 1/3,          # F: sd of the innovations filtered through the seed model
        b_max_innov_sd = 3,
        b_warn_rejection_rate = 0.25   # warn when more than this share of the attempts is rejected
    )
    for(name in names(thr)){
        if(!is.null(args[[name]])){
            thr[[name]] <- as.numeric(args[[name]])
        }
    }
    return(thr)
}

bootstrap_reference <- function(model){
    # seed model statistics the bootstrap guardrails are measured against
    subjects <- colnames(model$z)
    modeldata <- model$fit@model$modeldata$data

    # rugarch only fills in the univariate parameter bounds at fit time, they are taken
    # from univariate fits to the seed data (bounds only depend on the spec, mu's on the data scale)
    ubounds <- lapply(
        seq_along(subjects),
        function(j){
            ufit <- ugarchfit(spec = model$config$uspecs[[j]], data = modeldata[,j], solver = 'hybrid')
            pars <- ufit@model$pars
            pars[pars[,'Include'] == 1, c('LB', 'UB'), drop = FALSE]
        }
    )

    return(list(
        subjects = subjects,
        desc = describe_dcc_model(model, subjects),
        ret_sd = unname(apply(modeldata, 2, sd)),
        ret_mean = unname(colMeans(modeldata)),
        max_condstd = unname(apply(coredata(sigma(model$fit)), 2, max)),
        ubounds = ubounds
    ))
}

validate_bootstrap_sample <- function(simX, ref, thr){
    # stage A: the bootstrap sample a model gets refit to
    fail <- function(reason) list(ok = FALSE, stage = 'A', reason = reason)

    if(!all(is.finite(simX))){
        return(fail('non-finite returns'))
    }
    for(j in seq_along(ref$subjects)){
        s <- ref$subjects[[j]]
        max_abs_sds <- max(abs(simX[,j])) / ref$ret_sd[[j]]
        if(max_abs_sds > thr$b_max_abs_return_sds){
            return(fail(sprintf('%s max |return| = %.1f seed sds > %g', s, max_abs_sds, thr$b_max_abs_return_sds)))
        }
        sd_ratio <- sd(simX[,j]) / ref$ret_sd[[j]]
        if(sd_ratio < thr$b_min_sd_ratio || sd_ratio > thr$b_max_sd_ratio){
            return(fail(sprintf('%s return sd / seed return sd = %.2f outside [%.2f, %.2f]', s, sd_ratio, thr$b_min_sd_ratio, thr$b_max_sd_ratio)))
        }
    }
    return(list(ok = TRUE, stage = NA_character_, reason = NA_character_))
}

validate_dcc_fit <- function(model, ref, thr, probe_lags=c(8,13,26)){
    # stages B (refit validity), C (parameter space), D (plausibility against the seed) reject,
    # stage E (residual diagnostics) and missing standard errors only add soft flags
    fit <- model$fit
    subjects <- ref$subjects
    flags <- character(0)
    desc <- tryCatch(describe_dcc_model(model, subjects), error = function(e) NULL)
    result <- function(ok, stage = NA_character_, reason = NA_character_){
        list(ok = ok, stage = stage, reason = reason, flags = paste(flags, collapse = ';'), desc = desc)
    }

    # B: refit validity
    if(fit@mfit$convergence != 0){
        return(result(FALSE, 'B', paste('convergence code', fit@mfit$convergence)))
    }
    if(!is.finite(fit@mfit$llh)){
        return(result(FALSE, 'B', 'non-finite llh'))
    }
    cf <- coef(fit)
    if(!all(is.finite(cf))){
        return(result(FALSE, 'B', paste('non-finite coefficients:', paste(names(cf)[!is.finite(cf)], collapse = ', '))))
    }
    if(!all(is.finite(fit@mfit$matcoef[,2]))){
        flags <- c(flags, 'se_nonfinite')
    }

    ubounds <- lapply(
        seq_along(subjects),
        function(j){
            b <- ref$ubounds[[j]]
            rownames(b) <- paste0('[', subjects[[j]], '].', rownames(b))
            return(b)
        }
    )
    jbounds <- fit@model$pars
    jbounds <- jbounds[jbounds[,'Include'] == 1, c('LB', 'UB'), drop = FALSE]
    rownames(jbounds) <- paste0('[Joint]', rownames(jbounds))
    bounds <- do.call(rbind, c(ubounds, list(jbounds)))[names(cf), , drop = FALSE]
    near_bound <- function(b) abs(cf - b) <= thr$b_bound_tol * pmax(1, abs(b))
    hits <- names(cf)[which(near_bound(bounds[,'LB']) | near_bound(bounds[,'UB']))]
    if(length(hits) > 0){
        return(result(FALSE, 'B', paste('at optimizer bound:', paste(sprintf('%s = %.6g', hits, cf[hits]), collapse = ', '))))
    }

    # C: parameter space / stationarity
    for(j in seq_along(subjects)){
        s <- subjects[[j]]
        p <- function(name) desc[[paste0('a', j-1, '_', name)]]
        if(abs(p('beta1')) >= thr$b_max_beta){
            return(result(FALSE, 'C', sprintf('%s |beta1| = %.4f >= %g', s, abs(p('beta1')), thr$b_max_beta)))
        }
        if(p('shape') <= thr$b_min_shape){
            return(result(FALSE, 'C', sprintf('%s shape = %.3f <= %g', s, p('shape'), thr$b_min_shape)))
        }
        if(p('skew') <= thr$b_min_skew || p('skew') >= thr$b_max_skew){
            return(result(FALSE, 'C', sprintf('%s skew = %.3f outside (%g, %g)', s, p('skew'), thr$b_min_skew, thr$b_max_skew)))
        }
    }
    if(desc$dcc_persistence >= thr$b_max_dcc_persistence){
        return(result(FALSE, 'C', sprintf('dcca1 + dccb1 = %.4f >= %g', desc$dcc_persistence, thr$b_max_dcc_persistence)))
    }
    if(desc$dcca1 <= thr$b_min_dcca){
        return(result(FALSE, 'C', sprintf('dcca1 = %.2e <= %g', desc$dcca1, thr$b_min_dcca)))
    }
    if(desc$mshape <= thr$b_min_mshape){
        return(result(FALSE, 'C', sprintf('mshape = %.3f <= %g', desc$mshape, thr$b_min_mshape)))
    }
    Qbar_eigen <- eigen(fit@mfit$Qbar, symmetric = TRUE, only.values = TRUE)$values
    if(min(Qbar_eigen) <= 0){
        return(result(FALSE, 'C', sprintf('Qbar not positive definite (min eigenvalue %.3g)', min(Qbar_eigen))))
    }
    if(abs(desc$unc_corr) >= thr$b_max_unc_corr){
        return(result(FALSE, 'C', sprintf('|unc_corr| = %.3f >= %g', abs(desc$unc_corr), thr$b_max_unc_corr)))
    }

    # D: plausibility against the seed model, loose on purpose, every rejection here trims the bootstrap distribution
    for(j in seq_along(subjects)){
        uncvol_col <- paste0('a', j-1, '_uncvol')
        uncvol_ratio <- desc[[uncvol_col]] / ref$desc[[uncvol_col]]
        if(uncvol_ratio < thr$b_min_uncvol_ratio || uncvol_ratio > thr$b_max_uncvol_ratio){
            return(result(FALSE, 'D', sprintf('%s uncvol / seed uncvol = %.2f outside [%g, %g]', subjects[[j]], uncvol_ratio, thr$b_min_uncvol_ratio, thr$b_max_uncvol_ratio)))
        }
    }
    unc_corr_diff <- abs(desc$unc_corr - ref$desc$unc_corr)
    if(unc_corr_diff > thr$b_max_unc_corr_diff){
        return(result(FALSE, 'D', sprintf('|unc_corr - seed unc_corr| = %.3f > %g', unc_corr_diff, thr$b_max_unc_corr_diff)))
    }

    # E: residual diagnostics, the refit is correctly specified by construction so rejections point at the optimizer
    umodelinc <- fit@model$umodel$modelinc
    mean_df <- umodelinc['ar', 1] + umodelinc['ma', 1]
    var_df <- umodelinc['alpha', 1] + umodelinc['beta', 1]
    for(j in seq_along(subjects)){
        s <- subjects[[j]]
        pvals <- tryCatch(
            list(
                LB = unlist(run_LB_tests(x = model$u[,j], lags = c(mean_df+1, probe_lags), fitdf = mean_df, verbose = FALSE)),
                LiM = unlist(run_LiM_tests(z = model$u[,j], h = model$h[,j], lags = c(var_df+1, probe_lags), fitdf = var_df, verbose = FALSE))
            ),
            error = function(e) NULL
        )
        if(is.null(pvals)){
            flags <- c(flags, paste0('resid_tests_failed_', s))
            next
        }
        for(test in names(pvals)){
            if(min(pvals[[test]]) < thr$b_resid_pval){
                flags <- c(flags, paste0('resid_', test, '_', s))
            }
        }
    }

    return(result(TRUE))
}

validate_trajectory <- function(simX, innovations, cond_std, cond_corr, ref, thr){
    # stage F: a synthetic test trajectory and its features filtered through the seed model
    fail <- function(reason) list(ok = FALSE, reason = reason)

    if(!all(is.finite(simX))){
        return(fail('non-finite returns'))
    }
    if(!all(is.finite(innovations)) || !all(is.finite(cond_std)) || !all(is.finite(cond_corr))){
        return(fail('non-finite filtered features'))
    }
    for(j in seq_along(ref$subjects)){
        s <- ref$subjects[[j]]
        r <- simX[1:(nrow(simX) - 1), j] # the returns entering the price path
        max_abs_sds <- max(abs(r)) / ref$ret_sd[[j]]
        if(max_abs_sds > thr$b_max_abs_return_sds){
            return(fail(sprintf('%s max |return| = %.1f seed sds > %g', s, max_abs_sds, thr$b_max_abs_return_sds)))
        }
        cum_return_sds <- (sum(r) - ref$ret_mean[[j]] * length(r)) / (ref$ret_sd[[j]] * sqrt(length(r)))
        if(abs(cum_return_sds) > thr$b_max_cum_return_sds){
            return(fail(sprintf('%s cumulative return = %.1f seed sds from the seed drift > %g', s, cum_return_sds, thr$b_max_cum_return_sds)))
        }
        if(min(cond_std[,j]) <= 0){
            return(fail(sprintf('%s non-positive condstd', s)))
        }
        condstd_ratio <- max(cond_std[,j]) / ref$max_condstd[[j]]
        if(condstd_ratio > thr$b_max_condstd_ratio){
            return(fail(sprintf('%s max condstd / max seed condstd = %.2f > %g', s, condstd_ratio, thr$b_max_condstd_ratio)))
        }
        innov_sd <- sd(innovations[,j])
        if(innov_sd < thr$b_min_innov_sd || innov_sd > thr$b_max_innov_sd){
            return(fail(sprintf('%s innovation sd = %.2f outside [%.2f, %.2f]', s, innov_sd, thr$b_min_innov_sd, thr$b_max_innov_sd)))
        }
    }
    if(max(abs(cond_corr)) >= 1){
        return(fail('|condcorr| >= 1'))
    }
    if(sd(cond_corr) < 1e-8){
        return(fail('constant condcorr'))
    }
    return(list(ok = TRUE, reason = NA_character_))
}

rbind_fill <- function(rows){
    # rbind data frames with differing columns, missing columns are filled with NA
    cols <- unique(unlist(lapply(rows, names)))
    do.call(
        rbind,
        lapply(
            rows,
            function(row){
                row[setdiff(cols, names(row))] <- NA
                row[cols]
            }
        )
    )
}

generate_bootstrapped_trajectories <- function(bootstrapped_model, original_model, starting_prices, sim_steps=209, m.sim=100, n.start=52, seed=42, startMethod='unconditional', mc.cores=4, traj_ids=0:(m.sim-1), ref=NULL, thr=NULL){
    # returns a list of m.sim list(result, ok, reason), result being the trajectory's columns named by traj_ids,
    # ok / reason the stage F check of the trajectory (skipped when ref is NULL)
    subjects <- colnames(original_model$z)

    if(length(seed) == 1){
        rseed <- seed:(seed + m.sim - 1)
    } else if (length(seed) == m.sim) {
        rseed <- seed
    } else {
        stop(paste0(seed, " is not a valid seed for m.sim=", m.sim))
    }

    # simulate according to the bootstrapped model
    n.sim <- sim_steps + 1
    sim <- dccsim(
        fitORspec = bootstrapped_model$fit,
        n.sim = n.sim,
        m.sim = m.sim,
        n.start = n.start,
        startMethod = startMethod,
        rseed = rseed
    )

    # filter through the original model
    spec <- get_fixed_param_dccspec(original_model)

    modeldata <- original_model$fit@model$modeldata$data
    T_model <- dim(modeldata)[1]

    results <- mclapply(
        1:(m.sim),
        FUN = function(i){
            tryCatch({
                traj_id <- traj_ids[[i]]
                simX <- sim@msim$simX[[i]]
                colnames(simX) <- subjects
                rownames(simX) <- NULL # delete row names to save memory

                filtdata <- rbind(modeldata, simX)

                filt <- dccfilter(
                    spec=spec,
                    data=filtdata,
                    filter.control = list(n.old=T_model)
                )

                returns <- filtdata[(T_model):(T_model+sim_steps),] # (sim_steps + 1) elements, p_t <- p_{t-1}*exp(r_t)
                colnames(returns) <- lapply(
                    seq_along(subjects),
                    function(j){
                        paste0(traj_id,"_",j-1,"_returns")
                    }
                )

                innovations <- filt@mfilter$stdresid
                innovations <- innovations[(T_model):(T_model+sim_steps),] # (sim_steps + 1) elements, aligned to returns, r_t <- mu + H_t * z_t
                colnames(innovations) <- lapply(
                    seq_along(subjects),
                    function(j){
                        paste0(traj_id,"_",j-1,"_innov")
                    }
                )

                cond_std <- coredata(sigma(filt))
                cond_std <- cond_std[(T_model+1):(T_model+sim_steps+1),] # (sim_steps + 1) elements, shifted to align as predictions
                colnames(cond_std) <- lapply(
                    seq_along(subjects),
                    function(j){
                        paste0(traj_id,"_",j-1,"_condstd")
                    }
                )

                cond_corr <- as.numeric(rcor(filt)[1,2,])
                cond_corr <- cond_corr[(T_model+1):(T_model+sim_steps+1)] # (sim_steps + 1) elements, shifted to align as predictions

                prices <- matrix(nrow = sim_steps + 1, ncol = length(subjects))
                for(j in seq_along(subjects)){
                    subject <- subjects[[j]]
                    prices[1,j] = starting_prices[[subject]]
                    prices[2:(sim_steps+1),j] = exp(cumsum(simX[1:sim_steps,subject])) * starting_prices[[subject]]
                }
                colnames(prices) <- lapply(
                    seq_along(subjects),
                    function(j){
                        paste0(traj_id,"_",j-1,"_prices")
                    }
                )

                result_matrix <- cbind(returns, prices, innovations, cond_std, cond_corr)
                colnames(result_matrix)[
                    length(colnames(result_matrix))
                ] <- paste0(traj_id,"_condcorr") # set the cond_corr column name

                check <- if(is.null(ref)) list(ok = TRUE, reason = NA_character_) else validate_trajectory(simX, innovations, cond_std, cond_corr, ref, thr)

                list(result = result_matrix, ok = check$ok, reason = check$reason)
            },
            error = function(e) list(result = NULL, ok = FALSE, reason = paste('filter error:', conditionMessage(e)))
            )
        },
        mc.cores = mc.cores
    )

    # a crashed mclapply worker returns a try-error instead of a list
    results <- lapply(
        results,
        function(res){
            if(inherits(res, 'try-error')) list(result = NULL, ok = FALSE, reason = paste('worker error:', as.character(res))) else res
        }
    )

    return(results)
}

create_bootstrapped_simulation_data <- function(
    bootstrapped_model,
    original_model,
    starting_prices,
    sim_steps,
    m.sim,
    n.start,
    seed,
    startMethod,
    mc.cores,
    save_path,
    ref,
    thr,
    extra_seed
){
    # trajectories failing stage F are resimulated one by one with seeds extra_seed, extra_seed + 1, ...
    # nothing is written if a trajectory still fails after thr$b_max_traj_retries resimulations
    # returns list(ok, reason, n_resim, extra_seed), extra_seed being the next unused resimulation seed
    trajectories <- generate_bootstrapped_trajectories(
        bootstrapped_model=bootstrapped_model,
        original_model=original_model,
        starting_prices=starting_prices,
        sim_steps=sim_steps,
        m.sim=m.sim,
        n.start=n.start,
        seed=seed,
        startMethod=startMethod,
        mc.cores=mc.cores,
        ref=ref,
        thr=thr
    )

    n_resim <- 0
    for(k in seq_along(trajectories)){
        retries <- 0
        while(!trajectories[[k]]$ok){
            if(retries >= thr$b_max_traj_retries){
                return(list(
                    ok = FALSE,
                    reason = sprintf('trajectory %d still rejected after %d resimulations (%s)', k-1, retries, trajectories[[k]]$reason),
                    n_resim = n_resim,
                    extra_seed = extra_seed
                ))
            }
            print(paste0("    trajectory ", k-1, " rejected (", trajectories[[k]]$reason, "), resimulating with seed ", extra_seed))
            trajectories[[k]] <- generate_bootstrapped_trajectories(
                bootstrapped_model=bootstrapped_model,
                original_model=original_model,
                starting_prices=starting_prices,
                sim_steps=sim_steps,
                m.sim=1,
                n.start=n.start,
                seed=extra_seed,
                startMethod=startMethod,
                mc.cores=1,
                traj_ids=k-1,
                ref=ref,
                thr=thr
            )[[1]]
            extra_seed <- extra_seed + 1
            retries <- retries + 1
            n_resim <- n_resim + 1
        }
    }

    simulation_results <- do.call(
        cbind,
        lapply(trajectories, function(trajectory) trajectory$result)
    ) # concatenate the results into a matrix

    df <- as.data.frame(simulation_results)

    save_dir <- dirname(save_path) # get save directory
    filename <- tools::file_path_sans_ext(basename(save_path)) # get filename without extension
    i <- 1 # duplicate cound
    fexists <- file.exists(save_path)
    while(fexists){
        print(paste("WARNING:", save_path, "exists in the filesystem. Saving as next up duplicate."))
        dupl_fname <- paste0(filename, "_", i)
        save_path <- path(save_dir, dupl_fname, ext = 'parquet') # construct new path
        fexists <- file.exists(save_path) # check if new path exists
        i <- i + 1
    }
    write_parquet(df, save_path)

    return(list(ok = TRUE, reason = NA_character_, n_resim = n_resim, extra_seed = extra_seed))
}

generate_bootstrapped_synth_test_data <- function(args){
    weekly_returns <- read_weekly_return_data(path=args[['seed_df_path']], mod=TRUE, format=args[['seed_df_format']])

    subjects <- args[['subjects']]
    if(is.null(subjects)){
        cn <- colnames(weekly_returns)
        subjects <- cn[cn!='date']
    } else {
        subjects <- as.character(subjects)
    }
    model <- fit_to_seed(
        weekly_returns=weekly_returns,
        modeling_end_date=args[['modeling_end_date']],
        subjects=subjects,
        model_save_path=args[['model_save_path']]
    )

    thr <- bootstrap_thresholds(args)
    num_bootstrapped_models <- as.integer(args[['num_bootstrapped_models']])
    b_m_sim <- as.integer(args[['b_m_sim']])
    max_attempts <- as.integer(thr$b_max_attempts)
    seed <- as.integer(args[['test_sim_seed']]) + as.integer(args[['test_m_sim']]) # first seed after the regular synth test batch

    # seed layout: attempt k (1-based) draws its bootstrap sample with seed + k - 1, accepted model i (1-based)
    # simulates its trajectories with seeds seed + i*b_m_sim + [0, b_m_sim), trajectory resimulations use
    # seed + (num_bootstrapped_models + 1)*b_m_sim onwards
    if(num_bootstrapped_models > max_attempts || max_attempts > b_m_sim){
        stop(sprintf("need num_bootstrapped_models (%d) <= b_max_attempts (%d) <= b_m_sim (%d) for non-overlapping seeds", num_bootstrapped_models, max_attempts, b_m_sim))
    }
    extra_seed <- seed + (num_bootstrapped_models + 1) * b_m_sim

    # bootstrapped trajectories span the same dates as the seed model's fit window
    train_boundaries <- get_train_boundaries(cutoff = args[['modeling_end_date']], weekly_returns = weekly_returns)
    fit_dates <- weekly_returns[weekly_returns$date >= train_boundaries[1] & weekly_returns$date <= train_boundaries[2], 'date']
    T_fit <- length(fit_dates)

    bootstrap_traj_dir <- args[['bootstrap_traj_dir']]
    print(paste0("Bootstrap: ", num_bootstrapped_models, " models x ", b_m_sim, " trajectories, at most ", max_attempts, " attempts, ", T_fit, " weeks per bootstrap sample"))
    print(paste0("Bootstrap: sample seeds ", seed, "..", seed + max_attempts - 1, ", trajectory seeds ", seed + b_m_sim, "..", extra_seed - 1, ", resimulation seeds ", extra_seed, "+"))
    print(paste("Bootstrap: writing to", bootstrap_traj_dir))

    ref <- bootstrap_reference(model)

    bootstrap_sim <- dccsim(
        fitORspec = model$fit,
        n.sim = T_fit,
        m.sim = max_attempts,
        n.start = args[['test_n_start']],
        startMethod = args[['startMethod']],
        rseed = seed:(seed + max_attempts - 1)
    ) # path k only depends on its seed, so the first num_bootstrapped_models paths do not depend on max_attempts

    dir_create(bootstrap_traj_dir)
    rejections_path <- path(bootstrap_traj_dir, 'bootstrap_rejections', ext = 'csv')
    rejections <- list()
    write_rejections <- function(){
        if(length(rejections) == 0){
            log <- data.frame(attempt = integer(0), seed = integer(0), stage = character(0), reason = character(0), flags = character(0))
        } else {
            log <- rbind_fill(rejections)
        }
        write.csv(log, rejections_path, row.names = FALSE)
    }
    save_bootstrap_model <- function(model, file){
        save(model, file = file) # export_bootstrap_params loads the object by the name model
    }

    accepted <- 0
    attempt <- 0
    total_resim <- 0
    while(accepted < num_bootstrapped_models && attempt < max_attempts){
        attempt <- attempt + 1
        sample_seed <- seed + attempt - 1
        tag <- sprintf("[attempt %d/%d | accepted %d/%d] seed=%d", attempt, max_attempts, accepted, num_bootstrapped_models, sample_seed)

        simX <- bootstrap_sim@msim$simX[[attempt]]
        colnames(simX) <- subjects

        # A: bootstrap sample
        check <- validate_bootstrap_sample(simX, ref, thr)

        # B-E: refit a model of the seed model's configuration to the bootstrap sample
        bootstrapped_model <- NULL
        if(check$ok){
            bootstrapped_returns <- data.frame(date = fit_dates, simX)
            bootstrapped_model <- tryCatch(
                fit_to_seed(
                    weekly_returns=bootstrapped_returns,
                    modeling_end_date=args[['modeling_end_date']],
                    subjects=subjects,
                    model_save_path=NULL
                ),
                error = function(e) e
            )
            if(inherits(bootstrapped_model, 'error')){
                check <- list(ok = FALSE, stage = 'B', reason = paste('fit error:', conditionMessage(bootstrapped_model)))
            } else {
                check <- validate_dcc_fit(bootstrapped_model, ref, thr)
            }
        }

        # F: synth test trajectories from the bootstrapped model
        if(check$ok){
            model_name <- paste0("bootstrap_model_", accepted)
            flags_note <- if(nchar(check$flags) > 0) paste0(" (flags: ", check$flags, ")") else ""
            print(paste0(tag, " refit passed", flags_note, ", simulating ", b_m_sim, " trajectories"))
            sim_check <- create_bootstrapped_simulation_data(
                bootstrapped_model=bootstrapped_model,
                original_model=model,
                starting_prices=args[['starting_prices']],
                sim_steps=args[['test_sim_steps']],
                m.sim=b_m_sim,
                n.start=args[['test_n_start']],
                seed=seed + (accepted + 1)*b_m_sim,
                startMethod=args[['startMethod']],
                mc.cores=args[['mc_cores']],
                save_path=path(bootstrap_traj_dir, model_name, ext = 'parquet'),
                ref=ref,
                thr=thr,
                extra_seed=extra_seed
            )
            extra_seed <- sim_check$extra_seed
            total_resim <- total_resim + sim_check$n_resim
            if(!sim_check$ok){
                check <- list(ok = FALSE, stage = 'F', reason = sim_check$reason, flags = check$flags, desc = check$desc)
            }
        }

        if(check$ok){
            bootstrapped_model$bootstrap <- list(attempt = attempt, seed = sample_seed, flags = check$flags)
            save_bootstrap_model(bootstrapped_model, path(bootstrap_traj_dir, model_name, ext = 'RData'))
            accepted <- accepted + 1
            print(paste0(tag, " ACCEPTED as ", model_name, ", ", sim_check$n_resim, " trajectories resimulated"))
        } else {
            print(paste0(tag, " REJECTED (stage ", check$stage, ": ", check$reason, ")"))
            row <- data.frame(attempt = attempt, seed = sample_seed, stage = check$stage, reason = check$reason, flags = if(is.null(check$flags)) NA_character_ else check$flags)
            if(!is.null(check$desc)){
                row <- cbind(row, check$desc)
            }
            rejections[[length(rejections) + 1]] <- row
        }
        write_rejections()
    }

    stage_counts <- table(vapply(rejections, function(row) row$stage, character(1)))
    print(paste0("Bootstrap: accepted ", accepted, "/", num_bootstrapped_models, " models in ", attempt, " attempts, ", total_resim, " trajectories resimulated"))
    if(length(rejections) > 0){
        print(paste("Bootstrap: rejections by stage:", paste(names(stage_counts), stage_counts, sep = "=", collapse = ", ")))
    }
    print(paste("Bootstrap: rejection log", rejections_path))
    if(accepted > 0){
        print(paste0("Bootstrap: models and trajectories ", path(bootstrap_traj_dir, "bootstrap_model_"), "{0..", accepted - 1, "}.{RData,parquet}"))
    }
    if(attempt > 0 && length(rejections) / attempt > thr$b_warn_rejection_rate){
        print(sprintf("WARNING: %.0f%% of the bootstrap attempts were rejected (> %.0f%%), check %s, the seed model may be the problem rather than the bootstrap", 100 * length(rejections) / attempt, 100 * thr$b_warn_rejection_rate, rejections_path))
    }
    if(accepted < num_bootstrapped_models){
        stop(sprintf("only %d of %d bootstrapped models accepted after %d attempts, see %s", accepted, num_bootstrapped_models, attempt, rejections_path))
    }
}

describe_dcc_model <- function(model, subjects){
    # flat parameter summary of a fitted DCC-eGARCH model (one row); assets are
    # labelled by position (a0 = subjects[1] = anchor, a1 = subjects[2])
    cf <- coef(model$fit)
    row <- list(
        convergence = model$fit@mfit$convergence,
        llh = model$fit@mfit$llh
    )
    uncvol <- numeric(length(subjects))
    for(j in seq_along(subjects)){
        for(p in c('mu', 'omega', 'alpha1', 'beta1', 'gamma1', 'skew', 'shape')){
            row[[paste0('a', j-1, '_', p)]] <- unname(cf[paste0('[', subjects[[j]], '].', p)])
        }
        # eGARCH: E[log sigma^2] = omega / (1 - beta), geometric unconditional vol
        uncvol[j] <- exp(row[[paste0('a', j-1, '_omega')]] / (2 * (1 - row[[paste0('a', j-1, '_beta1')]])))
        row[[paste0('a', j-1, '_uncvol')]] <- uncvol[j]
    }
    row[['dcca1']] <- unname(cf['[Joint]dcca1'])
    row[['dccb1']] <- unname(cf['[Joint]dccb1'])
    row[['mshape']] <- unname(cf['[Joint]mshape'])
    row[['dcc_persistence']] <- row[['dcca1']] + row[['dccb1']]
    Qbar <- model$fit@mfit$Qbar
    row[['unc_corr']] <- Qbar[1,2] / sqrt(Qbar[1,1] * Qbar[2,2])
    row[['uncvol_ratio']] <- uncvol[1] / uncvol[2]
    row[['unc_mv_ratio']] <- row[['unc_corr']] * row[['uncvol_ratio']] # MV ratio at the unconditional moments
    return(as.data.frame(row))
}

export_bootstrap_params <- function(args){
    # writes <bootstrap_traj_dir>/bootstrap_params.csv: one row per bootstrapped
    # model (model = 0..num_bootstrapped_models-1) plus the seed model (model = -1)
    # accepted / reject_stage / reject_reason / flags re-run the refit checks (stages B-E) under the
    # current thresholds, attempt / bootstrap_seed are NA for models generated without the guardrails
    weekly_returns <- read_weekly_return_data(path=args[['seed_df_path']], mod=TRUE, format=args[['seed_df_format']])

    subjects <- args[['subjects']]
    if(is.null(subjects)){
        cn <- colnames(weekly_returns)
        subjects <- cn[cn!='date']
    } else {
        subjects <- as.character(subjects)
    }
    model <- fit_to_seed(
        weekly_returns=weekly_returns,
        modeling_end_date=args[['modeling_end_date']],
        subjects=subjects,
        model_save_path=NULL
    )
    ref <- bootstrap_reference(model)
    thr <- bootstrap_thresholds(args)
    describe_row <- function(model_idx, m){
        check <- validate_dcc_fit(m, ref, thr)
        cbind(
            data.frame(
                model_idx = model_idx,
                accepted = check$ok,
                reject_stage = check$stage,
                reject_reason = check$reason,
                flags = check$flags,
                attempt = if(is.null(m$bootstrap)) NA_integer_ else m$bootstrap$attempt,
                bootstrap_seed = if(is.null(m$bootstrap)) NA_integer_ else m$bootstrap$seed
            ),
            describe_dcc_model(m, subjects)
        )
    }
    rows <- list(describe_row(-1L, model))

    num_bootstrapped_models <- as.integer(args[['num_bootstrapped_models']])
    for(i in 1:num_bootstrapped_models){
        env <- new.env()
        load(path(args[['bootstrap_traj_dir']], paste0("bootstrap_model_", i-1), ext = 'RData'), envir = env)
        rows[[length(rows) + 1]] <- describe_row(i - 1L, env$model)
    }
    n_failing <- sum(!vapply(rows[-1], function(row) row$accepted, logical(1)))
    if(n_failing > 0){
        print(paste("WARNING:", n_failing, "of", num_bootstrapped_models, "bootstrapped models fail the refit checks under the current thresholds, see the accepted column"))
    }

    params <- do.call(rbind, rows)
    write.csv(params, path(args[['bootstrap_traj_dir']], 'bootstrap_params', ext = 'csv'), row.names = FALSE)
}
