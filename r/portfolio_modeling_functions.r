library(rugarch)
library(WeightedPortTest)
library(FinTS)
library(ggplot2)
library(rmgarch)
library(xts)
library(checkmate)
library(arrow)
library(stringr)

## Utils
read_weekly_return_data <- function(path, mod = TRUE, format='csv') {
    read_file <- switch(
        format,
        csv=read.csv,
        parquet=read_parquet
    )

    weekly_returns <- read_file(path)
    weekly_returns <- data.frame(weekly_returns)

    if(mod){
        weekly_returns$date <- as.Date(weekly_returns$date, '%Y-%m-%d')
        names(weekly_returns)[names(weekly_returns) == 'X.GSPC'] <- 'GSPC'
    }
    
    return(weekly_returns)
}

get_train_boundaries <- function(weekly_returns, cutoff = '2022-01-01') {
    c(as.Date(head(weekly_returns$date, 1)), as.Date(tail(weekly_returns[weekly_returns$date < cutoff,'date'], 1)))
}

get_tic_weekly_returns <- function(weekly_returns, tic, start_date, end_date){
    y = weekly_returns[weekly_returns$date >= start_date & weekly_returns$date <= end_date,]
    xts(y[[tic]], order.by = y$date)
}

## Class Constructors
new_ugarch_config <- function(model='eGARCH', ARCH_lags=1, GARCH_lags=1, AR_lags=0, MA_lags=0, dist='sstd', mean_x_reg=NULL, solver="hybrid") {
    obj <- list(
        model=model,
        ARCH_lags=ARCH_lags,
        GARCH_lags=GARCH_lags,
        AR_lags=AR_lags,
        MA_lags=MA_lags,
        dist=dist,
        mean_x_reg=mean_x_reg,
        solver=solver
    )

    class(obj) <- 'ugarch_config'

    return(obj)
    
}

new_mgarch_config <- function(uspecs, model='DCC', dccOrder=c(1,1), distribution='mvt', solver=c("hybrid", "solnp"), subjects=c("subject1", "subject2")) {
    obj <- list(
        uspecs=uspecs,
        model=model,
        dccOrder=dccOrder,
        distribution=distribution,
        solver=solver,
        subjects=subjects
    )

    class(obj) <- 'mgarch_config'

    return(obj)
}


new_ugarch_model <- function(fit, z, h, spec, config) {
    obj <- list(
        fit = fit,
        z = z,
        h = h,
        spec = spec,
        config=config
    )

    class(obj) <- 'ugarch_model'

    return(obj)
}

new_mgarch_model <- function(fit, z, h, u, spec, config) {
    obj <- list(
        fit = fit,
        z = z,
        h = h,
        u = u,
        spec = spec,
        config=config
    )

    class(obj) <- 'mgarch_model'

    return(obj)
}

## Statistical Tests, Effects Testing & related util functions
print_test_results <- function(pvals, test_name, exceptional_event='none', identifier=NULL){
    exceptional_pvals <- list()
    if(exceptional_event == 'sig'){
        exceptional_pvals <- pvals[pvals <= 0.05]
    } else if (exceptional_event == 'insig'){
        exceptional_pvals <- pvals[pvals > 0.05]
    }

    if (length(exceptional_pvals)==0){
        print(paste(identifier, test_name, 'tests are good.'))
    } else {
        for (exceptional_lag in names(exceptional_pvals)){
            if (is.null(identifier)){
                print("Warning: no identifier for the test...")
                identifier <- "_"
            }
            print(paste("Exception in", identifier, test_name, "test with", exceptional_lag, "-> p-value:", exceptional_pvals[[exceptional_lag]]))
        }
    }
}

run_LB_tests <- function(x, lags=c(8,13,26), fitdf=0, weighted=TRUE, exceptional_event='none', identifier=NULL, verbose=TRUE) {
    checkmate::assert_integerish(lags, lower = 1, any.missing = FALSE, .var.name = "Lags")
    checkmate::assert_integerish(fitdf, lower = 0, .var.name = "Model d.o.f")
    
    lags <- setNames(as.list(lags), paste0("lag_", lags))

    pvals <- lapply(
        lags,
        function(lag){
            Weighted.Box.test(x, lag=lag, type="Ljung-Box", fitdf=fitdf, weighted=TRUE)$p.value
        }
    )
    
    if (verbose){
        print_test_results(pvals, test_name = "LB", exceptional_event=exceptional_event, identifier=identifier)
    }

    return(pvals)
}

run_Arch_LM_tests <- function(x, lags=c(8,13,26), exceptional_event='none', identifier=NULL, verbose=TRUE) {
    checkmate::assert_integerish(lags, lower = 1, any.missing = FALSE, .var.name = "Lags")
    
    lags <- setNames(as.list(lags), paste0("lag_", lags))
    pvals <- lapply(
        lags,
        function(lag){
            ArchTest(x, lags=lag, demean=FALSE)$p.value
        }
    )

    if(verbose){
        print_test_results(pvals, test_name = "ARCH-LM", exceptional_event=exceptional_event, identifier=identifier)
    }

    return(pvals)
}

run_LiM_tests <- function(z, h, lags=c(8,13,26), fitdf=2, weighted=TRUE, exceptional_event='none', identifier=NULL, verbose=TRUE) {
    checkmate::assert_integerish(lags, lower = 1, any.missing = FALSE, .var.name = "Lags")
    checkmate::assert_integerish(fitdf, lower = 0, .var.name = "Model d.o.f")

    lags <- setNames(as.list(lags), paste0("lag_", lags))
    pvals <- lapply(
        lags,
        function(lag){
            Weighted.LM.test(z, h, lag=lag, fitdf=fitdf, weighted=TRUE)$p.value
        }
    )

    if(verbose) {
        print_test_results(pvals, test_name = "LiM", exceptional_event=exceptional_event, identifier=identifier)
    }

    return(pvals)
}

run_DCC_tests <- function(data, lags=c(8,13,26), garchOrder=c(1,1), exceptional_event='none', identifier=NULL, verbose=TRUE) {
    checkmate::assert_integerish(lags, lower = 1, any.missing = FALSE, .var.name = "Lags")

    lags <- setNames(as.list(lags), paste0("lag_", lags))

    pvals <- lapply(
        lags,
        function(lag){
            DCCtest(data, n.lags=lag, garchOrder = garchOrder)$p.value
        }
    )

    if(verbose) {
        print_test_results(pvals, test_name = "DCC", exceptional_event=exceptional_event, identifier=identifier)
    }
    
    return(pvals)
}

test_univar_unmodeled_effects <- function(z, h, probe_lags=c(8,13,26), mean_df=0, var_df=2, identifier='Std. resid.', verbose=TRUE) {
    zlb <- run_LB_tests(x = z, lags=c(mean_df+1, probe_lags), fitdf=mean_df, weighted=TRUE, exceptional_event = 'sig', identifier = identifier, verbose=verbose)
    zlim <- run_LiM_tests(z = z, h = h, lags = c(var_df+1, probe_lags),fitdf = var_df,weighted = TRUE, exceptional_event = 'sig', identifier = identifier, verbose=verbose)
}

check_DCC_effects_in_z <- function(subjects, models, lags=c(8,13,26)){
    # the univariate models are assumed to have the same garchOrder
    # stopifnot(identical(models[[subjects[1]]]$spec, models[[subjects[2]]]$spec))
    spec <- models[[subjects[1]]]$spec
    garchOrder <- c(spec@model$modelinc["alpha"], spec@model$modelinc["beta"])

    zs = cbind(
        models[[subjects[1]]]$z,
        models[[subjects[2]]]$z
    )

    dcc_pvals <- run_DCC_tests(data = zs, garchOrder = garchOrder, lags = lags, verbose=FALSE)
    print(paste(paste(subjects, collapse = '|'), '-- Std. Resid. DCC Test'))
    print(as.data.frame(dcc_pvals), row.names = FALSE)
}

test_multivar_unmodeled_effects <- function(model, probe_lags=c(8,13,26), verbose=TRUE) {
    spec <- model$spec
    subjects <- model$config$subjects

    # assumes the ugarchOrder is the same for both univariate models
    garchOrder <- c(spec@umodel$modelinc["alpha",1], spec@umodel$modelinc["beta",1])

    mean_df <- spec@umodel$modelinc["ar",1] + spec@umodel$modelinc["ma",1]
    var_df <- sum(garchOrder)


    test_univar_unmodeled_effects(z=model$u[,1], h=model$h[,1], probe_lags=probe_lags, mean_df=mean_df, var_df=var_df, identifier = paste(subjects[1], 'Corr. Adj. Resid.'), verbose=TRUE)
    test_univar_unmodeled_effects(z=model$u[,2], h=model$h[,2], probe_lags=probe_lags, mean_df=mean_df, var_df=var_df, identifier = paste(subjects[2], 'Corr. Adj. Resid.'), verbose=TRUE)

    dcc_pvals <- run_DCC_tests(
        data = model$u,
        garchOrder = garchOrder,
        lags = probe_lags,
        exceptional_event='sig',
        identifier=paste(paste(subjects, collapse = "|"), 'Corr. Adj. Resid.'),
        verbose=verbose
    )
}


## Calculations & Analysis
calculate_rolling_spearman_correlation <- function(data, window_size=104, step_size=26, return_no_na=TRUE){
    stopifnot(nrow(data) >= window_size)
    stopifnot(all(!is.na(data)))

    res <- rollapplyr(
        data=data,
        width=window_size,
        by=step_size,
        by.column=FALSE,
        FUN=function(x) cor(x[,1],x[,2],method = "spearman")
    )

    if (return_no_na) {
        res <- na.omit(res)
    }

    return(res)
}

ols_ci <- function(data, response_col, regressor_col) {
    fit <- lm(
        reformulate(regressor_col, response = response_col),
        data = data
    )

    # Extract coefficient and standard error from the summary
    est <- coef(summary(fit))[regressor_col, "Estimate"]
    se  <- coef(summary(fit))[regressor_col, "Std. Error"]

    # Calculate 95% Confidence Interval bounds
    lower <- est - 1.96 * se
    upper <- est + 1.96 * se
  
    # Returning a named vector creates columns in the rollapplyr output
    return(c(rolling_beta = est, ci_lower = lower, ci_upper = upper))
}

rolling_ols_analysis <- function(data, response_col, regressor_col, window_size=104, step_size=26, return_no_na=TRUE){
    stopifnot(nrow(data) >= window_size)
    stopifnot(all(!is.na(data)))

    res <- rollapplyr(
        data=data,
        width=window_size,
        by=step_size,
        by.column=FALSE,
        FUN=ols_ci,
        response_col=response_col,
        regressor_col=regressor_col
    )

    if (return_no_na) {
        res <- na.omit(res)
    }

    return(res)
}

## ugarch & mgarch Class Functions
# get spec
get_spec <- function(config, ...) UseMethod("get_spec")

get_spec.ugarch_config <- function(config) {
    if(!is.null(config$mean_x_reg)){
        mean_model <- list(
            armaOrder = c(config$AR_lags, config$MA_lags),
            include.mean = TRUE,
            external.regressors = matrix(config$mean_x_reg, ncol=1)
        )
    } else {
        mean_model <- list(
            armaOrder = c(config$AR_lags, config$MA_lags),
            include.mean = TRUE
        )
    }
    
    ugarchspec(
        variance.model = list(
            model = config$model,
            garchOrder = c(config$ARCH_lags, config$GARCH_lags)
        ),
        mean.model = mean_model,
        distribution.model = config$dist
    )
}

get_spec.mgarch_config <- function(config) {
    # Multivariate DCC Spec
    mspec <- multispec(config$uspecs)

    # DCC
    dcc_spec <- dccspec(
        uspec       = mspec,
        dccOrder    = config$dccOrder,     # DCC(1,1): a, b parameters governing Q_t dynamics
        distribution = config$distribution,      # joint t distribution for the DCC stage
        model       = config$model        # might try aDCC later
    )
}

fit_model <- function(model_config, weekly_returns, fit_start, fit_end, probe_lags, ...) UseMethod("fit_model")

fit_model.ugarch_config <- function(model_config, tic, weekly_returns, fit_start, fit_end, probe_lags = c(8,13,26)) {
    y = get_tic_weekly_returns(weekly_returns=weekly_returns, tic=tic,start_date = fit_start, end_date = fit_end)

    spec <- get_spec(model_config)

    fit <- ugarchfit(spec = spec, data = y, solver = model_config$solver)

    std_resid <- residuals(fit, standardize = TRUE)
    cond_var <- fit@fit$var
    
    return(new_ugarch_model(fit=fit, z=std_resid, h=cond_var, spec=spec, config = model_config))

    return(model)
}

fit_model.mgarch_config <- function(model_config, weekly_returns, fit_start, fit_end, probe_lags = c(8,13,26), tic_codes = NULL) {
    subjects <- model_config$subjects

    if (is.null(tic_codes)){
        tic_codes <- subjects
    } else {
        print(paste("NOTICE: Using tic codes", paste(tic_codes, collapse = ",")))
    }
    
    ret1 = get_tic_weekly_returns(tic = tic_codes[1],start_date = fit_start, end_date = fit_end, weekly_returns = weekly_returns)
    ret2 = get_tic_weekly_returns(tic = tic_codes[2],start_date = fit_start, end_date = fit_end, weekly_returns = weekly_returns)

    spec <- get_spec(config = model_config)

    data <- cbind(ret1, ret2)
    colnames(data) <- subjects

    dcc_fit <- dccfit(
        spec,
        data = data,
        fit.control = list(eval.se = TRUE),
        solver = model_config$solver
    )

    z <- residuals(dcc_fit) / sigma(dcc_fit)
    h <- sigma(dcc_fit)^2

    u <- xts(dcc_fit@mfit$stdresid, order.by = index(z))
    colnames(u) <- colnames(z)

    return(new_mgarch_model(fit=dcc_fit, z=z, h=h, u=u, spec=spec, config = model_config))

    return(model)
}

# evaluate model residuals
evaluate_model_resid <- function(model, probe_lags, ...) UseMethod("evaluate_model_resid")

evaluate_model_resid.ugarch_model <- function(model, probe_lags=c(8,13,26), verbose=TRUE){
    u_fit <- model$fit

    mean_df = u_fit@model$modelinc["ar"] + u_fit@model$modelinc["ma"]
    var_df = u_fit@model$modelinc["alpha"] + u_fit@model$modelinc["beta"]

    test_univar_unmodeled_effects(z=model$z, h=model$h, probe_lags=probe_lags, mean_df=mean_df, var_df=var_df)
}

evaluate_model_resid.mgarch_model <- function(model, probe_lags=c(8,13,26), verbose=TRUE){
    subjects <- model$config$subjects
    
    u <- model$u
    h <- model$h

    # umodels are assumed to have the same GARCH configuration
    mean_df <- model$fit@model$umodel$modelinc['ar',][1] + model$fit@model$umodel$modelinc['ma',][1]
    var_df <- model$fit@model$umodel$modelinc['alpha',][1] + model$fit@model$umodel$modelinc['beta',][1]
    
    test_multivar_unmodeled_effects(model=model, probe_lags=probe_lags, verbose=verbose)
}

run_experiment <- function(model_config, weekly_returns, fit_start, fit_end, probe_lags, ...) UseMethod("run_experiment")

run_experiment.ugarch_config <- function(model_config, tic, weekly_returns, fit_start, fit_end, probe_lags = c(8,13,26)) {
    model <- fit_model(
        model_config = model_config,
        tic = tic,
        weekly_returns = weekly_returns,
        fit_start = fit_start,
        fit_end = fit_end,
        probe_lags = probe_lags
    )

    evaluate_model_resid(model = model, probe_lags = probe_lags, verbose=TRUE)

    return(model)
}

run_experiment.mgarch_config <- function(model_config, weekly_returns, fit_start, fit_end, probe_lags = c(8,13,26), tic_codes = NULL) {
    model <- fit_model(
        model_config = model_config,
        weekly_returns = weekly_returns,
        fit_start = fit_start,
        fit_end = fit_end,
        probe_lags = probe_lags,
        tic_codes = tic_codes
    )

    evaluate_model_resid(model=model, probe_lags = probe_lags, verbose=TRUE)

    return(model)
}


## Rolling Analysis Plots
plot_rolling_ols_graph <- function(weekly_returns, response_tic, regressor_tic, window_size=104, step_size=26){
    data <- cbind(
        get_tic_weekly_returns(tic=response_tic, start_date = fit_start, end_date = fit_end, weekly_returns = weekly_returns),
        get_tic_weekly_returns(tic=regressor_tic, start_date = fit_start, end_date = fit_end, weekly_returns = weekly_returns)
    )
    colnames(data) <- c(response_tic, regressor_tic)

    res <- rolling_ols_analysis(data, response_col = response_tic, regressor_col = regressor_tic, window_size = window_size, step_size = step_size)

    options(repr.plot.width = 15, repr.plot.height =8)
    # 4. Plot parameter and confidence interval against time
    ggplot(res, aes(x = index(res), y = rolling_beta)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    # Add the shaded region using geom_ribbon (place it before geom_line so it stays behind)
    geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), fill = "#0072B2", alpha = 0.2) +
    geom_line(color = "#0072B2", linewidth = 0.9) +
    geom_point(color = "#0072B2", size = 2) + 
    labs(
        title = paste0(response_tic, "~", regressor_tic, " Rolling OLS Parameter Estimate (Window = ", window_size, ", Step = ", step_size, ")"),
        subtitle = "Shaded region represents 95% Confidence Interval",
        x = "Time",
        y = expression(hat(beta))
    ) +
    theme_minimal(base_size = 12)
}

plot_rolling_corr_comparison_graph <- function(multivariate_model, window_size=52, step_size=4){
    model <- multivariate_model
    subjects <- model$config$subjects

    zs <- cbind(
        model$z[,1],
        model$z[,2]
    )
    colnames(zs) <- subjects
    us <- cbind(
        model$u[,1],
        model$u[,2]
    )
    colnames(us) <- subjects


    zs_rolling_corr <- calculate_rolling_spearman_correlation(data=zs, window_size=window_size, step_size=step_size)
    us_rolling_corr <- calculate_rolling_spearman_correlation(data=us, window_size=window_size, step_size=step_size)

    results <- cbind(zs_rolling_corr, us_rolling_corr)
    colnames(results) <- c('z', 'u')

    z_corr_name <- paste0(paste0('(z_',subjects[1]), ',', paste0('z_',subjects[2]), ') rolling corr.')
    u_corr_name <- paste0(paste0('(u_',subjects[1]), ',', paste0('u_',subjects[2]), ') rolling corr.')

    color_values <- c("#0072B2", "#D5500A")
    names(color_values) <- c(
    z_corr_name,
    u_corr_name
    )

    options(repr.plot.width = 15, repr.plot.height =8)
    ggplot(results, aes(x = index(results))) +
    ylim(-1,1) +
    geom_line(aes(y = z, color = z_corr_name), linewidth = 0.9) +
    geom_line(aes(y = u, color = u_corr_name), linewidth = 0.9) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray31") +
    labs(
        title = paste(paste(subjects, collapse = "|"), "Rolling Spearman Correlation Comparison between Univariate and DCC models ( window=", window_size, ", step=", step_size, ")"),
        x = "Time",
        y = "Spearman Correlation",
        color = "Legend"
    ) +
    theme_minimal(base_size = 12) +
    scale_color_manual(values = color_values)
}

## Filtering
get_stage_specific_pnames <- function(spec){
    names(which(spec@model$pars[,'Include']==1))
}

pnames_to_labeled_pnames <- function(pnames, label){
    lapply(
        pnames,
        function(pname){
            return(paste0(label,pname))
        }
    )
}

unlabel_params <- function(labeled_params, split_on){
    names(labeled_params) <- str_split(names(labeled_params), split_on, simplify = TRUE)[,2]
    return(labeled_params)
}

get_first_stage_params <- function(model){
    subjects <- model$config$subjects
    uspecs <- model$config$uspecs
    results <- vector("list", length = length(subjects))
    for(j in 1:length(subjects)){
        uspec_pnames <- get_stage_specific_pnames(uspecs[[j]])
        mspec_pnames <- pnames_to_labeled_pnames(
            uspec_pnames,
            paste0('[',subjects[[j]],'].')
        )
        labeled_params <- coef(model$fit)[
            as.character(mspec_pnames)
        ]
        results[[j]] <- unlabel_params(labeled_params, split_on='].')
    }
    names(results) <- subjects
    return(results)
}

get_fixed_param_uspecs <- function(model){
    subjects <- model$config$subjects
    uspecs <- model$config$uspecs
    params <- get_first_stage_params(model = model)
    fixed_uspecs <- vector("list", length(subjects))
    for(j in seq_along(uspecs)){
        spec <- uspecs[[j]]
        setfixed(spec) <- params[[j]]
        fixed_uspecs[[j]] <- spec
    }
    names(fixed_uspecs) <- subjects
    return(fixed_uspecs)
}

get_second_stage_params <- function(model){
    subjects <- model$config$subjects
    second_stage_pnames <- get_stage_specific_pnames(model$spec)
    labeled_pnames <- pnames_to_labeled_pnames(
        pnames = second_stage_pnames,
        label = '[Joint]'
    )
    labeled_params <- coef(model$fit)[
        as.character(labeled_pnames)
    ]
    unlabel_params(labeled_params, split_on=']')
}

get_fixed_param_dccspec <- function(model){
    # duplicate model spec
    fixed_param_config <- model$config
    
    # replace uspecs with fixed param uspecs
    fixed_param_config$uspecs <- get_fixed_param_uspecs(model)

    # get new dcc spec where uspecs have fixed parameters
    dcc_spec <- get_spec(fixed_param_config)

    setfixed(dcc_spec) <- get_second_stage_params(model)
    
    return(dcc_spec)
}