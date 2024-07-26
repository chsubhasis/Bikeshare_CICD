import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from bikeshare_model.config.core import config
from bikeshare_model.processing.features import WeekdayImputer, WeathersitImputer
from bikeshare_model.processing.features import Mapper
from bikeshare_model.processing.features import OutlierHandler, WeekdayOneHotEncoder

bikeshare_pipe = Pipeline(
    [
        ######### Imputation ###########
        (
            "weekday_imputation",
            WeekdayImputer(
                variable=config.model_config.weekday_var,
                date_var=config.model_config.date_var,
            ),
        ),
        (
            "weathersit_imputation",
            WeathersitImputer(variable=config.model_config.weathersit_var),
        ),
        ######### Mapper ###########
        (
            "map_yr",
            Mapper(
                variable=config.model_config.yr_var,
                mappings=config.model_config.yr_mappings,
            ),
        ),
        (
            "map_mnth",
            Mapper(
                variable=config.model_config.mnth_var,
                mappings=config.model_config.mnth_mappings,
            ),
        ),
        (
            "map_season",
            Mapper(
                variable=config.model_config.season_var,
                mappings=config.model_config.season_mappings,
            ),
        ),
        (
            "map_weathersit",
            Mapper(
                variable=config.model_config.weathersit_var,
                mappings=config.model_config.weathersit_mappings,
            ),
        ),
        (
            "map_holiday",
            Mapper(
                variable=config.model_config.holiday_var,
                mappings=config.model_config.holiday_mappings,
            ),
        ),
        (
            "map_workingday",
            Mapper(
                variable=config.model_config.workingday_var,
                mappings=config.model_config.workingday_mappings,
            ),
        ),
        (
            "map_hr",
            Mapper(
                variable=config.model_config.hr_var,
                mappings=config.model_config.hr_mappings,
            ),
        ),
        ######## Handle outliers ########
        ("handle_outliers_temp", OutlierHandler(variable=config.model_config.temp_var)),
        (
            "handle_outliers_atemp",
            OutlierHandler(variable=config.model_config.atemp_var),
        ),
        ("handle_outliers_hum", OutlierHandler(variable=config.model_config.hum_var)),
        (
            "handle_outliers_windspeed",
            OutlierHandler(variable=config.model_config.windspeed_var),
        ),
        ######## One-hot encoding ########
        (
            "encode_weekday",
            WeekdayOneHotEncoder(variable=config.model_config.weekday_var),
        ),
        # Scale features
        ("scaler", StandardScaler()),
        # Regressor
        (
            "model_rf",
            RandomForestRegressor(
                n_estimators=config.model_config.n_estimators,
                max_depth=config.model_config.max_depth,
                random_state=config.model_config.random_state,
            ),
        ),
    ]
)
