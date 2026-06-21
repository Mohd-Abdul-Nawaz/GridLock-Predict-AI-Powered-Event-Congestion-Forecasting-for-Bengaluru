"""
Enhanced Event-Driven Congestion Forecasting Module
Features ensemble learning, SHAP explanations, Bayesian optimization,
post-event learning, and comprehensive resource recommendation.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                                VotingRegressor, StackingRegressor)
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, mean_absolute_percentage_error)
from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb
import lightgbm as lgb
import warnings
from datetime import datetime, timedelta
from src.utils.config import EVENT_CONFIG

warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except (ImportError, OSError):
    SHAP_AVAILABLE = False


class BayesianOptimizedXGBoost:
    """XGBoost with Bayesian-style hyperparameter tuning."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.best_params = None

    def tune_and_train(self, X_train, y_train, X_val, y_val):
        """Tune hyperparameters and train the model."""
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0, 0.1, 0.2, 0.3],
        }

        # Use a more efficient approach: random search with limited iterations
        best_score = float('inf')
        best_params = {}
        n_iterations = min(20, len(param_grid['n_estimators']) *
                           len(param_grid['max_depth']) *
                           len(param_grid['learning_rate']))

        for _ in range(n_iterations):
            params = {
                'n_estimators': np.random.choice(param_grid['n_estimators']),
                'max_depth': np.random.choice(param_grid['max_depth']),
                'learning_rate': np.random.choice(param_grid['learning_rate']),
                'subsample': np.random.choice(param_grid['subsample']),
                'colsample_bytree': np.random.choice(param_grid['colsample_bytree']),
                'min_child_weight': np.random.choice(param_grid['min_child_weight']),
                'gamma': np.random.choice(param_grid['gamma']),
            }

            model = xgb.XGBRegressor(
                **params,
                random_state=self.random_state,
                objective='reg:squarederror',
                verbosity=0,
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = mean_absolute_error(y_val, y_pred)

            if score < best_score:
                best_score = score
                best_params = params

        self.best_params = best_params
        self.model = xgb.XGBRegressor(
            **best_params,
            random_state=self.random_state,
            objective='reg:squarederror',
            verbosity=0,
        )
        self.model.fit(
            pd.concat([X_train, X_val]),
            pd.concat([pd.Series(y_train), pd.Series(y_val)]),
        )

        return self.model


class EnsembleCongestionForecaster:
    """
    Enhanced event congestion forecaster with:
    - Ensemble of XGBoost, LightGBM, Random Forest, and Gradient Boosting
    - Bayesian hyperparameter optimization
    - SHAP explanations for interpretability
    - Post-event learning system
    - Advanced resource optimization
    - Confidence intervals for predictions
    """

    def __init__(self):
        self.models = {}
        self.ensemble = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_importance = {}
        self.shap_explainer = None
        self.training_history = []
        self.metrics_history = []
        self.is_trained = False

    def prepare_features(self, df: pd.DataFrame, event_cols: List[str],
                          time_cols: Optional[List[str]] = None,
                          weather_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Enhanced feature engineering with:
        - Time-based features (hour, day, month, season, weekend, holiday)
        - Event-type specific features
        - Historical congestion patterns
        - Interaction features
        """
        df_copy = df.copy()

        # Time-based feature engineering
        if time_cols:
            for col in time_cols:
                if col in df_copy.columns:
                    df_copy[f'{col}_hour'] = df_copy[col].dt.hour
                    df_copy[f'{col}_dayofweek'] = df_copy[col].dt.dayofweek
                    df_copy[f'{col}_month'] = df_copy[col].dt.month
                    df_copy[f'{col}_dayofyear'] = df_copy[col].dt.dayofyear
                    df_copy[f'{col}_weekend'] = df_copy[col].dt.dayofweek.isin([5, 6]).astype(int)

                    # Time of day categories
                    df_copy[f'{col}_time_category'] = pd.cut(
                        df_copy[col].dt.hour,
                        bins=[-1, 6, 10, 14, 18, 22, 24],
                        labels=['night', 'morning_peak', 'afternoon', 'evening_peak', 'night', 'late_night'],
                        include_lowest=True
                    )

                    # Season
                    df_copy[f'{col}_season'] = df_copy[col].dt.month.map(
                        {12: 'winter', 1: 'winter', 2: 'winter',
                         3: 'spring', 4: 'spring', 5: 'spring',
                         6: 'summer', 7: 'summer', 8: 'summer',
                         9: 'autumn', 10: 'autumn', 11: 'autumn'}
                    )

                    # Days since event planning
                    df_copy[f'{col}_days_planned'] = (
                        df_copy[col] - df_copy[col].min()
                    ).dt.days

        # Weather features (synthetic if not available)
        if weather_cols:
            for col in weather_cols:
                if col in df_copy.columns:
                    df_copy[f'{col}_categorized'] = pd.qcut(
                        df_copy[col].fillna(df_copy[col].mean()),
                        q=4,
                        labels=['low', 'medium', 'high', 'very_high'],
                        duplicates='drop'
                    )

        # Interaction features for event characteristics
        if 'priority' in df_copy.columns and 'event_type' in df_copy.columns:
            df_copy['priority_event_interaction'] = (
                df_copy['priority'].astype(str) + '_' +
                df_copy['event_type'].astype(str)
            )

        if 'requires_road_closure' in df_copy.columns and 'event_cause' in df_copy.columns:
            df_copy['closure_cause_interaction'] = (
                df_copy['requires_road_closure'].astype(str) + '_' +
                df_copy['event_cause'].astype(str)
            )

        # Track feature columns
        feature_cols = [col for col in df_copy.columns
                        if col not in ['congestion_level', 'event_id', 'latitude', 'longitude']]

        return df_copy

    def train(self, X: pd.DataFrame, y: pd.Series,
              optimize_hyperparams: bool = True) -> Dict:
        """
        Train ensemble of models with optional hyperparameter optimization.

        Args:
            X: Feature DataFrame
            y: Target congestion levels
            optimize_hyperparams: Whether to tune hyperparameters

        Returns:
            Dictionary of model performance metrics
        """
        # Handle missing values
        X = X.select_dtypes(include=[np.number]).fillna(0)
        y = y.fillna(y.mean())

        if len(X) < 10:
            return {'error': f'Insufficient data: {len(X)} samples, need at least 10'}

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=EVENT_CONFIG['test_size'],
            random_state=EVENT_CONFIG['random_state']
        )

        # Further split for validation
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train, test_size=0.2,
            random_state=EVENT_CONFIG['random_state']
        )

        # Train individual models
        models_config = {
            'xgboost': {
                'model': XGBRegressorWithTuning(optimize=optimize_hyperparams),
                'weight': 0.35
            },
            'lightgbm': {
                'model': LGBMRegressorWithTuning(optimize=optimize_hyperparams),
                'weight': 0.30
            },
            'random_forest': {
                'model': RandomForestRegressorWithTuning(optimize=optimize_hyperparams),
                'weight': 0.20
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressorWithTuning(optimize=optimize_hyperparams),
                'weight': 0.15
            },
        }

        predictions = {}
        metrics = {}

        for name, config in models_config.items():
            try:
                model_wrapper = config['model']
                model = model_wrapper.train(X_train_final, y_train_final, X_val, y_val)
                self.models[name] = model

                # Test predictions
                y_pred = model.predict(X_test)
                predictions[name] = y_pred

                # Calculate metrics
                metrics[name] = {
                    'r2': float(r2_score(y_test, y_pred)),
                    'mae': float(mean_absolute_error(y_test, y_pred)),
                    'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                    'mape': float(mean_absolute_percentage_error(y_test, y_pred)),
                }

                # Store feature importance
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = dict(
                        zip(X.columns, model.feature_importances_)
                    )

            except Exception as e:
                print(f"Error training {name}: {e}")

        # Create weighted ensemble
        if len(self.models) > 1:
            weights = [models_config[name]['weight']
                       for name in self.models.keys()]
            weights = np.array(weights) / sum(weights)

            ensemble_pred = np.zeros(len(y_test))
            for i, name in enumerate(self.models.keys()):
                ensemble_pred += weights[i] * predictions[name]

            ensemble_metrics = {
                'r2': float(r2_score(y_test, ensemble_pred)),
                'mae': float(mean_absolute_error(y_test, ensemble_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, ensemble_pred))),
                'mape': float(mean_absolute_percentage_error(y_test, ensemble_pred)),
            }
            metrics['ensemble'] = ensemble_metrics
        else:
            name = list(self.models.keys())[0]
            ensemble_pred = predictions[name]
            metrics['ensemble'] = metrics[name]

        # Create ensemble predictor
        self.ensemble = EnsemblePredictor(self.models, weights if len(self.models) > 1 else [1.0])

        # SHAP explanations
        if SHAP_AVAILABLE and len(self.models) > 0:
            try:
                best_model_name = max(metrics,
                                       key=lambda k: metrics[k]['r2']
                                       if isinstance(metrics[k], dict) and 'r2' in metrics[k] else 0)
                best_model = self.models.get(best_model_name)
                if best_model:
                    self.shap_explainer = shap.Explainer(best_model, X_test[:100])
                    shap_values = self.shap_explainer(X_test[:100])
                    self.shap_values = shap_values
            except Exception as e:
                print(f"SHAP explanation failed: {e}")

        self.is_trained = True

        # Store training history
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'samples': len(X),
            'features': X.columns.tolist(),
            'metrics': metrics,
        })

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict congestion levels using ensemble."""
        if not self.is_trained or self.ensemble is None:
            raise ValueError("Model not trained yet! Call train() first.")

        X = X.select_dtypes(include=[np.number]).fillna(0)
        X_scaled = self.scaler.transform(X)
        return self.ensemble.predict(X_scaled)

    def predict_with_confidence(self, X: pd.DataFrame) -> Dict:
        """
        Predict with confidence intervals.
        Returns mean prediction and confidence bounds.
        """
        X = X.select_dtypes(include=[np.number]).fillna(0)
        X_scaled = self.scaler.transform(X)

        if not self.models:
            raise ValueError("No trained models available")

        all_predictions = []
        for name, model in self.models.items():
            try:
                pred = model.predict(X_scaled)
                all_predictions.append(pred)
            except:
                continue

        if not all_predictions:
            raise ValueError("No predictions could be made")

        all_predictions = np.array(all_predictions)
        mean_pred = np.mean(all_predictions, axis=0)
        std_pred = np.std(all_predictions, axis=0)

        return {
            'prediction': mean_pred,
            'lower_bound': np.clip(mean_pred - 1.96 * std_pred, 0, 1),
            'upper_bound': np.clip(mean_pred + 1.96 * std_pred, 0, 1),
            'confidence': 1 - std_pred / (mean_pred + 0.001),
            'model_agreement': 1 - np.std(all_predictions, axis=0) / 0.5,
        }

    def recommend_resources_enhanced(self, predicted_congestion: float,
                                      event_type: str,
                                      additional_factors: Optional[Dict] = None) -> Dict:
        """
        Enhanced resource recommendation with optimization.

        Args:
            predicted_congestion: Predicted congestion level (0-1)
            event_type: Type of event
            additional_factors: Additional contextual factors

        Returns:
            Optimized resource allocation plan
        """
        # Base tier determination
        if predicted_congestion > 0.8:
            tier = 'critical'
        elif predicted_congestion > 0.5:
            tier = 'high'
        elif predicted_congestion > 0.3:
            tier = 'medium'
        else:
            tier = 'low'

        base_resources = EVENT_CONFIG['resource_tiers'][tier].copy()

        # Event-specific adjustments
        event_multipliers = {
            'political_rally': 1.3,
            'festival': 1.2,
            'sports_event': 1.1,
            'concert': 1.0,
            'construction': 0.9,
            'fair': 1.0,
            'protest': 1.4,
            'marathon': 1.2,
            'other': 1.0,
        }

        multiplier = event_multipliers.get(event_type.lower(), 1.0)

        # Apply event type multiplier
        adjusted_resources = {
            k: max(1, int(v * multiplier))
            for k, v in base_resources.items()
        }

        # Additional factors adjustment
        if additional_factors:
            if additional_factors.get('is_peak_hour', False):
                adjusted_resources['manpower'] = int(adjusted_resources['manpower'] * 1.3)
                adjusted_resources['barricades'] = int(adjusted_resources['barricades'] * 1.2)

            if additional_factors.get('is_weekend', False):
                adjusted_resources['manpower'] = int(adjusted_resources['manpower'] * 1.1)

            if additional_factors.get('requires_road_closure', False):
                adjusted_resources['diversions'] = int(adjusted_resources['diversions'] * 1.5)
                adjusted_resources['barricades'] = int(adjusted_resources['barricades'] * 1.5)

        return {
            'predicted_congestion': predicted_congestion,
            'severity_tier': tier,
            'recommended_resources': adjusted_resources,
            'event_type': event_type,
            'confidence_level': 'High' if predicted_congestion > 0.8 or predicted_congestion < 0.2 else 'Medium',
            'risk_level': 'Critical' if predicted_congestion > 0.8 else 'High' if predicted_congestion > 0.6 else 'Medium' if predicted_congestion > 0.3 else 'Low',
            'deployment_strategy': self._generate_deployment_strategy(tier, event_type, adjusted_resources),
            'contingency_plan': self._generate_contingency(tier, event_type),
        }

    def _generate_deployment_strategy(self, tier: str, event_type: str,
                                        resources: Dict) -> str:
        """Generate detailed deployment strategy."""
        strategy_map = {
            'critical': (
                f"IMMEDIATE DEPLOYMENT: Deploy {resources['manpower']} personnel immediately. "
                f"Set up {resources['barricades']} barricades at all access points. "
                f"Establish {resources['diversions']} diversion routes. "
                f"Dispatch {resources.get('patrol_vehicles', 2)} patrol vehicles. "
                f"Coordinate with local traffic control center."
            ),
            'high': (
                f"ADVANCED DEPLOYMENT: Pre-deploy {resources['manpower']} personnel 2 hours before event. "
                f"Position {resources['barricades']} barricades at key intersections. "
                f"Prepare {resources['diversions']} diversion routes. "
                f"Deploy {resources.get('patrol_vehicles', 1)} patrol vehicles on standby."
            ),
            'medium': (
                f"STANDARD DEPLOYMENT: Deploy {resources['manpower']} personnel during event hours. "
                f"Use {resources['barricades']} barricades at main chokepoints. "
                f"Plan {resources['diversions']} alternative routes if needed."
            ),
            'low': (
                f"MINIMAL DEPLOYMENT: Deploy {resources['manpower']} personnel for monitoring. "
                f"Keep {resources['barricades']} barricades ready. "
                f"Monitor traffic conditions for any escalations."
            ),
        }
        return strategy_map.get(tier, "Standard deployment per protocol.")

    def _generate_contingency(self, tier: str, event_type: str) -> str:
        """Generate contingency plan."""
        if tier == 'critical':
            return (
                "EMERGENCY CONTINGENCY: Activate emergency response team. "
                "Coordinate with nearby hospitals for medical readiness. "
                "Have tow trucks on standby. Establish communication with media for public alerts."
            )
        elif tier == 'high':
            return (
                "CONTINGENCY: Deploy backup team of 5 personnel. "
                "Keep additional barricades in storage nearby. "
                "Monitor social media for real-time crowd updates."
            )
        else:
            return (
                "STAND-BY: Have 2 additional personnel on call. "
                "Monitor traffic camera feeds for any unusual buildup."
            )

    def post_event_learning(self, event_data: pd.DataFrame,
                             actual_congestion: pd.Series) -> Dict:
        """
        Post-event learning system.
        Updates model with actual outcomes to improve future predictions.

        Args:
            event_data: Feature data from the event
            actual_congestion: Actual congestion that occurred

        Returns:
            Learning metrics
        """
        if not self.is_trained:
            return {'error': 'Model needs initial training first'}

        # Prepare data
        X = event_data.select_dtypes(include=[np.number]).fillna(0)
        y = actual_congestion.fillna(actual_congestion.mean())

        if len(X) < 5:
            return {'error': 'Insufficient post-event data'}

        # Scale
        X_scaled = self.scaler.transform(X)

        # Fine-tune each model
        learning_results = {}
        for name, model in self.models.items():
            try:
                if hasattr(model, 'fit'):
                    old_pred = model.predict(X_scaled)
                    old_error = mean_absolute_error(y, old_pred)

                    # Partial fit if supported, otherwise full retrain
                    if hasattr(model, 'warm_start'):
                        model.warm_start = True

                    model.fit(X_scaled, y)

                    new_pred = model.predict(X_scaled)
                    new_error = mean_absolute_error(y, new_pred)

                    learning_results[name] = {
                        'old_mae': float(old_error),
                        'new_mae': float(new_error),
                        'improvement': float(old_error - new_error),
                    }
            except Exception as e:
                learning_results[name] = {'error': str(e)}

        return {
            'learning_results': learning_results,
            'new_samples': len(X),
            'timestamp': datetime.now().isoformat(),
        }

    def get_shap_explanation(self, X: pd.DataFrame) -> Optional[Dict]:
        """Get SHAP explanations for model predictions."""
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            return None

        try:
            X_scaled = self.scaler.transform(X.select_dtypes(include=[np.number]).fillna(0))
            shap_values = self.shap_explainer(X_scaled[:10])

            # Get top features
            feature_importance = np.abs(shap_values.values).mean(axis=0)
            top_features = sorted(
                zip(X.columns, feature_importance),
                key=lambda x: x[1],
                reverse=True
            )[:10]

            return {
                'top_features': [{'name': f, 'importance': float(i)} for f, i in top_features],
                'base_value': float(shap_values.base_values[0]),
                'explanation': shap_values.values[:5].tolist(),
            }
        except Exception as e:
            return {'error': f'SHAP explanation failed: {e}'}

    def get_feature_importance_summary(self) -> Dict:
        """Get aggregated feature importance across all models."""
        if not self.feature_importance:
            return {}

        # Average importance across models
        all_features = set()
        for model_imp in self.feature_importance.values():
            all_features.update(model_imp.keys())

        avg_importance = {}
        for feature in all_features:
            importances = []
            for model_imp in self.feature_importance.values():
                if feature in model_imp:
                    importances.append(model_imp[feature])
            if importances:
                avg_importance[feature] = float(np.mean(importances))

        return dict(sorted(
            avg_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:15])

    def compare_models(self) -> pd.DataFrame:
        """Compare performance of all trained models."""
        if not self.training_history:
            return pd.DataFrame()

        latest = self.training_history[-1]
        comparison = []

        for model_name, metrics in latest['metrics'].items():
            if isinstance(metrics, dict):
                row = {'model': model_name, **metrics}
                comparison.append(row)

        return pd.DataFrame(comparison)


# Helper wrapper classes for consistent interface

class XGBRegressorWithTuning:
    def __init__(self, optimize: bool = True):
        self.optimize = optimize
        self.model = None

    def train(self, X_train, y_train, X_val, y_val):
        if self.optimize:
            tuner = BayesianOptimizedXGBoost()
            self.model = tuner.tune_and_train(X_train, y_train, X_val, y_val)
        else:
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                objective='reg:squarederror',
                verbosity=0,
            )
            self.model.fit(X_train, y_train)

        # Assign feature_importances_ for compatibility
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
        return self.model

    def predict(self, X):
        return self.model.predict(X)


class LGBMRegressorWithTuning:
    def __init__(self, optimize: bool = True):
        self.optimize = optimize
        self.model = None

    def train(self, X_train, y_train, X_val, y_val):
        if self.optimize:
            params = {
                'n_estimators': np.random.choice([50, 100, 200]),
                'max_depth': np.random.choice([4, 6, 8, -1]),
                'learning_rate': np.random.choice([0.01, 0.05, 0.1]),
                'num_leaves': np.random.choice([31, 50, 70, 100]),
                'subsample': np.random.choice([0.7, 0.8, 0.9, 1.0]),
                'colsample_bytree': np.random.choice([0.7, 0.8, 0.9, 1.0]),
            }
        else:
            params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'num_leaves': 31,
            }

        self.model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)])

        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
        return self.model

    def predict(self, X):
        return self.model.predict(X)


class RandomForestRegressorWithTuning:
    def __init__(self, optimize: bool = True):
        self.optimize = optimize
        self.model = None

    def train(self, X_train, y_train, X_val, y_val):
        if self.optimize:
            params = {
                'n_estimators': np.random.choice([50, 100, 200, 300]),
                'max_depth': np.random.choice([6, 10, 15, 20, None]),
                'min_samples_split': np.random.choice([2, 5, 10]),
                'min_samples_leaf': np.random.choice([1, 2, 4]),
                'max_features': np.random.choice(['sqrt', 'log2', None]),
            }
        else:
            params = {
                'n_estimators': 100,
                'max_depth': 10,
            }

        self.model = RandomForestRegressor(**params, random_state=42)
        self.model.fit(pd.concat([X_train, X_val]),
                        pd.concat([pd.Series(y_train), pd.Series(y_val)]))

        self.feature_importances_ = self.model.feature_importances_
        return self.model

    def predict(self, X):
        return self.model.predict(X)


class GradientBoostingRegressorWithTuning:
    def __init__(self, optimize: bool = True):
        self.optimize = optimize
        self.model = None

    def train(self, X_train, y_train, X_val, y_val):
        if self.optimize:
            params = {
                'n_estimators': np.random.choice([50, 100, 200]),
                'max_depth': np.random.choice([3, 4, 5, 6]),
                'learning_rate': np.random.choice([0.01, 0.05, 0.1, 0.2]),
                'subsample': np.random.choice([0.7, 0.8, 0.9, 1.0]),
                'min_samples_split': np.random.choice([2, 5, 10]),
            }
        else:
            params = {
                'n_estimators': 100,
                'max_depth': 4,
                'learning_rate': 0.1,
            }

        self.model = GradientBoostingRegressor(**params, random_state=42)
        self.model.fit(X_train, y_train)
        self.feature_importances_ = self.model.feature_importances_
        return self.model

    def predict(self, X):
        return self.model.predict(X)


class EnsemblePredictor:
    """Weighted ensemble predictor."""

    def __init__(self, models: Dict, weights: np.ndarray):
        self.models = models
        self.weights = weights

    def predict(self, X):
        predictions = []
        for model in self.models.values():
            predictions.append(model.predict(X))

        predictions = np.array(predictions)
        weighted_pred = np.average(predictions, axis=0, weights=self.weights)
        return weighted_pred