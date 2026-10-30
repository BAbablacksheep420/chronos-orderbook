"""
Chronos Order Book - Calibrated Forecasters & Conformal Selective Prediction
=============================================================================
Implements:
- DeepLOB / Dilated Causal TCN and MLP / Gradient Boosting / OFI Baselines
- Temperature Scaling for probability calibration (Guo et al. 2017)
- Adaptive Conformal Inference (ACI) for drift-robust coverage guarantees (Gibbs & Candès 2021)
- Validation-calibrated Risk-Coverage curves for optimal learned abstention thresholds
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from scipy.optimize import minimize

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier


if HAS_TORCH:
    class ChausalConv1d(nn.Module):
        """1D Causal Convolution with dilation for temporal sequence modeling."""
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
            super().__init__()
            self.padding = (kernel_size - 1) * dilation
            self.conv = nn.Conv1d(
                in_channels, out_channels, kernel_size,
                padding=self.padding, dilation=dilation
            )
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.conv(x)
            if self.padding > 0:
                out = out[:, :, :-self.padding]
            return out

    class TemporalBlock(nn.Module):
        """Residual block composed of two causal dilated convolutions."""
        def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, dilation: int, dropout: float = 0.2):
            super().__init__()
            self.conv1 = ChausalConv1d(n_inputs, n_outputs, kernel_size, dilation=dilation)
            self.relu1 = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout)
            
            self.conv2 = ChausalConv1d(n_outputs, n_outputs, kernel_size, dilation=dilation)
            self.relu2 = nn.ReLU()
            self.dropout2 = nn.Dropout(dropout)
            
            self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
            self.relu = nn.ReLU()
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            res = x if self.downsample is None else self.downsample(x)
            out = self.dropout1(self.relu1(self.conv1(x)))
            out = self.dropout2(self.relu2(self.conv2(out)))
            return self.relu(out + res)

    class TCNForecaster(nn.Module):
        """Temporal Convolutional Network for 3-class LOB Price Direction Forecasting."""
        def __init__(self, n_features: int, num_classes: int = 3, num_channels: Optional[list] = None, kernel_size: int = 3, dropout: float = 0.2):
            super().__init__()
            if num_channels is None:
                num_channels = [32, 64]
                
            layers = []
            num_levels = len(num_channels)
            for i in range(num_levels):
                dilation_size = 2 ** i
                in_ch = n_features if i == 0 else num_channels[i - 1]
                out_ch = num_channels[i]
                layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation=dilation_size, dropout=dropout))
                
            self.network = nn.Sequential(*layers)
            self.fc = nn.Linear(num_channels[-1], num_classes)
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 2:
                x = x.unsqueeze(-1)
            y = self.network(x)
            out = self.fc(y[:, :, -1])
            return out
else:
    TCNForecaster = None


class TemperatureScaler:
    """Post-hoc probability calibration via Platt scaling / Temperature Scaling."""
    def __init__(self):
        self.temperature = 1.0
        
    def fit(self, logits: np.ndarray, y_true: np.ndarray):
        """Fit single scalar temperature T > 0 on validation split to minimize Cross-Entropy."""
        def nll(t):
            t_val = max(1e-3, t[0])
            scaled_logits = logits / t_val
            # Stable log-sum-exp
            max_l = np.max(scaled_logits, axis=1, keepdims=True)
            log_sum_exp = max_l + np.log(np.sum(np.exp(scaled_logits - max_l), axis=1, keepdims=True))
            log_probs = scaled_logits - log_sum_exp
            nll_loss = -np.mean(log_probs[np.arange(len(y_true)), y_true])
            return nll_loss
            
        res = minimize(nll, x0=[1.0], bounds=[(0.05, 5.0)], method='L-BFGS-B')
        self.temperature = float(res.x[0])
        return self
        
    def scale_logits(self, logits: np.ndarray) -> np.ndarray:
        return logits / self.temperature


class LOBForecaster:
    """
    Unified multi-model forecaster supporting:
    - 'tcn': Deep Causal Temporal Convolutional Network
    - 'mlp': Multi-Layer Perceptron baseline
    - 'gbdt': Gradient Boosted Decision Trees
    - 'ofi_linear': Linear Logistic Regression on Order Flow Imbalance
    """
    def __init__(self, model_type: str = "tcn", n_classes: int = 3):
        self.model_type = model_type
        self.n_classes = n_classes
        self.scaler = StandardScaler()
        self.temp_scaler = TemperatureScaler()
        self.is_fitted = False
        self.nn_model: Optional[Any] = None
        self.sk_model: Optional[Any] = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 25, lr: float = 0.003, val_split: float = 0.2):
        """Train model and calibrate temperature on validation partition."""
        n_val = max(30, int(len(X) * val_split))
        X_train, y_train = X[:-n_val], y[:-n_val]
        X_val, y_val = X[-n_val:], y[-n_val:]
        
        X_train_sc = self.scaler.fit_transform(X_train)
        X_val_sc = self.scaler.transform(X_val)
        self.is_fitted = True
        
        # Compute balanced class weights to prevent stationary class dominance
        from sklearn.utils.class_weight import compute_class_weight
        unique_classes = np.unique(y_train)
        if len(unique_classes) == self.n_classes:
            class_weights = compute_class_weight("balanced", classes=unique_classes, y=y_train)
        else:
            class_weights = np.ones(self.n_classes)
            
        if self.model_type == "tcn" and HAS_TORCH:
            n_features = X.shape[1]
            self.nn_model = TCNForecaster(n_features=n_features, num_classes=self.n_classes)
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
            criterion = nn.CrossEntropyLoss(weight=weights_tensor)
            optimizer = optim.Adam(self.nn_model.parameters(), lr=lr, weight_decay=1e-4)
            
            tensor_x = torch.tensor(X_train_sc, dtype=torch.float32)
            tensor_y = torch.tensor(y_train, dtype=torch.long)
            dataset = TensorDataset(tensor_x, tensor_y)
            loader = DataLoader(dataset, batch_size=64, shuffle=True)
            
            self.nn_model.train()
            for _ in range(epochs):
                for bx, by in loader:
                    optimizer.zero_grad()
                    out = self.nn_model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()
                    
            # Compute validation logits for temperature scaling
            self.nn_model.eval()
            with torch.no_grad():
                val_logits = self.nn_model(torch.tensor(X_val_sc, dtype=torch.float32)).cpu().numpy()
            self.temp_scaler.fit(val_logits, y_val)
            
        elif self.model_type == "mlp":
            self.sk_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
            self.sk_model.fit(X_train_sc, y_train)
        elif self.model_type == "ofi_linear":
            self.sk_model = LogisticRegression(max_iter=200, C=1.0, class_weight="balanced", random_state=42)
            self.sk_model.fit(X_train_sc, y_train)
        else:
            # Gradient Boosting baseline with sample weights
            sample_w = np.array([class_weights[int(c)] for c in y_train])
            self.sk_model = GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.08, random_state=42)
            self.sk_model.fit(X_train_sc, y_train, sample_weight=sample_w)
            
        return self

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted.")
        X_sc = self.scaler.transform(X)
        if self.model_type == "tcn" and HAS_TORCH and self.nn_model is not None:
            self.nn_model.eval()
            with torch.no_grad():
                return self.nn_model(torch.tensor(X_sc, dtype=torch.float32)).cpu().numpy()
        elif hasattr(self.sk_model, "decision_function"):
            df = self.sk_model.decision_function(X_sc)
            if df.ndim == 1:
                df = np.column_stack([-df, df])
            return df
        else:
            probs = self.sk_model.predict_proba(X_sc)
            return np.log(np.clip(probs, 1e-7, 1.0 - 1e-7))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self.predict_logits(X)
        scaled = self.temp_scaler.scale_logits(logits)
        # Softmax
        exp_l = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return exp_l / np.sum(exp_l, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


class AdaptiveConformalSelectiveHead:
    """
    Adaptive Conformal Inference (ACI) and learned selective prediction wrapper.
    Implements:
    1. Optimal abstention threshold:
       tau* = argmax { Coverage(tau) : Precision(tau) >= Target }
       Equivalent to: inf { tau : Precision(tau) >= Target } — lowest tau = highest coverage.
       NOTE: This is a point estimate on validation data. For formal guarantees, use
       conformal risk control (Angelopoulos et al.) or Geifman & El-Yaniv bounds.
    2. Conformal prediction sets C_t(alpha_t) = { k : 1 - P_hat(k|x_t) <= q_hat(1 - alpha_t) }.
       NOTE: This is a simplified APS-like score. Full RAPS, SAOCP, or PID-based
       variants (Angelopoulos et al. 2024) would provide tighter sets.
    3. Gibbs & Candes (2021) online ACI quantile updates with delayed feedback (h-ticks):
       err_{t-h} = 1[y_{t-h} NOT IN C_{t-h}(alpha_{t-h})]  (set miscoverage, not misclassification)
       alpha_{t+1} = alpha_t + gamma * (alpha* - err_{t-h}).
    4. Conformal selective execution: trade ONLY if C_t is singleton {Up} or {Down},
       abstaining if set is ambiguous (|C_t| > 1), stationary {1}, or empty.
    
    Limitations (acknowledged post-review):
    - At ~25% base accuracy, alpha_t tends to slam toward its floor because most
      predictions miss. The ACI update then mostly tightens sets, making the gate
      extremely conservative regardless of regime.
    - Label delay of h=5 ticks is short; real feedback loops may be longer.
    - Better variants: AgACI, DtACI, SF-OGD, SAOCP, ACI+PID.
    """
    def __init__(
        self,
        base_model: LOBForecaster,
        target_error_rate: float = 0.20,
        aci_step_size: float = 0.01,
        horizon: int = 5
    ):
        self.base_model = base_model
        self.target_error = target_error_rate
        self.gamma = aci_step_size
        self.horizon = horizon
        self.conformal_alpha_t = target_error_rate
        self.optimal_tau: Optional[float] = 0.45
        self.precision_target_met: bool = False
        self.cal_scores: np.ndarray = np.array([0.5])
        self.pending_queue: List[Tuple[List[int], float]] = []
        
    def calibrate_conformal(self, X_val: np.ndarray, y_val: np.ndarray, target_precision: float = 0.65):
        """
        Calibrate non-conformity scores and sweep tau* on validation slice.
        
        tau* = inf { tau in [0.35, 0.90] : Precision(tau) >= target_precision }
        The infimum gives the LOWEST threshold = HIGHEST coverage that still meets precision.
        This is equivalent to argmax Coverage s.t. Precision >= target.
        """
        probs = self.base_model.predict_proba(X_val)
        n_val = len(y_val)
        # Non-conformity score: s_i = 1 - P_hat(y_i | x_i)
        true_class_probs = probs[np.arange(n_val), y_val]
        self.cal_scores = np.sort(1.0 - true_class_probs)
        
        # Risk-coverage sweep: tau* = inf { tau : Precision(tau) >= target_precision }
        confidences = np.max(probs, axis=1)
        preds = np.argmax(probs, axis=1)
        
        candidate_taus = np.linspace(0.35, 0.90, 30)
        found_tau = None
        for tau in candidate_taus:
            accepted = (confidences >= tau)
            if np.sum(accepted) >= 15:
                active = accepted & (preds != 1)
                if np.sum(active) >= 10:
                    acc = np.mean(preds[active] == y_val[active])
                    if acc >= target_precision:
                        found_tau = float(tau)
                        break
                        
        if found_tau is not None:
            self.optimal_tau = found_tau
            self.precision_target_met = True
        else:
            self.optimal_tau = 0.85
            self.precision_target_met = False
            
        return self.optimal_tau

    def calibrate_risk_coverage(self, X_val: np.ndarray, y_val: np.ndarray, target_accuracy: float = 0.65):
        """Backwards compatibility alias for calibrate_conformal."""
        return self.calibrate_conformal(X_val, y_val, target_precision=target_accuracy)

    def get_conformal_set(self, probs: np.ndarray, alpha: float) -> List[int]:
        """Evaluate conformal prediction set C_t(alpha) = { k : 1 - P_hat(k|x) <= q_hat(1 - alpha) }."""
        q_level = float(np.clip(1.0 - alpha, 0.01, 0.99))
        q_val = float(np.quantile(self.cal_scores, q_level))
        return [k for k in range(len(probs)) if (1.0 - probs[k]) <= q_val]

    def predict_step(
        self,
        x_vector: np.ndarray,
        confidence_multiplier: float = 1.0,
        ground_truth_y: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Predict single tick with adaptive conformal selective execution.
        """
        x_2d = x_vector.reshape(1, -1) if x_vector.ndim == 1 else x_vector
        calibrated_probs = self.base_model.predict_proba(x_2d)[0]
        raw_pred = int(np.argmax(calibrated_probs))
        conviction = float(np.max(calibrated_probs))
        
        c = float(np.clip(confidence_multiplier, 0.20, 1.0))
        
        # 1. Conformal prediction set at current alpha_t
        conf_set = self.get_conformal_set(calibrated_probs, self.conformal_alpha_t)
        
        # 2. Decision rule: Trade ONLY if singleton {0} (Down) or {2} (Up)
        # Gated by detector state: if confidence drops below 0.80, abstain
        if conf_set == [2] and c >= 0.80:
            action_class = 2  # Up (Long)
            abstained = False
        elif conf_set == [0] and c >= 0.80:
            action_class = 0  # Down (Short)
            abstained = False
        else:
            action_class = 1  # Stationary / Neutral (Abstain)
            abstained = bool(raw_pred in [0, 2])
            
        # 3. Queue prediction for h-step delayed feedback
        self.pending_queue.append((conf_set, float(self.conformal_alpha_t)))
        
        # 4. Process delayed feedback: label arrives h ticks late
        if ground_truth_y is not None:
            if len(self.pending_queue) >= self.horizon:
                delayed_set, delayed_alpha = self.pending_queue.pop(0)
                # SET MISCOVERAGE (not misclassification): err = 1 if y not in C, else 0
                # This is the correct ACI error term per Gibbs & Candès (2021).
                set_err = 1.0 if (ground_truth_y not in delayed_set) else 0.0
                # Gibbs & Candès (2021) ACI online quantile update
                # Clip bounds [0.005, 0.995] are deliberately wide to preserve the
                # marginal coverage guarantee. Tighter clips (e.g. [0.02, 0.95]) void it.
                self.conformal_alpha_t = float(np.clip(
                    self.conformal_alpha_t + self.gamma * (self.target_error - set_err),
                    0.005, 0.995
                ))
                
        entropy = -float(np.sum(calibrated_probs * np.log(calibrated_probs + 1e-9)))
        
        return {
            "calibrated_probs": calibrated_probs,
            "pred_class": action_class,
            "raw_pred": raw_pred,
            "conviction": conviction,
            "conformal_set": conf_set,
            "set_size": len(conf_set),
            "abstained": abstained,
            "entropy": entropy,
            "conformal_alpha": float(self.conformal_alpha_t)
        }
