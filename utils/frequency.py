"""
Modular measurement frequency calculator for training loop.

This module provides a configurable system for determining when to perform
various measurements during training, replacing the scattered 'how_often' logic.
"""

from typing import Dict, Callable, Any
from dataclasses import dataclass


@dataclass
class MeasurementContext:
    """Container for variables needed by frequency calculation rules."""
    step_number: int
    batch_size: int
    epoch: int = 0
    initial_sharpness: float = 0.0
    sharpness_every: int = 256
    device: str = "cpu",
    lr: float = 0.01
    precise_plots: bool = False
    # Rare-measure regime: sparsify expensive measurements
    rare_measure: bool = False
    # Force early full loss logging
    steps_since_restart: int = 0
    # Add other variables as needed


class FrequencyCalculator:
    """
    Configurable frequency calculator for training measurements.
    
    Each measurement type has a rule function that takes a MeasurementContext
    and returns True if the measurement should be performed at this step.
    """
    
    def __init__(self):
        self.rules: Dict[str, Callable[[MeasurementContext], bool]] = {}
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default frequency rules matching current behavior."""
        
        def _rare_scale(ctx: MeasurementContext, freq: int, heavy: bool = False) -> int:
            """Increase measurement frequency interval under rare-measure regime.

            - heavy=True applies stronger sparsification (8x) vs light (4x).
            - If precise_plots is requested, do not sparsify (precise plots win).
            """
            if getattr(ctx, 'rare_measure', False) and not getattr(ctx, 'precise_plots', False):
                # scale = 8 if heavy else 4
                scale = 4 if heavy else 2
                return max(1, int(freq * scale))
            return freq
        
        def full_batch_lambda_max_rule(ctx: MeasurementContext) -> bool:
            """Full batch lambda max frequency rule."""
            # Base frequency depends on batch size
            if ctx.batch_size <= 33:
                base_freq = 256
            else:
                base_freq = 256

            if ctx.precise_plots:
                base_freq = min(base_freq, 32)

            
            # Reduce frequency as training progresses
            freq = base_freq
            if ctx.step_number > 10_000:
                freq *= 2
            if ctx.step_number > 30_000:
                freq *= 2
            # if ctx.step_number > 100_000:
                # freq *= 2
            
            
            # Rare-measure: make much rarer
            freq = _rare_scale(ctx, freq, heavy=True)

            return ctx.step_number % freq == 0
        
        def full_batch_lambda_max_early_rule(ctx: MeasurementContext) -> bool:
            """Full batch lambda max with early frequent measurements."""
            # FIRST_FEW = 256
            # FIRST_SUPER_FEW = 128
            if ctx.batch_size < 33:
                base_freq = 128
            elif ctx.batch_size < 128:
                base_freq = 64
            else:
                base_freq = 32

            if ctx.step_number < 10 / ctx.lr:
                if ctx.batch_size < 33:
                    base_freq = 64
                else:
                    base_freq = 16
            
            freq = base_freq
            if ctx.step_number > 10_000:
                freq *= 2
            if ctx.step_number > 30_000:
                freq *= 2
            if ctx.step_number > 50_000:
                freq *= 2

            # Rare-measure: make much rarer
            freq = _rare_scale(ctx, freq, heavy=True)

            return ctx.step_number % freq == 0
        



            
            # if ctx.step_number < FIRST_SUPER_FEW:
            #     return True
            # if ctx.step_number < FIRST_FEW:
            #     return ctx.step_number % 4 == 0
                
            # Fall back to regular rule
            # return full_batch_lambda_max_rule(ctx)
        
        def batch_lambda_max_rule(ctx: MeasurementContext) -> bool:
            """Batch lambda max frequency rule."""

            # Base frequency depends on batch size
            if ctx.batch_size <= 33:
                base_freq = 512
            else:
                base_freq = 128

            if ctx.precise_plots:
                base_freq = min(base_freq, 32)

            
            # Reduce frequency as training progresses
            freq = base_freq
            if ctx.step_number > 10_000:
                freq *= 2
            if ctx.step_number > 30_000:
                freq *= 2
            # if ctx.step_number > 100_000:
                # freq *= 2
            
            
            # Rare-measure: make much rarer
            freq = _rare_scale(ctx, freq, heavy=True)

            return ctx.step_number % freq == 0

        def full_batch_lambda_max_gn_rule(ctx: MeasurementContext) -> bool:
            """Gauss-Newton analog of full batch lambda max; mirrors Hessian cadence."""
            return full_batch_lambda_max_rule(ctx)

        def step_batch_lambda_max_rule(ctx: MeasurementContext) -> bool:
            """Lambda max on the live mini-batch (single batch)."""
            if ctx.batch_size <= 33:
                base_freq = 32
            else:
                base_freq = 32

            if ctx.step_number < max(1, int(10 / max(ctx.lr, 1e-8))):
                base_freq = min(base_freq, 16)

            freq = base_freq
            if ctx.step_number > 10_000:
                freq *= 2
            if ctx.step_number > 30_000:
                freq *= 2

            freq = _rare_scale(ctx, freq, heavy=True)
            return ctx.step_number % max(1, freq) == 0
        
            # # if ctx.batch_size > 32:
            # #     base_freq = 16
            # # else:
            # #     base_freq = 32
            # #     if ctx.batch_size > 16:
            # #         base_freq = 64


            
            # freq = base_freq
            # # Rare-measure: lightly sparsify batch variant
            # freq = _rare_scale(ctx, freq, heavy=False)
            # return ctx.step_number % freq == 0
        
        def batch_sharpness_rule(ctx: MeasurementContext) -> bool:
            """Batch sharpness frequency rule (expected Rayleigh quotient)."""
            if ctx.batch_size < 33:
                base_freq = 128
            else:
                base_freq = 128
            
            freq = base_freq
            if ctx.step_number > 10_000:
                freq *= 2
            if ctx.step_number > 50_000:
                freq *= 2
            if ctx.step_number > 100_000:
                freq *= 2
                
            # Preferred batch sharpness; light sparsification only
            freq = _rare_scale(ctx, freq, heavy=True)
            return ctx.step_number % freq == 0

        def batch_sharpness_gn_rule(ctx: MeasurementContext) -> bool:
            """Gauss-Newton batch sharpness cadence; reuses Hessian schedule."""
            return batch_sharpness_rule(ctx)
        
        def momentum_batch_sharpness_rule(ctx: MeasurementContext) -> bool:
            """Momentum-aware batch sharpness cadence; mirrors batch sharpness."""
            return batch_sharpness_rule(ctx)
        

        def second_moment_contraction_rule(ctx: MeasurementContext) -> bool:
            # Base frequency depends on batch size
            if ctx.batch_size <= 33:
                base_freq = 256
            else:
                base_freq = 256

            if ctx.precise_plots:
                base_freq = min(base_freq, 32)

            # Reduce frequency as training progresses
            freq = base_freq
            if ctx.step_number > 10_000:
                freq *= 2
            if ctx.step_number > 30_000:
                freq *= 2
            # if ctx.step_number > 100_000:
                # freq *= 2
            
            
            # Rare-measure: make much rarer
            freq = _rare_scale(ctx, freq, heavy=True)

            return ctx.step_number % freq == 0
        
        
        def step_sharpness_rule(ctx: MeasurementContext) -> bool:
            """Step sharpness frequency rule (single batch Rayleigh quotient)."""
            freq = 16
            # Keep cheap step sharpness largely unchanged in rare mode
            return ctx.step_number % freq == 0


        

        
        def batch_sharpness_exp_inside_rule(ctx: MeasurementContext) -> bool:
            """Batch sharpness (but with expectation inside) frequency rule."""
            if ctx.batch_size < 33:
                base_freq = 128
            else:
                base_freq = 64
            
            freq = base_freq
            if ctx.step_number > 10_000:
                freq *= 2
            if ctx.step_number > 30_000:
                freq *= 2
                
            # Rare-measure: averaging across many batches is costly
            freq = _rare_scale(ctx, freq, heavy=True)
            return ctx.step_number % freq == 0

        
        def full_ghg_rule(ctx: MeasurementContext) -> bool:
            """Full GHG measurement frequency rule."""
            if ctx.batch_size > 32:
                base_freq = 256
            else:
                base_freq = 512
            
            freq = base_freq
            if ctx.step_number > 10_000:
                freq *= 2
            # Rare-measure: full-dataset gHg is expensive; sparsify strongly
            freq = _rare_scale(ctx, freq, heavy=True)
            return ctx.step_number % freq == 0

        def hessian_trace_rule(ctx: MeasurementContext) -> bool:
            """Full-batch Hessian trace estimation cadence."""
            if ctx.batch_size < 33:
                base_freq = 256
            else:
                base_freq = 64

            if ctx.step_number > 10_000:
                base_freq *= 2
            if ctx.step_number > 50_000:
                base_freq *= 2

            base_freq = _rare_scale(ctx, base_freq, heavy=True)
            return ctx.step_number % base_freq == 0

        def batch_ghg_rule(ctx: MeasurementContext) -> bool:
            """Batch GHG measurement frequency rule."""
            freq = 8
            # Light sparsification only for batch version
            freq = _rare_scale(ctx, freq, heavy=False)
            return ctx.step_number % freq == 0
        
        def fisher_total_rule(ctx: MeasurementContext) -> bool:
            """Fisher information total frequency rule."""
            if ctx.batch_size > 32:
                base_freq = 64 * 2
            else:
                base_freq = 128 * 2
            
            freq = base_freq
            if ctx.step_number > 10_000:
                freq *= 2
            if ctx.step_number > 20_000:
                freq *= 2
            # Rare-measure: total Fisher is expensive; sparsify strongly
            freq = _rare_scale(ctx, freq, heavy=True)
            return ctx.step_number % freq == 0
        
        def fisher_batch_rule(ctx: MeasurementContext) -> bool:
            """Fisher information batch frequency rule."""
            if ctx.batch_size > 32:
                base_freq = 8
            else:
                base_freq = 4
            
            freq = base_freq
            if ctx.step_number > 10_000:
                freq *= 2
            # Light sparsification only
            freq = _rare_scale(ctx, freq, heavy=False)
            return ctx.step_number % freq == 0
        
        def gni_rule(ctx: MeasurementContext) -> bool:
            """Gradient-noise interaction frequency rule."""
            if ctx.batch_size < 33:
                base_freq = 256
            else:
                base_freq = 64

            # temp override for CNN
            base_freq = 2500
            
            # temp override for the run!
            # if ctx.step_number < 257:
            #     return ctx.step_number % 8 == 0

            # if ctx.step_number < 10 / ctx.lr:
            #     if ctx.batch_size < 33:
            #         base_freq = 64
            #     else:
            #         base_freq = 32
                
            #     if ctx.lr > 0.005:
            #         base_freq = base_freq / 2
            
            if ctx.step_number > 10_000:
                base_freq *= 4
            if ctx.step_number > 50_000:
                base_freq *= 2
            # Rare-measure: GNI heavy; sparsify strongly
            base_freq = _rare_scale(ctx, base_freq, heavy=True)
            return ctx.step_number % base_freq == 0
        
        def param_distance_rule(ctx: MeasurementContext) -> bool:
            """Parameter distance measurement frequency rule."""
            return ctx.step_number % 1 == 0  # Every step
        
        def checkpoint_rule(ctx: MeasurementContext) -> bool:
            """Checkpoint saving frequency rule."""
            return ctx.step_number % 32 == 0
        
        def full_loss_rule(ctx: MeasurementContext) -> bool:
            """Full loss calculation frequency rule for --force-full-loss."""
            warmup_steps = getattr(ctx, 'full_loss_warmup_steps', 0)
            if getattr(ctx, 'force_full_loss_warmup', False) and ctx.steps_since_restart < warmup_steps:
                return True

            freq = 32
            if ctx.step_number > 10_000:
                freq *= 2
            if ctx.step_number > 50_000:
                freq *= 2

            freq = _rare_scale(ctx, freq, heavy=True)
            freq = max(1, freq)
            return ctx.step_number % freq == 0
    
        def full_loss_warmup_rule(ctx: MeasurementContext) -> bool:
            """Full loss warmup measurement frequency rule."""
            warmup_steps = 128

            if ctx.steps_since_restart < warmup_steps:
                return True

            return False

        def gradient_norm_squared_rule(ctx: MeasurementContext) -> bool:
            """Gradient norm squared measurement frequency rule."""

            if ctx.batch_size > 32:
                base_freq = 128
            else:
                base_freq = 64
            
            freq = base_freq
            if ctx.step_number > 10_000:
                freq *= 2
            # Rare-measure: can be expensive depending on estimator
            freq = _rare_scale(ctx, freq, heavy=False)
            return ctx.step_number % freq == 0
        
        def one_step_loss_change_rule(ctx: MeasurementContext) -> bool:
            """Expected one-step loss change measurement frequency rule."""
            if ctx.batch_size < 33:
                base_freq = 256
            else:
                base_freq = 64

            if ctx.step_number < 10 / ctx.lr:
                if ctx.batch_size < 33:
                    base_freq = 64
                else:
                    base_freq = 32
                    
            
            freq = base_freq
            if ctx.step_number > 10_000:
                freq *= 2
            if ctx.step_number > 30_000:
                freq *= 2
            # Rare-measure: Monte Carlo expectation is heavy
            freq = _rare_scale(ctx, freq, heavy=True)
            return ctx.step_number % freq == 0

        def grad_projection_rule(ctx: MeasurementContext) -> bool:
            """Gradient projection frequency rule (full-batch gradient projection).

            Independent schedule (not tied to lmax).
            - Smaller batches: measure sparser (full-batch gradient is costly)
            - Later in training: measure even sparser
            """
            if ctx.batch_size < 33:
                base_freq = 32
            else:
                base_freq = 8

            freq = base_freq
            if ctx.step_number > 10_000:
                freq *= 2
            if ctx.step_number > 50_000:
                freq *= 2
            if ctx.step_number > 100_000:
                freq *= 2

            if ctx.precise_plots and ctx.step_number < 400:
                # Be more frequent early on if precise plotting requested
                freq = min(freq, 16)
            # Rare-measure: full-batch gradient + projections are expensive
            freq = _rare_scale(ctx, freq, heavy=True)
            return ctx.step_number % freq == 0

        def proj_eigens_refresh_rule(ctx: MeasurementContext) -> bool:
            """Refresh cadence for projection eigendirections.

            Default: refresh every step. Exposed as a rule so cadence can be
            tuned centrally later without changing training code.
            """
            return True

        def trajectory_projection_rule(ctx: MeasurementContext) -> bool:
            """Trajectory projection logging cadence (fixed at every 32 steps)."""
            return ctx.step_number % 32 == 0
        
        # Register all default rules
        self.rules.update({
            'full_batch_lambda_max': full_batch_lambda_max_rule,
            'full_batch_lambda_max_early': full_batch_lambda_max_early_rule,
            'full_batch_lambda_max_gn': full_batch_lambda_max_gn_rule,
            'second_moment_contraction': second_moment_contraction_rule,
            'batch_lambda_max': batch_lambda_max_rule,
            'step_batch_lambda_max': step_batch_lambda_max_rule,
            'step_sharpness': step_sharpness_rule,
            'batch_sharpness': batch_sharpness_rule,
            'batch_sharpness_gn': batch_sharpness_gn_rule,
            'momentum_batch_sharpness': momentum_batch_sharpness_rule,
            'batch_sharpness_exp_inside': batch_sharpness_exp_inside_rule,
            'full_ghg': full_ghg_rule,
            'hessian_trace': hessian_trace_rule,
            'batch_ghg': batch_ghg_rule,
            'fisher_total': fisher_total_rule,
            'fisher_batch': fisher_batch_rule,
            'gni': gni_rule,
            'param_distance': param_distance_rule,
            'checkpoint': checkpoint_rule,
            'full_loss': full_loss_rule,
            'full_loss_warmup': full_loss_warmup_rule,
            'gradient_norm_squared': gradient_norm_squared_rule,
            'one_step_loss_change': one_step_loss_change_rule,
            'grad_projection': grad_projection_rule,
            'proj_eigens_refresh': proj_eigens_refresh_rule,
            'trajectory_projection': trajectory_projection_rule,
        })
    
    def should_measure(self, measurement_type: str, ctx: MeasurementContext) -> bool:
        """
        Check if a measurement should be performed at this step.
        
        Args:
            measurement_type: Name of the measurement type
            ctx: Context containing step_number, batch_size, etc.
            
        Returns:
            True if measurement should be performed
        """
        if measurement_type not in self.rules:
            raise ValueError(f"Unknown measurement type: {measurement_type}")
        
        return self.rules[measurement_type](ctx)
    
    def set_rule(self, measurement_type: str, rule_func: Callable[[MeasurementContext], bool]):
        """
        Set a custom frequency rule for a measurement type.
        
        Args:
            measurement_type: Name of the measurement type
            rule_func: Function that takes MeasurementContext and returns bool
        """
        self.rules[measurement_type] = rule_func
    
    def get_available_measurements(self) -> list:
        """Get list of available measurement types."""
        return list(self.rules.keys())


# Global instance that can be configured
frequency_calculator = FrequencyCalculator()


# Convenience functions for easy migration from existing code
def should_measure_full_batch_lambda_max(step_number: int, batch_size: int, early_measure: bool = False) -> bool:
    """Convenience function for full batch lambda max frequency."""
    ctx = MeasurementContext(step_number=step_number, batch_size=batch_size)
    measurement_type = 'full_batch_lambda_max_early' if early_measure else 'full_batch_lambda_max'
    return frequency_calculator.should_measure(measurement_type, ctx)


def should_measure_batch_lambda_max(step_number: int, batch_size: int) -> bool:
    """Convenience function for batch lambda max frequency."""
    ctx = MeasurementContext(step_number=step_number, batch_size=batch_size)
    return frequency_calculator.should_measure('batch_lambda_max', ctx)


def should_measure_step_sharpness(step_number: int, batch_size: int) -> bool:
    """Convenience function for step sharpness frequency."""
    ctx = MeasurementContext(step_number=step_number, batch_size=batch_size)
    return frequency_calculator.should_measure('step_sharpness', ctx)


def should_measure_batch_sharpness(step_number: int, batch_size: int) -> bool:
    """Convenience function for batch sharpness frequency."""
    ctx = MeasurementContext(step_number=step_number, batch_size=batch_size)
    return frequency_calculator.should_measure('batch_sharpness', ctx)


def should_measure_gni(step_number: int, batch_size: int) -> bool:
    """Convenience function for GNI frequency."""
    ctx = MeasurementContext(step_number=step_number, batch_size=batch_size)
    return frequency_calculator.should_measure('gni', ctx)


def should_save_checkpoint(step_number: int) -> bool:
    """Convenience function for checkpoint frequency."""
    ctx = MeasurementContext(step_number=step_number, batch_size=1)  # batch_size doesn't matter for checkpoints
    return frequency_calculator.should_measure('checkpoint', ctx)
