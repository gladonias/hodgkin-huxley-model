import numpy as np
from scipy.integrate import odeint
from scipy import signal

import original_hh as ohh
import linearized_hh as lhh
import transfer_function_hh as tfhh


class TestGatingKinetics:

    def test_alpha_m_at_singularity(self):
        result = ohh.alpha_m(-40.0)
        assert np.isfinite(result)
        np.testing.assert_approx_equal(float(result), 1.0, significant=5)

    def test_alpha_n_at_singularity(self):
        result = ohh.alpha_n(-55.0)
        assert np.isfinite(result)
        np.testing.assert_approx_equal(float(result), 0.1, significant=5)

    def test_alpha_m_away_from_singularity(self):
        result = ohh.alpha_m(-65.0)
        assert np.isfinite(result)
        assert float(result) > 0

    def test_alpha_n_away_from_singularity(self):
        result = ohh.alpha_n(-65.0)
        assert np.isfinite(result)
        assert float(result) > 0

    def test_gating_rates_positive(self):
        for V in [-80.0, -65.0, -40.0, 0.0, 20.0]:
            assert float(ohh.alpha_m(V)) >= 0
            assert float(ohh.beta_m(V)) >= 0
            assert float(ohh.alpha_h(V)) >= 0
            assert float(ohh.beta_h(V)) >= 0
            assert float(ohh.alpha_n(V)) >= 0
            assert float(ohh.beta_n(V)) >= 0

    def test_steady_state_bounded(self):
        for V in [-80.0, -65.0, -40.0, 0.0, 20.0]:
            assert 0 <= float(ohh.m_inf(V)) <= 1
            assert 0 <= float(ohh.h_inf(V)) <= 1
            assert 0 <= float(ohh.n_inf(V)) <= 1

    def test_vectorized_input(self):
        V = np.array([-80.0, -65.0, -55.0, -40.0, 0.0, 20.0])
        result_m = ohh.alpha_m(V)
        result_n = ohh.alpha_n(V)
        assert result_m.shape == V.shape
        assert result_n.shape == V.shape
        assert np.all(np.isfinite(result_m))
        assert np.all(np.isfinite(result_n))


class TestOriginalModel:

    def test_action_potential_generated(self):
        t = np.arange(0.0, 400.0, 0.1)
        X = odeint(ohh.dALLdt, [-65, 0.05, 0.6, 0.32], t)
        V = X[:, 0]
        assert np.max(V) > 0, "Model should produce action potentials above 0 mV"

    def test_resting_potential_stable(self):
        t = np.arange(0.0, 50.0, 0.1)
        X = odeint(ohh.dALLdt, [-65, 0.05, 0.6, 0.32], t)
        V = X[:, 0]
        assert np.all(V[:100] < -60), "Voltage should remain near rest before stimulus"

    def test_gating_variables_bounded(self):
        t = np.arange(0.0, 400.0, 0.1)
        X = odeint(ohh.dALLdt, [-65, 0.05, 0.6, 0.32], t)
        for i in range(1, 4):
            assert np.all(X[:, i] >= -0.01), f"Gating variable {i} went below 0"
            assert np.all(X[:, i] <= 1.01), f"Gating variable {i} went above 1"

    def test_current_injection(self):
        assert ohh.I_inj(50) == 0
        assert ohh.I_inj(150) == 10
        assert ohh.I_inj(350) == 0


class TestLinearizedModel:

    def test_runs_without_error(self):
        t = np.arange(0.0, 400.0, 0.1)
        X = odeint(lhh.Linear_dALLdt, [0, 0.0529, 0.5961, 0.3177], t)
        assert X.shape == (4000, 4)
        assert np.all(np.isfinite(X))

    def test_linearized_gating_kinetics_match(self):
        for V in [-80.0, -65.0, 0.0]:
            np.testing.assert_approx_equal(
                float(lhh.alpha_m(V)), float(ohh.alpha_m(V)), significant=5
            )
            np.testing.assert_approx_equal(
                float(lhh.alpha_n(V)), float(ohh.alpha_n(V)), significant=5
            )


class TestTransferFunction:

    def test_transfer_function_defined(self):
        assert isinstance(tfhh.tf_sys, signal.TransferFunction)

    def test_bode_computable(self):
        w, mag, phase = signal.bode(tfhh.tf_sys)
        assert len(w) > 0
        assert np.all(np.isfinite(mag))
        assert np.all(np.isfinite(phase))

    def test_low_frequency_gain_finite(self):
        w, mag, phase = signal.bode(tfhh.tf_sys, w=np.array([0.01]))
        assert np.isfinite(mag[0])
