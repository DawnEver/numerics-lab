"""Complex edge phasors preserve eddy mass and its conductor-aware gauge."""

import numpy as np
import pytest

from motronics.contracts.ir.equation_ir import ConstitutiveRef, CurlCurlTerm, EquationIR
from motronics.hamilton.discretizations.hcurl import box_tet_mesh, compile_hcurl_semidiscrete
from motronics.hamilton.discretizations.hcurl.harmonic import solve_hcurl_harmonic


def problem(sigma):
    mesh = box_tet_mesh(1, 1, 1)
    equation = EquationIR(terms=(CurlCurlTerm(constitutive=ConstitutiveRef(name='vacuum')),))
    return compile_hcurl_semidiscrete(
        equation, mesh, reluctivity=1.0, conductivity=sigma, current_density=np.array([0.0, 0.0, 1.0])
    )


def test_conducting_edge_phasor_matches_full_complex_system():
    compiled = problem(2.0)
    actual = solve_hcurl_harmonic(compiled, 3.0)
    operator = compiled._k_csr() + 3j * compiled._m_csr()
    expected = np.linalg.solve(operator.toarray(), compiled.load)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    assert np.linalg.norm(actual.imag) > 0
    assert np.real(np.vdot(actual, compiled._m_csr() @ actual)) > 0


def test_nonconducting_harmonic_uses_the_static_gradient_gauge():
    compiled = problem(0.0)
    static = solve_hcurl_harmonic(compiled, 0.0)
    harmonic = solve_hcurl_harmonic(compiled, 4.0)
    np.testing.assert_allclose(harmonic, static)
    assert np.all(harmonic[compiled.gauge_dofs] == 0)


def test_invalid_frequency_and_ungauged_dc_conductor_are_refused():
    compiled = problem(2.0)
    with pytest.raises(ValueError, match='DC'):
        solve_hcurl_harmonic(compiled, 0.0)
    with pytest.raises(ValueError, match='frequency'):
        solve_hcurl_harmonic(compiled, -1.0)
