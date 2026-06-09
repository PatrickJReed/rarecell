def test_fixture_shape(synthetic_adata):
    a = synthetic_adata
    assert a.n_obs == 5000
    rare_frac = (a.obs["true_cluster"] == "3").mean()
    assert 0.04 < rare_frac < 0.06
