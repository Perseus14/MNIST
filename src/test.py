def test(model, ds_test):
	return model.evaluate(ds_test, verbose=0)

