from statsmodels.tsa.statespace.sarimax import SARIMAX

def train_model(train, target, exog):
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 7)
    model = SARIMAX(
        endog=train[target],
        exog=train[exog],
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    return model.fit(disp=False)

def predict(model, exog, start_date, end_date, dynamic=False):
    return model.get_prediction(start=start_date, end=end_date, exog=exog, dynamic=dynamic)
