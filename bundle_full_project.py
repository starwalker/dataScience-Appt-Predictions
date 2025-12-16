if __name__ == '__main__':
    import pickle
    import gc
    import pandas as pd
    import datetime
    import sklearn
    gc.collect()
    from TextMiningMachine.bundle import XGBModelBundle



    # appointments - level all model

    # set target col
    target_col = 'NonCompletedAppointments'

    # load transform
    with open('models/text_cat_transformer_all.p', 'rb') as f:
        trans = pickle.load(f)
    # load transformed data
    with open('data/features.p', 'rb') as f:
        features = pickle.load(f)

    model_name = 'NoShow_AppointmentLevel_All'
    with open('models/'+model_name+'_Model.p', 'rb') as f:
        model = pickle.load(f)

    trans.feature_names = trans.feature_names_clean
    mb = XGBModelBundle(model_name)
    mb.update({'trans':trans,
               'model':model,
               'target_col':'NonCompletedAppointments'})
    data_transformations = "data[target_col] = [1 if data.loc[i, 'CompletedAppointments'] == 0 else 0 for i in range(data.shape[0])]"
    mb.data_transformations = data_transformations
    preds = mb.generate_predictions(features)
    mb.fit_deciles(list(preds))
    model_file_name = 'models/'+model_name+'_Bundle.p'
    mb.save(model_file_name)




    ##patient - level all model

    # set target col
    target_col = 'NonCompletedAppointments'

    # load transform
    with open('models/text_cat_transformer_patientlevel_all.p', 'rb') as f:
        trans = pickle.load(f)
    # load transformed data
    with open('data/features_patientlevel.p', 'rb') as f:
        features = pickle.load(f)

    model_name = 'NoShow_PatientLevel_All'
    with open('models/'+model_name+'_Model.p', 'rb') as f:
        model = pickle.load(f)

    trans.feature_names = trans.feature_names_clean
    mb = XGBModelBundle(model_name)
    mb.update({'trans':trans,
               'model':model,
               'target_col':'NonCompletedAppointments'})
    data_transformations = "data[target_col] = [1 if data.loc[i, 'CompletedAppointments'] == 0 else 0 for i in range(data.shape[0])]"
    mb.data_transformations = data_transformations
    preds = mb.generate_predictions(features)
    mb.fit_deciles(list(preds))
    model_file_name = 'models/'+model_name+'_Bundle.p'
    mb.save(model_file_name)



    ##appointment - level CW model

    # set target col
    target_col = 'NonCompletedAppointments'

    data = pd.read_pickle('data/raw_data.p')
    # load transform
    with open('models/text_cat_transformer_CW.p', 'rb') as f:
        trans = pickle.load(f)
    # load transformed data
    features = trans.transform(data)

    model_name = 'NoShow_AppointmentLevel_CW'
    with open('models/'+model_name+'_Model.p', 'rb') as f:
        model = pickle.load(f)

    trans.feature_names = trans.feature_names_clean
    mb = XGBModelBundle(model_name)
    mb.update({'trans':trans,
               'model':model,
               'target_col':'NonCompletedAppointments'})
    data_transformations = "data[target_col] = [1 if data.loc[i, 'CompletedAppointments'] == 0 else 0 for i in range(data.shape[0])]"
    mb.data_transformations = data_transformations
    preds = mb.generate_predictions(features)
    mb.fit_deciles(list(preds))
    model_file_name = 'models/'+model_name+'_Bundle.p'
    mb.save(model_file_name)

