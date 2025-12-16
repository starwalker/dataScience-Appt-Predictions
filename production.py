if __name__ == '__main__':
    from TextMiningMachine.io import get_data, write_data
    from TextMiningMachine.feature_contribution import contrib_context
    from scipy import sparse
    import pandas as pd
    import pickle
    import xgboost as xgb
    import numpy as np
    import gc
    num_top_features = 10

    #perform on just childrens and womens dataset

    #system_path = 'C:\\Users\\bidev221.CLINLAN\\PycharmProjects\\Appointment_Predictions\\'
    system_path = 'C:/PyProjects/Appointment_Predictions/'
    dsn = 'statmods'
    source_sql = '[dbo].[AppointmentsPipelinePatientsToPredict]'
    transform_path = system_path+'models/text_cat_transformer.p'
    key = 'PatEncCSNID'
    write_table = 'Appointment_ChildrenWomen_'
    date_col = 'AppointmentDateTime'
    # set target col
    target_col = 'NonCompletedAppointments'
    model_path = system_path + 'models/appointments_model.p'
    cutoff = .5

    # Query Data (takes about 2 hrs )
    data = get_data(dsn, source_sql)
    #data = pd.read_pickle('data/raw_data_preds.p')
    # create feature flags
    data['CompletedAppointmentPercentageLast12'] = [
    data.loc[i, 'CompletedAppointmentsLast12'] / data.loc[i, 'AppointmentsLast12'] for i in range(data.shape[0])]
    data['PatientCancelPercentageLast12'] = [
        data.loc[i, 'PatientCancelledAppointmentsLast12'] / data.loc[i, 'AppointmentsLast12'] for i in
        range(data.shape[0])]
    data['ProviderCancelPercentageLast12'] = [
        data.loc[i, 'ProviderCancelledAppointmentsLast12'] / data.loc[i, 'AppointmentsLast12'] for i in
        range(data.shape[0])]
    data['FromSCFlag'] = [1 if data.loc[i, 'PatientStateCode'] == 'SC' else 0 for i in range(data.shape[0])]
    data['AppointmentMonth'] = [data.loc[i, 'AppointmentDate'].month for i in range(data.shape[0])]
    data['AppointmentDayOfMonth'] = [data.loc[i, 'AppointmentDate'].day for i in range(data.shape[0])]
    data['ACOPatientPopulationFlag'] = [1 if data.loc[i, 'ACOPatientPopulation'] == 'ACO Medicare Current Roster' else 0
                                        for i in range(data.shape[0])]

    # load transform
    with open(transform_path, 'rb') as f:
        trans = pickle.load(f)

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    #subset the dataframe to be only the columns that need to be predicted on and transform it
    features = trans.transform(data)
    #extract the 100 features(in order) that were used to generate the model and subset
    feature_inds = []
    for mod_feat in model.feature_names:
        feature_inds.extend(np.where([1 if trans.feature_names_clean[i]==mod_feat else 0 for i in range(len(trans.feature_names_clean))])[0])
    features = sparse.csc_matrix(features)
    features = features[:,feature_inds]


    preds = model.predict(xgb.DMatrix(features, feature_names=model.feature_names))
    shaps = model.predict(xgb.DMatrix(features, feature_names=model.feature_names), pred_contribs=True)
    shap_list = contrib_context(shaps, feature_names=model.feature_names, num_top_features=num_top_features, features=features)
    preds_binary = np.array([1 if preds[i]>cutoff else 0 for i in range(len(preds))])

    preds_output = data[[key, date_col]].values
    preds_output = np.concatenate((preds_output, preds[:, None], preds_binary[:, None]), axis=1)
    preds_output = pd.DataFrame(preds_output)
    preds_output.columns = [key, date_col,'NoShowProbability','NoShowBinary']
    preds_output.columns = [preds_output.columns[i].replace('.', '_') for i in range(len(preds_output.columns))]
    # Write the predictions to the output table
    write_data('statmods', write_table+'Predictions', preds_output, drop=True,append=False)

    shap_output = data[[key, date_col]].values
    feature_vals = pd.DataFrame(features.todense())
    shap_output = np.concatenate((shap_output,np.array(shap_list), feature_vals, shaps), axis=1)
    shap_output = pd.DataFrame(shap_output)
    shap_output.columns = [key, date_col] + \
                          ['TopFeature' + str(i + 1) for i in range(num_top_features)] +\
                          [model.feature_names[i]+'_value' for i in range(len(model.feature_names))] + \
                          [model.feature_names[i] + '_shap' for i in range(len(model.feature_names))]+['bias']
    shap_output.columns = [shap_output.columns[i].replace('.', '_') for i in range(len(shap_output.columns))]

    # Write the predictions to the output table
    write_data('statmods', write_table+'Shaps', shap_output, drop=True,append=False)
    gc.collect()



    #perform on all appointments
    source_sql = '[dbo].[AllAppointmentsPipelinePatientsToPredict]'
    transform_path = system_path+'models/text_cat_transformer_all.p'
    write_table = 'Appointment_All_'
    # set target col
    target_col = 'NonCompletedAppointments'
    model_path = system_path + 'models/appointments_model_all.p'
    cutoff = .5

    # Query Data (takes about 2 hrs )
    data = get_data(dsn, source_sql)
    # data = pd.read_pickle('data/raw_data_all_preds.p')
    # create feature flags
    data['CompletedAppointmentPercentageLast12'] = [
    data.loc[i, 'CompletedAppointmentsLast12'] / data.loc[i, 'AppointmentsLast12'] for i in range(data.shape[0])]
    data['PatientCancelPercentageLast12'] = [
        data.loc[i, 'PatientCancelledAppointmentsLast12'] / data.loc[i, 'AppointmentsLast12'] for i in
        range(data.shape[0])]
    data['ProviderCancelPercentageLast12'] = [
        data.loc[i, 'ProviderCancelledAppointmentsLast12'] / data.loc[i, 'AppointmentsLast12'] for i in
        range(data.shape[0])]
    data['FromSCFlag'] = [1 if data.loc[i, 'PatientStateCode'] == 'SC' else 0 for i in range(data.shape[0])]
    data['AppointmentMonth'] = [data.loc[i, 'AppointmentDate'].month for i in range(data.shape[0])]
    data['AppointmentDayOfMonth'] = [data.loc[i, 'AppointmentDate'].day for i in range(data.shape[0])]
    data['ACOPatientPopulationFlag'] = [1 if data.loc[i, 'ACOPatientPopulation'] == 'ACO Medicare Current Roster' else 0
                                        for i in range(data.shape[0])]

    # load transform
    with open(transform_path, 'rb') as f:
        trans = pickle.load(f)

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    #subset the dataframe to be only the columns that need to be predicted on and transform it
    features = trans.transform(data)
    #extract the 100 features(in order) that were used to generate the model and subset
    feature_inds = []
    for mod_feat in model.feature_names:
        feature_inds.extend(np.where([1 if trans.feature_names_clean[i]==mod_feat else 0 for i in range(len(trans.feature_names_clean))])[0])
    features = sparse.csc_matrix(features)
    features = features[:,feature_inds]


    preds = model.predict(xgb.DMatrix(features, feature_names=model.feature_names))
    shaps = model.predict(xgb.DMatrix(features, feature_names=model.feature_names), pred_contribs=True)
    shap_list = contrib_context(shaps, feature_names=model.feature_names, num_top_features=num_top_features, features=features)
    preds_binary = np.array([1 if preds[i]>cutoff else 0 for i in range(len(preds))])

    preds_output = data[[key, date_col]].values
    preds_output = np.concatenate((preds_output, preds[:, None], preds_binary[:, None]), axis=1)
    preds_output = pd.DataFrame(preds_output)
    preds_output.columns = [key, date_col,'NoShowProbability','NoShowBinary']
    preds_output.columns = [preds_output.columns[i].replace('.', '_') for i in range(len(preds_output.columns))]
    # Write the predictions to the output table
    write_data('statmods', write_table+'Predictions', preds_output, drop=True,append=False)

    shap_output = data[[key, date_col]].values
    feature_vals = pd.DataFrame(features.todense())
    shap_output = np.concatenate((shap_output,np.array(shap_list), feature_vals, shaps), axis=1)
    shap_output = pd.DataFrame(shap_output)
    shap_output.columns = [key, date_col] + \
                          ['TopFeature' + str(i + 1) for i in range(num_top_features)] + \
                          [model.feature_names[i]+'_value' for i in range(len(model.feature_names))] + \
                          [model.feature_names[i] + '_shap' for i in range(len(model.feature_names))]+['bias']
    shap_output.columns = [shap_output.columns[i].replace('.', '_') for i in range(len(shap_output.columns))]

    # Write the predictions to the output table
    write_data('statmods', write_table+'Shaps', shap_output, drop=True,append=False)
    gc.collect()


