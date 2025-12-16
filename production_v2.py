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

    #project_path = 'C:\\Users\\bidev221.CLINLAN\\PycharmProjects\\Appointment_Predictions\\'
    project_path = 'C:/PyProjects/Appointment_Predictions/'
    #project_path = 'C:/Python/PyProjects/Appointment_Predictions/'
    dsn = 'statmods'
    key = 'PatEncCSNID'
    date_col = 'AppointmentDateTime'
    # set target col
    target_col = 'NonCompletedAppointments'






    ###**** AppointmentLevel All predictions

    source_sql = '[dbo].[AppointmentsPipelinePatientsToPredict]'
    model_name = 'NoShow_AppointmentLevel_CW'
    bundle_path = project_path + 'models/'+model_name+'_Bundle.p'
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
    with open(bundle_path, 'rb') as f:
        mb = pickle.load(f)
    mb.valid_prediction_days = 1
    output = mb.generate_predictions_dataframe(data,table_keys = key,include_cols = date_col)
    output = output.sort_values(by=date_col)
    write_data('statmods', model_name+'_Predictions', output,drop=True,append=False)
    del data
    del output
    del mb
    gc.collect()




###**** AppointmentLevel All predictions
    source_sql = '[dbo].[AllAppointmentsPipelinePatientsToPredict]'
    model_name = 'NoShow_AppointmentLevel_All'
    bundle_path = project_path + 'models/'+model_name+'_Bundle.p'

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
    with open(bundle_path, 'rb') as f:
        mb = pickle.load(f)
    mb.valid_prediction_days = 1
    output = mb.generate_predictions_dataframe(data,table_keys = key,include_cols = date_col)
    output = output.sort_values(by=date_col)
    write_data('statmods', model_name+'_Predictions', output,drop=True,append=False)
    del data
    del output
    del mb
    gc.collect()




    key = 'PatientMRN'
    model_name = 'NoShow_PatientLevel_All'
    bundle_path = project_path + 'models/'+model_name+'_Bundle.p'
    source_sql = '[dbo].[PatientLevelAppointmentsPipelinePatientsToPredict]'


    data = get_data(dsn, source_sql)
    #data = pd.read_pickle('data/raw_data_all_patientlevel_preds.p')
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
    data['ACOPatientPopulationFlag'] = [1 if data.loc[i, 'ACOPatientPopulation'] == 'ACO Medicare Current Roster' else 0
                                        for i in range(data.shape[0])]

    #data = pd.read_pickle('data/raw_data_preds.p')
    # load transform
    with open(bundle_path, 'rb') as f:
        mb = pickle.load(f)
    mb.valid_prediction_days = 1
    output = mb.generate_predictions_dataframe(data,table_keys = key,include_cols = date_col)
    output = output.sort_values(by=date_col)
    write_data('statmods', model_name+'_Predictions', output,drop=True,append=False)
    gc.collect()
