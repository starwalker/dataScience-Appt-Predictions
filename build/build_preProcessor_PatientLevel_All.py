from TextMiningMachine.io import get_data
from TextMiningMachine.feature_extraction import DataSetBuilder
import pandas as pd
import pickle
from nltk.corpus import stopwords
import gc
sql = '[dbo].[PatientLevelAppointmentPipeline]'
data = get_data('statmods', sql)


#data.to_csv('data/raw_data.csv')
################################

#create feature flags
data['CompletedAppointmentPercentageLast12'] = [data.loc[i,'CompletedAppointmentsLast12']/data.loc[i,'AppointmentsLast12']for i in range(data.shape[0])]
data['PatientCancelPercentageLast12'] = data['PatientCancelledAppointmentsLast12']/data['AppointmentsLast12']
data['ProviderCancelPercentageLast12'] = data['ProviderCancelledAppointmentsLast12']/data['AppointmentsLast12']
data['FromSCFlag'] = [1 if data.loc[i,'PatientStateCode']=='SC' else 0 for i in range(data.shape[0])]
data['ACOPatientPopulationFlag'] = [1 if data.loc[i,'ACOPatientPopulation']=='ACO Medicare Current Roster' else 0 for i in range(data.shape[0])]
# save the raw_data
file = 'data/raw_data_all.p'
pickle.dump(data, open(file, 'wb'))

# load the pickled data
data = pd.read_pickle('data/raw_data_all.p')

# set up the parameters
col_dict = {'cat_cols': ['PatientMarketingRegion','PatientSex','PatientMaritalStatus','PatientMyChartStatus'],
            'zero_imputer_cols':['PatientAge','CompletedAppointmentPercentageLast12',
                                'PatientCancelPercentageLast12','ProviderCancelPercentageLast12','FromSCFlag',
                                 'ACOPatientPopulationFlag',
                                'AppointmentsLast12',
                                'CompletedAppointmentsLast12',
                                'NoShowAppointmentsLast12',
                                'PatientCancelledWithin24HoursAppointmentsLast12',
                                'CompletedEstablishedPatientAppointmentsLast12',
                                'CompletedNewPatientAppointmentsLast12',
                                'CompletedNurseOrAncillaryAppointmentsLast12',
                                'CompletedOtherAppointmentsLast12',
                                'CompletedProcedureAppointmentsLast12',
                                'AllCancelledAppointmentsLast12',
                                'AllCancelledWithin24HoursAppointmentsLast12',
                                'PatientCancelledAppointmentsLast12',
                                'ProviderCancelledAppointmentsLast12',
                                'ProviderCancelledWithin30DaysAppointmentsLast12',
                                'OtherCancelledAppointmentsLast12',
                                'ScheduledMinutesLast12',
                                'CompletedMinutesLast12',
                                'CheckInToApptLast12',
                                'CheckInToRoomLast12',
                                'ApptToRoomLast12',
                                'RoomToNurseLeaveLast12',
                                'RoomToProvEnterLast12',
                                'NurseLeaveToProvEnterLast12',
                                'ProvEnterToVisitEndLast12',
                                'RoomToVisitEndLast12',
                                'ApptToVisitEndLast12',
                                'VisitEndToCheckOutLast12',
                                'CheckInToCheckOutLast12',
                                'AvgScheduledMinutesLast12',
                                'AvgCompletedMinutesLast12',
                                'AvgCheckInToApptLast12',
                                'AvgCheckInToRoomLast12',
                                'AvgApptToRoomLast12',
                                'AvgRoomToNurseLeaveLast12',
                                'AvgRoomToProvEnterLast12',
                                'AvgNurseLeaveToProvEnterLast12',
                                'AvgProvEnterToVisitEndLast12',
                                'AvgRoomToVisitEndLast12',
                                'AvgApptToVisitEndLast12',
                                'AvgVisitEndToCheckOutLast12',
                                'AvgCheckInToCheckOutLast12',
                                'ArrivalTimelinessLast12',
                                'AverageLagDaysLast12',
                                'CopayDueLast12',
                                'CopayCollectedLast12',
                                'DistinctProvidersLast12',
                                'DistinctProviderSpecialtiesLast12',
                                'DistinctDepartmentsLast12',
                                'DistinctDepartmentSpecialtiesLast12',
                                'InpatientDischargesLast12',
                                'AverageLOSLast12',
                                'AverageCMILast12',
                                'InpatientChargesLast12',
                                'InpatientPaymentsLast12',
                                'InpatientAdjustmentsLast12',
                                'EDVisitsLast12',
                                'TotalMinInEDLast12',
                                'AvgMinInEDLast12',
                                'EDObservationVisitsLast12',
                                'EDLeftWithoutSeenVisitsLast12',
                                'EDBehavioralHealthVisitsLast12',
                                'EDInpatientAdmissionsLast12',
                                'EDReadmitBounceBackIn72HoursLast12',
                                'PBDepartmentsLast12',
                                'PBDepartmentSpecialtiesLast12',
                                'PBClinicalDepartmentsLast12',
                                'PBPerformingProvidersLast12',
                                'PBPerformingProviderSpecialtiesLast12',
                                'PBServiceCodesLast12',
                                'ASAUnitsTotalLast12',
                                'PBServiceCountLast12',
                                'PBServiceUnitsLast12',
                                'PBChargeAmountLast12',
                                'PBWorkRVUsLast12',
                                'PBAdjustmentsLast12',
                                'PBBadDebtWriteOffLast12',
                                'PBChartiyWriteOffLast12',
                                'PBContractualWriteOffLast12',
                                'PBDiscountsLast12',
                                'PBPaymentsLast12']}

gc.collect()

# Learn the preProcessing
stops = stopwords.words('english')+['stage', 'unspecified', 'hold', 'call', 'defined', 'secondary', 'welcome',
                                    'mention', 'hospital','delivery']
trans = DataSetBuilder(col_dict=col_dict)
trans.params['cat_cols']['min_freq'] = .003
trans.update_params({'cat_cols': {'min_freq': .001}})
trans.fit(data)

# save the transform
file = 'models/text_cat_transformer_patientlevel_all.p'
pickle.dump(trans, open(file, 'wb'))




# save the training data
#features = xgb.DMatrix(trans.transform(data), feature_names=trans.feature_names)
#xgb.DMatrix.save_binary(features, 'data/xgb.features.data')




