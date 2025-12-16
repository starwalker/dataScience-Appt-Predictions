from TextMiningMachine.io import get_data
import pickle
import gc


# sql='[dbo].[PatientLevelAppointmentPipeline] '
# data = get_data('statmods', sql)
# # save the raw_data
# file = 'data/raw_data_all_patients.p'
# pickle.dump(data, open(file, 'wb'))
# del data
# gc.collect()


sql='[dbo].[PatientLevelAppointmentsPipelinePatientsToPredict]'
data = get_data('statmods', sql)
# save the raw_data
file = 'data/raw_data_all_patientlevel_preds.p'
pickle.dump(data, open(file, 'wb'))
del data
gc.collect()




#
# sql='[dbo].[AppointmentsPipelinePatientsToPredict]'
# data = get_data('statmods', sql)
#
# # save the raw_data
# file = 'data/raw_data_preds.p'
# pickle.dump(data, open(file, 'wb'))
# del data
# gc.collect()
#
#
# sql='[dbo].[AllAppointmentsPipelinePatientsToPredict]'
# data = get_data('statmods', sql)
#
# # save the raw_data
# file = 'data/raw_data_all_preds.p'
# pickle.dump(data, open(file, 'wb'))
# del data
# gc.collect()
#

# # Query Data (takes about 2 hrs )
# sql = '[dbo].[AllAppointmentsPipeline]'
# data = get_data('statmods', sql)
#
# # save the raw_data
# file = 'data/raw_data_all.p'
# pickle.dump(data, open(file, 'wb'))
# del data
# gc.collect()
#
#
# # Query Data (takes about 2 hrs )
# sql = '[dbo].[AppointmentsPipeline]'
# data = get_data('statmods', sql)
#
# # save the raw_data
# file = 'data/raw_data.p'
# pickle.dump(data, open(file, 'wb'))
# del data
# gc.collect()