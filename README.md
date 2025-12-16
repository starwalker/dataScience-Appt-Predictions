## Appoitment Based No Show Models

#### Objectives

Predict probablity of a patient completing their appointment. The training data for the model was developed from the MUSC inpatient data pipeline. Details are in the repository readme. 
A combination of semi structured and numeric data from updated each night including diagnosis, problem lists, and therapeutic classes was used to create dependent variables.

## Data Source

* sql='[dbo].[PatientLevelAppointmentsPipelinePatientsToPredict]'
* server =[umapbi].[StatisticalModels]

same day cancelation is considered a now show

## Capabilities


## Deployment Notes


## Appointment Level Predictions 
##### Source Data to Establish what to Predict
 Stored Proc on StatModels:    [dbo].[AllAppointmentsPipelinePatientsToPredict]

##### Schedueled every day at 5pm, predictive model pass through all data from source
(is pass through with no filtering)
Schedueled Proc writes to Output Table : [StatisticalModel].NoShow_AppointmentLevel_All_Predictions

#### Schedueled Refresh of tableu with table about 915pm
+ Aggregated in up to DEP level
+ Source Table: [StatisticalModel].NoShow_AppointmentLevel_All_Predictions



## Patient Level Predictions 
##### Source Data to Establish what to Predict
 Stored Proc on StatModels:   [dbo].[PatientLevelAppointmentsPipelinePatientsToPredict]

##### Schedueled every day at 5pm, predictive model pass through all data from source
(is pass through with no filtering)
Schedueled Proc writes to Output Table : [StatisticalModel].NoShow_PatientLevel_All_Predictions



