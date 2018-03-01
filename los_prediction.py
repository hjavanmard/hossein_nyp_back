
'''
general guidelines:
 - the general approach to create featuers in this DL is so that If there is a knowledge of some specific names like
 procedure name that is given by CS team we will create features based on those names, othewise we have a general
 feature created for that column. ( in some scenarios we might have both or neigther depending on how much time
 takes to build that feature)
'''
# for django and product purpose
import logging
from logUtils import Logger
from constants import app, event
from settings import AMD_client

from decision.actions.actionOption import ActionOption
from decision.actions.specializedAction import SpecializedAction
from main.decorators import method_cache
from django.db import connection, connections

# required libraries
import pandas as pd
import numpy as np
import time
import pytz
from datetime import datetime, timedelta
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import RandomizedSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline, Pipeline
from settings import AMD_client, USE_KAFKA_LOGGING

# functions from other modules
#import decision.actions.specialized.utilities.ipUtilities as ipUtils
import ut_check as ipUtils
from decision.models import IPLOSOutbound
# import main.analysis.singleMetric as singleMetric

#logger = Logger(logging.getLogger("application"), app=app.DECISION_ENGINE)
logger = logging.getLogger("dl_monitoring")

# to extract all relevant cases in a time range who stayed in the hospital.
def getHistoricalPatients(ddParams, hospital, start, end):
  # extra filter is an optional conditions to filtre out some cases if any
  encounterFilter = '(1=1)' if ddParams['encounterFilter'] is None else ddParams['encounterFilter']
  if 'connection' in ddParams.keys() and ddParams['connection'] == 'redshift':
    sqlConnection = connections['redshift']
    los_select = "DATEDIFF(DAY, admit_time, discharge_time) as los"
  else:
    los_select = "TIMESTAMPDIFF(DAY, admit_time, discharge_time) as los"
    sqlConnection = connection
  queryParams = {
    'hospital': hospital.id,
    'startDatetime': str(start),
    'endDatetime': str(end),
    'startSql': ddParams['startSql'],
    'endSql': ddParams['endSql'],
    'encounterFilter': encounterFilter,
    'los_select': los_select
  }

  patientQuery = '''
    SELECT
      instance1,
      instance2,
      instance3,
      {startSql} as admit_time,
      {endSql} as discharge_time,
      {los_select},
      UPPER(admitting_class) as admitting_class,
      UPPER(admitting_service) as admitting_service,
      UPPER(admitting_diagnosis) as admitting_diagnosis,
      UPPER(admit_source) as admit_source,
      age,
      gender,
      UPPER(language) as language,
      interpreter,
      UPPER(primary_race) as primary_race,
      UPPER(home_country) as home_country,
      UPPER(payor) as payor,
      discharge_disposition,
      admitting_dx_code
    FROM
      encounter_master
    WHERE
      hospital = {hospital} and
      {startSql} is not null and
      {endSql} is not null  and
      {endSql} >= '{startDatetime}' and
      {endSql} <= '{endDatetime}' and
      {startSql} < {endSql} and
      {encounterFilter}
    order by
      {startSql};
  '''.format(**queryParams)
  data = pd.read_sql_query(patientQuery, sqlConnection, coerce_float=True)
  data['admit_time'] = pd.to_datetime(data['admit_time'])
  data['discharge_time'] = pd.to_datetime(data['discharge_time'])
  return data


# --------------------------------------------------------------------------------------------------
# returns all patients who are in the unit in a time range and their exit time is null or after current time
#  currently there is a virtual bed starting with 9999 for patient ready to transfer in/out of unit ( we exclude them)
def getCurrentPatients(ddParams, hospital, curDate):
  encounterFilter = '(1=1)' if ddParams['encounterFilter'] is None else ddParams['encounterFilter']
  if 'connection' in ddParams.keys() and ddParams['connection'] == 'redshift':
    sqlConnection = connections['redshift']
  else:
    sqlConnection = connection
  queryParams = {
    'hospital': hospital.id,
    'startSql': ddParams['startSql'],
    'endSql': ddParams['endSql'],
    'curDate': curDate,
    'encounterFilter': encounterFilter
  }

  patientQuery = '''
    SELECT
      instance1,
      instance2,
      instance3,
      {startSql} as admit_time,
      {endSql} as discharge_time,
      UPPER(admitting_class) as admitting_class,
      UPPER(admitting_service) as admitting_service,
      UPPER(admitting_diagnosis) as admitting_diagnosis,
      UPPER(admit_source) as admit_source,
      age,
      gender,
      UPPER(language) as language,
      interpreter,
      UPPER(primary_race) as primary_race,
      UPPER(home_country) as home_country,
      UPPER(payor) as payor,
      discharge_disposition,
      admitting_dx_code
    FROM
      encounter_master
    WHERE
      hospital = {hospital} and
      {startSql} is not null and
      ({endSql} is null or {endSql} > '{curDate}') and
      {startSql} <= '{curDate}' and
      {encounterFilter}
    order by
      {startSql};
'''.format(**queryParams)
  data = pd.read_sql_query(patientQuery, sqlConnection, coerce_float=True)
  data['admit_time'] = pd.to_datetime(data['admit_time'])
  data['discharge_time'] = pd.to_datetime(data['discharge_time'])
  return data


def getTotalNumCases(ddParams, hospital, featureStr, end):
  # extra filter is an optional conditions to filtre out some cases if any
  extraFilter = '(1=1)' if ddParams['extraFilter'] is None else ddParams['extraFilter']
  if 'connection' in ddParams.keys() and ddParams['connection'] == 'redshift':
    sqlConnection = connections['redshift']
  else:
    sqlConnection = connection
  queryParams = {
    'hospital': hospital.id,
    'startDatetime': str(end - timedelta(days=2 * ddParams['historicalDays'])),
    'endDatetime': str(end),
    'startSql': ddParams['startSql'],
    'endSql': ddParams['endSql'],
    'featureStr': featureStr,
    'extraFilter': extraFilter
  }

  query = '''
    SELECT
      {featureStr}
    FROM
      encounter_master
    WHERE
      hospital = {hospital} and
      {endSql} >= '{startDatetime}' and
      {endSql} <= '{endDatetime}' and
      {extraFilter}
    ORDER BY
      admit_time;
  '''.format(**queryParams)
  data = pd.read_sql_query(query, sqlConnection, coerce_float=True)
  return data


def addScheduledSurgPatients(ddParams, hospital, data, start, end):
  if not ddParams['raw_or_schedule_table']:
    return data, []
  if start is None:
    start = data['TimeStamp'].min() - timedelta(days=1)
  if end is None:
    end = data['TimeStamp'].max() + timedelta(days=1)
  instances = "('%s')" % "','".join(set(data.instance2.values))
  scheduledSurgs = ipUtils.totalORCases(hospital, start, end, filterCondition="instance2 in %s" % instances)
  scheduledSurgs_indexed = scheduledSurgs.set_index(['instance2', 'SurgeryTime'])
  futureSurgs = [scheduledSurgs_indexed.loc(axis=0)[row['instance2'], row['TimStamp']:row['TimStamp'] + timedelta(days=7)]
                 for idx, row in data.iterrows()]
  data['hasFutureSurg'] = [0 if df.empty else 1 for df in futureSurgs]
  data['FutureSurg_time'] = [None if df.empty else df['SurgeryTime'].tolist()[0] for df in futureSurgs]
  data['FutureSurg_left'] = data.FutureSurg_time.apply(lambda x: (x['FutureSurg_time'] - x['TimStamp']).days
                                                       if x['FutureSurg_time'] is not None else 1000)
  data['FutureSurg_specialty'] = [None if df.empty else df['procedure_specialty'].tolist()[0] for df in futureSurgs]
  newFeatures = ['hasFutureSurg', 'FutureSurg_left', 'FutureSurg_specialty']
  print 'addScheduledSurgPatients'
  return data, newFeatures


def addSurgFeatures(ddParams, hospital, data, trainingEnd):
  if not ddParams['raw_or_thru_table']:
    return data, []
  if ('raw_or_thru_ProcedureNames' in ddParams) and (ddParams['raw_or_thru_ProcedureNames'] is not None):
    SurgProcedureNames = ddParams['raw_or_thru_ProcedureNames']
  else:
    SurgProcedureNames = []
  if ('raw_or_thru_ProcedureSpecialty' in ddParams) and (ddParams['raw_or_thru_ProcedureSpecialty'] is not None):
    SurgProcedureSpecialities = ddParams['raw_or_thru_ProcedureSpecialty']
  else:
    SurgProcedureSpecialities = []

  # check how long last proc took compare to scheduled time
  data['overrunProcedure'] = data['overrunProcedure'].apply(lambda x: 0.0 if np.isnan(x) else round(x, 1))
  data['procedureLength'] = data['actualDuration'].apply(lambda x: 0.0 if np.isnan(x) else x)
  data['daysFromSurg'] = data.apply(lambda x: (x['TimeStamp'] - x['admit_time']) / timedelta(days=1)
                                    if pd.isnull(x['SurgeryStop'])
                                    else (x['TimeStamp'] - x['SurgeryStop']) / timedelta(days=1), axis=1)
  # need another feature to distinguish between those had surgeries from those do not have
  data['hadSurg'] = data['SurgeryStop'].apply(lambda x: 0 if pd.isnull(x) else 1)
  surgFeatures = ['procedure_name_or', 'procedure_specialty', 'surgery_room', 'surgeon_name', 'asa_class',
                  'overrunProcedure', 'daysFromSurg', 'hadSurg', 'procedureLength']

  for p in SurgProcedureNames:
    col = 'SurgProcedureNames_' + p
    data[col] = 0
    surgFeatures += [col]
  for s in SurgProcedureSpecialities:
    col = 'SurgProcedureSpecialties_' + s
    data[col] = 0
    surgFeatures += [col]
  data['specificProcNames'] = 0
  data['specificSpecialties'] = 0
  surgFeatures += ['specificProcNames', 'specificSpecialties']
  #
  for idx, row in data.iterrows():
    procName, speciality = row['procedure_name_or'], row['procedure_specialty']
    if procName in SurgProcedureNames:
      col = 'SurgProcedureNames_' + procName
      data.loc[idx, col] = 1
      data.loc[idx, 'specificProcNames'] = 1
    if speciality in SurgProcedureSpecialities:
      col = 'SurgProcedureSpecialties_' + speciality
      data.loc[idx, col] = 1
      data.loc[idx, 'specificSpecialties'] = 1
  print 'addSurgFeatures'
  return data, surgFeatures


def addCategoricalAdmitFeat(ddParams, hospital, data, trainingEnd):
  # for each text columns (categorical) we create a binary feature if there is a given speical names
  # we have a one hot encoding for categorical features based on their most common 10 values
  if ('encounter_master_admitServices' in ddParams) and (ddParams['encounter_master_admitServices'] is not None):
    admitServices = ddParams['encounter_master_admitServices']
  else:
    admitServices = []
  if ('encounter_master_admitDiagnosises' in ddParams) and (ddParams['encounter_master_admitDiagnosises'] is not None):
    admitDiagnosises = ddParams['encounter_master_admitDiagnosises']
  else:
    admitDiagnosises = []
  if ('encounter_master_admitSources' in ddParams) and (ddParams['encounter_master_admitSources'] is not None):
    admitSources = ddParams['encounter_master_admitSources']
  else:
    admitSources = []

  data['gender'] = data['gender'].apply(lambda x: 1 if x == 'M' else 0)
  data['interpreter'] = data['interpreter'].apply(lambda x: 1 if x == 'Y' else 0)
  features = ['gender', 'interpreter', 'admitting_dx_code']
  features += ['language', 'primary_race', 'home_country', 'payor', 'admitting_service',
               'admitting_class', 'admitting_diagnosis', 'admit_source']

  for s in admitServices:
    col = 'admitServices_' + s
    data[col] = 0
    features += [col]
  for d in admitDiagnosises:
    col = 'admitDiagnosises_' + d
    data[col] = 0
    features += [col]
  for s in admitSources:
    col = 'admitSources_' + s
    data[col] = 0
    features += [col]
  data['specificServices'] = 0
  data['specificDiagnosis'] = 0
  data['specificSources'] = 0
  features += ['specificServices', 'specificDiagnosis', 'specificSources']
  #
  for idx, row in data.iterrows():
    service, diagnosis, source = row['admitting_service'], row['admitting_diagnosis'], row['admit_source']
    if service in admitServices:
      col = 'admitServices_' + service
      data.loc[idx, col] = 1
      data.loc[idx, 'specificServices'] = 1
    if diagnosis[1:-1] in admitDiagnosises:
      col = 'admitDiagnosises_' + diagnosis[1:-1]
      data.loc[idx, col] = 1
      data.loc[idx, 'specificDiagnosis'] = 1
    if source in admitSources:
      col = 'admitSources_' + source
      data.loc[idx, col] = 1
      data.loc[idx, 'specificSources'] = 1

  return data, features


def getUnitLevelInfo(ddParams, hospital, data, start=None, end=None):
  if start is None:
    start = data['admit_time'].min() - timedelta(days=1)
  if end is None:
    end = data['TimeStamp'].max() + timedelta(days=1)
  instances = "('%s')" % "','".join(set(data.instance2.values))
  #if 'connection' in ddParams.keys() and ddParams['connection'] == 'redshift':
  #  sqlConnection = connections['redshift']
  #  unitLOS_select = "MAX(NULLIF(DATEDIFF(DAY, enter_time, exit_time), 0)) as unitLOS"
  #else:
  #unitLOS_select = "MAX(NULLIF(TIMESTAMPDIFF(DAY, enter_time, exit_time), 0)) as unitLOS"
  sqlConnection = connection
  queryParams = {
    'hospital': hospital.id,
    'startDatetime': str(start),
    'endDatetime': str(end),
    'instances': instances
  }
  segemntQuery = '''
                  SELECT
                    instance2,
                    enter_time,
                    max(exit_time) as exit_time,
                    max(current_unit) as current_unit,
                    max(current_bed) as current_bed,
                    max(care_level) as care_level
                  FROM
                    raw_adt_segment
                  WHERE
                    hospital = {hospital} AND
                    enter_time >= '{startDatetime}' AND
                    enter_time <= '{endDatetime}' AND
                    instance2 IN {instances}
                  GROUP BY
                    instance2, enter_time
                  ORDER BY
                    instance2, enter_time;
                '''.format(**queryParams)
  SegmentData = pd.read_sql_query(segemntQuery, sqlConnection, index_col=['instance2', 'enter_time'], coerce_float=True)
  return SegmentData


def addUnitLevelFeatures(ddParams, hospital, data, start=None, end=None):
  '''
  build features for the last unit name, the previous one, and how long they stayed in the unit specially if it
  is a cardio unit.
  '''
  if not ddParams['raw_adt_segment_table']:
    return data, []
  newFeatures = []
  SegmentData = getUnitLevelInfo(ddParams, hospital, data, start, end)
  nUnits = []
  previousUnit = []
  currentUnit = []
  # bed is for sending notifications (not a feature)
  currentBed = []
  unitLOS = []
  careLevel = []
  ICUStay = []
  if ('ICU_unit_names' in ddParams) and (ddParams['ICU_unit_names'] is not None):
    ICU_unit_names = ddParams['ICU_unit_names']
  else:
    ICU_unit_names = []
  for idx, row in data.iterrows():
    patientInfo = SegmentData.loc(axis=0)[row['instance2']: row['instance2'], :row['TimeStamp']]
    patientInfo.reset_index(inplace=True, drop=False)
    nUnits.append(patientInfo.shape[0])
    previousUnits = patientInfo['current_unit'].tolist()
    currentUnit.append(previousUnits[-1] if len(previousUnits) > 0 else None)
    currentBed.append(patientInfo['current_bed'].tolist()[-1] if len(previousUnits) > 0 else '')
    unitLOS.append((row['TimeStamp'] - patientInfo['enter_time'].tolist()[-1]) /timedelta(days=1) if len(previousUnits) > 0 else 0.0)
    previousUnit.append(previousUnits[-2] if len(previousUnits) > 1 else None)
    careLevel.append(patientInfo['care_level'].tolist()[-1] if len(previousUnits) > 0 else None)
    ICUPreviousStay = patientInfo[patientInfo.current_unit.isin(ICU_unit_names)]
    ICUStay.append(0 if ICUPreviousStay.empty
                   else sum([(row['TimeStamp'] - patientInfo['enter_time'].tolist()[-1]) /timedelta(days=1)
                   for i, r in ICUPreviousStay.iterrows()]))
  data['nUnits'] = nUnits
  data['previousUnit'] = previousUnit
  data['currentUnit'] = currentUnit
  data['currentBed'] = currentBed
  data['unitLOS'] = [0 if pd.isnull(item) else item for item in unitLOS]
  print 'unitLOS----> ',data['unitLOS'].tolist()
  data['careLevel'] = careLevel
  data['ICUStay'] = [0 if pd.isnull(item) else item for item in ICUStay]
  newFeatures = ['nUnits', 'previousUnit', 'currentUnit', 'careLevel', 'unitLOS', 'ICUStay']
  print 'addUnitLevelFeatures'
  return data, newFeatures


def addMedicationFeatures(ddParams, hospital, data):
  '''
  build features for medications such as number of meds in the last 24 hours, total number, and how much change they had.
  '''
  if not ddParams['med_admin_table']:
    return data, []
  nMeds = []
  diffMeds = []
  # if we have knowledge which dose_route or med_name are important, we can create features for them.
  # Otherwise a general way is taken
  if ('med_admin_routes' in ddParams) and (ddParams['med_admin_routes'] is not None):
    routes = ddParams['med_admin_routes']
  else:
    routes = []
  if ('VasoactiveMedList' in ddParams) and (ddParams['VasoactiveMedList'] is not None):
    VasoactiveMedList = ddParams['VasoactiveMedList']
  else:
    VasoactiveMedList = []

  nRoutes = {}
  changeRoutes = {}
  AllRoutes = []
  nVasocatMeds = []
  diffVasocatMeds = []
  for idx, row in data.iterrows():
    curDate = row['TimeStamp']
    nMeds.append(row['medObj'].countMed(curDate))
    nVasocatMeds.append(row['medObj'].countMed(curDate, medList=VasoactiveMedList))
    diffMeds.append(row['medObj'].diffCountMed(curDate))
    diffVasocatMeds.append(row['medObj'].diffCountMed(curDate, medList=VasoactiveMedList))
    allRoutes24h = row['medObj'].getLast24HoursMedData(curDate)['dose_route'].tolist()
    AllRoutes.append(len(allRoutes24h) if len(allRoutes24h) > 0 else 0)
    for specificRoute in routes:
      numberOfRoute = row['medObj'].countSpecificDoseRoutes(curDate, doseRoute=[specificRoute])
      if specificRoute in nRoutes:
        nRoutes[specificRoute].append(numberOfRoute)
      else:
        nRoutes[specificRoute] = [numberOfRoute]
      numberOfRoute2 = row['medObj'].countSpecificDoseRoutes(curDate - timedelta(days=1), doseRoute=[specificRoute])
      changeOfRoute = numberOfRoute - numberOfRoute2
      if specificRoute in changeRoutes:
        changeRoutes[specificRoute].append(changeOfRoute)
      else:
        changeRoutes[specificRoute] = [changeOfRoute]

  data['nMeds'] = nMeds
  data['diffMeds'] = diffMeds
  data['nVasocatMeds'] = nVasocatMeds
  data['diffVasocatMeds'] = diffVasocatMeds
  data['AllRoutes'] = [0 if np.isnan(item) else item for item in AllRoutes]
  newFeatures = ['nMeds', 'diffMeds', 'nVasocatMeds', 'diffVasocatMeds', 'AllRoutes']

  for specificRoute in routes:
    colName1 = specificRoute + '_count'
    data[colName1] = nRoutes[specificRoute]
    colName2 = specificRoute + '_change'
    data[colName2] = changeRoutes[specificRoute]
    newFeatures += [colName1, colName2]

  return data, newFeatures


def getERdata(ddParams, hospital, data, eventFilter=None, start=None, end=None):
  eventFilter = '1=1'if eventFilter is None else eventFilter
  if start is None:
    start = data['TimeStamp'].min() - timedelta(days=1)
  if end is None:
    end = data['TimeStamp'].max() + timedelta(days=1)
  instances = "('%s')" % "','".join(set(data.instance2.values))
  if ddParams['connection'] == 'redshift':
    sqlConnection = connections['redshift']
  else:
    sqlConnection = connection
  queryParams = {
    'hospital': hospital.id,
    'startDatetime': str(start - timedelta(days=7)),
    'endDatetime': str(end),
    'instances': instances,
    'eventFilter': eventFilter
  }
  ERQuery = '''
              SELECT
                instance2,
                check_in_time,
                LOWER(chief_complaint) as chief_complaint,
                acuity
              FROM
                raw_er
              WHERE
                hospital = {hospital} AND
                check_in_time >= '{startDatetime}' AND
                check_in_time <= '{endDatetime}' AND
                instance2 IN {instances} AND
                {eventFilter};
  '''.format(**queryParams)
  ERData = pd.read_sql_query(ERQuery, sqlConnection, coerce_float=True)
  return ERData


def addERFeatures(ddParams, hospital, data, start=None, end=None):
  '''
  build features from emergency room info looking at chief_compliant, acuity, and if they come from ER or not!
  '''
  if not ddParams['raw_er_table']:
    return data, []
  ERData = getERdata(ddParams, hospital, data, eventFilter=None)
  data['cameFromER'] = [1 if row.instance2 in ERData.instance2 else 0 for idx, row in data.iterrows()]
  acuityList = []
  for idx, row in data.iterrows():
    temp = ERData.loc[ERData['instance2'] == row.instance2, 'acuity'].tolist()
    if len(temp) != 0:
      acuityList.append(temp[0] if ((temp[0] is not None) or (temp[0] != '')) else 0)
    else:
      acuityList.append(0)
  data['acuity'] = acuityList
  newFeatures = ['cameFromER', 'acuity']
  # keep those patinets who had some special chief_complaint's
  if ('chief_complaintList' in ddParams) and (ddParams['chief_complaintList'] is not None):
    chief_complaintList = ddParams['chief_complaintList']
  else:
    chief_complaintList = []

  eventFilter = '('
  for i, complaint in enumerate(chief_complaintList):
    if i != 0:
      eventFilter += ' OR '
    eventFilter += "(LOWER(chief_complaint) LIKE '{}{}{}' )".format('%', complaint, '%')
  eventFilter += ')' if len(chief_complaintList) != 0 else ' 1=1 )'
  ERDataFiltered = getERdata(ddParams, hospital, data, eventFilter=eventFilter)
  for complaint in chief_complaintList:
    data[complaint] = [1 if complaint in str(ERDataFiltered.loc[ERDataFiltered.instance2 == row.instance2,
                                             'chief_complaint']) else 0 for idx, row in data.iterrows()]
    newFeatures += [complaint]
  print 'addERFeatures'
  return data, newFeatures


def makeTupleOfList(lis):
  # a convenient function to change input list to tuple (mostly use for sql queries)
  if type(lis) == list:
    if len(lis) > 1:
      return tuple(lis)
    else:
      return tuple(lis[0], lis[0])
  else:
    return tuple([lis, lis])


def addProcOrderFeatures(ddParams, hospital, data, start=None, end=None):
  '''
  build features from proc_orders table using procedure_name and order_type column
  if there is a list of special names provided it will build a binary features for them
  if proc_order_procedures in ddParams:
    - 'null': then it does not use procedure_name column at all
    - a list of names: it uses those exact words to build features
  NOTES: in case we want to use top common names in historical data, be carefull about 'discharge' order name.
        maybe better look at each categories
        we might want to seatch similar words rather than exact words
        right now we do not use continious/disconnect info so only total stay feature is useful
  '''
  # for each order_type in the below list we look at the total number of them in the last 24 hours,
  # chnages, and in total stay
  if not ddParams['proc_orders_table']:
    return data, []
  newFeatures = []
  if ('proc_order_procedures' in ddParams) and (ddParams['proc_order_procedures'] is not None):
    procedures = makeTupleOfList(ddParams['proc_order_procedures'])
  else:
    procedures = []
  if ('proc_order_order_types' in ddParams) and (ddParams['proc_order_order_types'] is not None):
    orderList = makeTupleOfList(ddParams['proc_order_order_types'])
  else:
    orderList = []
  # getting all procedures for patients in the time range
  # >>>>>>>>>>>>>>>> part1: all names: look at the most common ones blindely (careful about discharge_order)<<<
  sp0=time.time()
  status_filter = ddParams['proc_orders_order_status_not']
  procOrders_filter = "((UPPER(order_status) not in {}) ".format(makeTupleOfList(status_filter)) + "AND (UPPER(procedure_name) NOT LIKE '%DISCHARGE%' )) "
  procData = ipUtils.getProcedures(data, ddParams, hospital, eventFilter=procOrders_filter)
  print 'procData ',procData.shape, procOrders_filter
  procData['roundedScheduledTime'] = procData['scheduled_time'].apply(lambda x: datetime(x.year, x.month, x.day, ddParams['triggerTime']))
  # capture the most recent one 
  procData_shrinked = procData.groupby(['instance2', 'roundedScheduledTime']).apply(lambda df: df.sort_values('scheduled_time').tail(1))
  if not procData.empty:
    procData_shrinked = procData_shrinked.set_index(['instance2', 'scheduled_time']).sort_index()
  sp1 = time.time()
  print 'all proc extracted ',procData.shape, procData_shrinked.shape, sp1-sp0
  if not procData.empty:
    totalStay_recent = [procData_shrinked.loc(axis=0)[row.instance2: row.instance2, row.admit_time: row.TimeStamp]
                        for idx, row in data.iterrows()]
  else:
    totalStay_recent = [pd.DataFrame() for idx, row in data.iterrows()]
  data['procedure_name_recent'] = [df.procedure_name.tolist()[-1] if not df.empty else None for df in totalStay_recent]
  data['order_type_recent'] = [df.order_type.tolist()[-1] if not df.empty else None for df in totalStay_recent]
  newFeatures += ['procedure_name_recent', 'order_type_recent']
  sp2 = time.time()
  print 'most recent', sp2 - sp0, data['procedure_name_recent'].value_counts()[0:5], data['order_type_recent'].value_counts()[0:5]

  sp3 = time.time()
  #
  # >>>>>>>> part2: specific names <<<<<<<<<<<<<
  # get relevant info from proc_orders table in the target time window
  procData = procData.set_index(['instance2', 'scheduled_time']).sort_index()
  eventFilter_exactNames = "((order_type in {}) or (procedure_name in {})) ".format(orderList, procedures)
  print eventFilter_exactNames
  #procData_exactNames = ipUtils.getProcedures(data, ddParams, hospital, eventFilter_exactNames)
  procData_exactNames = procData[(procData.order_type.isin(orderList)) & (procData.procedure_name.isin(procedures))]
  sp4 = time.time()
  print 'exact names', procData_exactNames.shape, sp4 - sp3
  # update list of procedures in case that name was not in data set (to save some time for later python feature coding)
  print len(procedures), len(orderList)
  procedures = [proc for proc in procedures if procData_exactNames.procedure_name.nunique() != 0]
  orderList = [proc for order in orderList if procData_exactNames.order_type.nunique() != 0]
  print len(procedures), len(orderList)
  #procData_exactNames.set_index(['instance2', 'scheduled_time'], inplace=True)
  #procData_exactNames.sort_index(inplace=True)
  totalStay = [procData_exactNames.loc(axis=0)[row.instance2: row.instance2, : row.TimeStamp] if not procData_exactNames.empty
               else procData_exactNames for idx, row in data.iterrows()]
  '''
  # for now we turn off looking at daily change for proc to speed up code! (to save running time)
  last24Hours = [procData_exactNames.loc(axis=0)[row.instance2: row.instance2,
                                                 row.TimeStamp - timedelta(days=1): row.TimeStamp]
                 for idx, row in data.iterrows()]
  last48Hours = [procData_exactNames.loc(axis=0)[row.instance2: row.instance2,
                                                 row.TimeStamp - timedelta(days=2): row.TimeStamp - timedelta(days=1)]
                 for idx, row in data.iterrows()]
  '''
  sp5 = time.time()
  print 'totalstay', sp5 - sp4

  for order in orderList:
    colName1 = order + '_totalStay'
    print colName1
    data[colName1] = [df[df.order_type == order].shape[0] for df in totalStay]
    newFeatures += [colName1]

  for proc in procedures:
    colName1 = proc + '_totalStay'
    print colName1
    data[colName1] = [df[df.procedure_name == proc].shape[0] for df in totalStay]

    newFeatures += [colName1]
  #
  # >>>>>>>>>>>>>> part3: similar names <<<<<<<<<<<<<<<<<<
  similarNameList = ddParams['proc_order_procedures_similarNames']
  similarNameListQuery = " OR "if len(similarNameList) != 0 else " "
  similarNameListQuery += " OR ".join(["(procedure_name like '%" + item + "%' )" for item in similarNameList])
  eventFilter_similarNames = "((procedure_name like 'PT%' ) or (procedure_name like 'OT%') or\
                             (procedure_name like 'SLP%') " + similarNameListQuery + ")"
  print eventFilter_similarNames
  #procData_similarNames = ipUtils.getProcedures(data, ddParams, hospital, eventFilter_similarNames)
  procData_similarNames = procData[procData.procedure_name.str.contains('|'.join(similarNameList))]
  sp6 = time.time()
  print procData_similarNames.shape, sp6 - sp5
  # it seems soem evaluations paly important roles in decision making process
  EvalNotes = ['PT', 'OT', 'SLP']
  #procData_similarNames.set_index(['instance2', 'scheduled_time'], inplace=True)
  #procData_similarNames.sort_index(inplace=True)
  totalStay = [procData_similarNames.loc(axis=0)[row.instance2: row.instance2, : row.TimeStamp]
               for idx, row in data.iterrows()]
  columnList = []
  for procNames in EvalNotes:
    colName1 = procNames + ' eval_totalStay'
    print colName1
    columnList += [colName1]
    data[colName1] = [df[df.procedure_name.str.startswith(procNames)].shape[0] for df in totalStay]
  newFeatures += columnList
  # we check if all three evaluations have been done for the patient
  data['PT_OT_SLP'] = data.apply(lambda x: len([1 for item in columnList if x[item] > 0]) == 3, axis=1)
  newFeatures += ['PT_OT_SLP']
  print 'PT_OT_SLP'
  #
  def findSimilarWords(df, word):
    temp = df[df.procedure_name.str.contains(word)]['procedure_name'].tolist()
    if len(temp) > 0:
      return temp[-1]
    else:
      return None
  #
  for word in similarNameList:
    colName = 'proc_orders' + word
    print colName
    data[colName] = [findSimilarWords(df, word) for df in totalStay]
    newFeatures += [colName]
  return data, newFeatures


def addPrevAdmitInfoFeatures(ddParams, hospital, data, end):
  '''
    builds features for patinets with previous admit information
    - a binary feature for re-admit or not
    - what was previous disposition
    - how long has been passed
  '''
  reAdmitFeatures = []
  allCases = getTotalNumCases(ddParams, hospital, 'instance1, admit_time, discharge_time, discharge_disposition', end)
  end5 = time.time()
  allCases = allCases.set_index(['instance1', 'admit_time'], drop=False)
  allCases.sort_index(inplace=True)
  end6 = time.time()
  print end6 - end5
  previousAdmitInfo = [allCases.loc(axis=0)[row.instance1:row.instance1, :row.admit_time - timedelta(days=1)]
                       for idx, row in data.iterrows()]
  data['previousAdmitDispos'] = ['' if prevAdmit.empty else prevAdmit['discharge_disposition'].tolist()[-1]
                                 for prevAdmit in previousAdmitInfo]
  data['previousAdmitTime'] = [None if prevAdmit.empty else prevAdmit['admit_time'].tolist()[-1]
                               for prevAdmit in previousAdmitInfo]
  data['re_admitted'] = [0 if prevAdmit.empty else 1 for prevAdmit in previousAdmitInfo]
  # as previous admit_time closer to current time we have more serious patients so we choose default very large
  data['timePassedFromPrevAdmit'] = [1000 if row['re_admitted'] == 0
                                     else (row['admit_time'] - row['previousAdmitTime']).days / 30
                                     for idx, row in data.iterrows()]
  # if patient has previous admit, lets use his previous disposition
  for dispos in ddParams['targetDispositions']:
    colName = 'prevDisposition_' + dispos[0:20]
    data[colName] = [1 if dispos in row['previousAdmitDispos'] else 0 for idx, row in data.iterrows()]
    reAdmitFeatures += [colName]
  reAdmitFeatures += ['re_admitted', 'timePassedFromPrevAdmit']
  return data, reAdmitFeatures


def addFeatures(ddParams, hospital, data, end):
  '''
  start, end: start and end time of input census data we tend to add new features
  patientData: a data frame of patient information used to build census data
  patLookup: a class storing all historical average of los for different categorial features
  '''
  data['month'] = data['TimeStamp'].apply(lambda x: x.month)
  data['year'] = data['TimeStamp'].apply(lambda x: x.year)
  data['season'] = data['TimeStamp'].apply(lambda x: ipUtils.get_season(x))
  # it seems some facilities are more restrict on qualified patients clsoe to the end of month
  data['dayofMonth'] = data['TimeStamp'].apply(lambda x: 1 if x.day > 15 else 0)
  features = ['month', 'year', 'season', 'dayofMonth']
  start0 = time.time()
  data, admitFeatures = addCategoricalAdmitFeat(ddParams, hospital, data, end)
  features += admitFeatures
  end0 = time.time()
  print 'admitFeatures', end0 - start0
  data, surgFeatures = addSurgFeatures(ddParams, hospital, data, end)
  features += surgFeatures
  data, unitFeatures = addUnitLevelFeatures(ddParams, hospital, data)
  features += unitFeatures
  end2 = time.time()
  print 'unitFeatures', end2 - end0
  data, medFeatures = addMedicationFeatures(ddParams, hospital, data)
  features += medFeatures
  end3 = time.time()
  print 'medFeatures', end3 - end2
  # for Packard we need to turn off raw_er features
  data, ERfeatures = addERFeatures(ddParams, hospital, data)
  features += ERfeatures
  data, procOrderFearures = addProcOrderFeatures(ddParams, hospital, data)
  features += procOrderFearures
  end4 = time.time()
  print 'procOrderFearures', end4 - end3
  # if patient is re-admitted
  data, reAdmitFeatures = addPrevAdmitInfoFeatures(ddParams, hospital, data, end)
  features += reAdmitFeatures
  end5 = time.time()
  print 're admission', end5 - end4

  features += ['currentLOS']

  # assuming age is in month, ages are in months corresponds to [0, 0.5, 2, 17, 40, 64, 74, 84, 94, 110] in years
  ageFeatureNames = ipUtils.addAgeFeatures(data, ageSteps=[0, 6, 24, 204, 480, 768, 888, 1008, 1128, 1320])
  features += ageFeatureNames

  return data, features


def cachedTrainData(ddParams, hospital, start, end):
  '''
  a function to return data and features for training and cross validation
  '''
  end1 = time.time()
  raw_data = getHistoricalPatients(ddParams, hospital, start, end)
  end2 = time.time()
  print 'raw_data', end2 - end1

  # add medication data
  if ddParams['med_admin_table']:
    raw_data_med = ipUtils.addExtraColumnsToRawData(ddParams, hospital, raw_data, start=start - timedelta(days=1),
                                                    end=end + timedelta(days=1), enterColumn='admit_time',
                                                    exitColumn='discharge_time')
  else:
    raw_data_med = raw_data.copy()
  end3 = time.time()
  print 'medication', end3 - end2
  # we blow up data to have time stamp for patients during thier stay
  patientsData = ipUtils.createTimeStampedData(raw_data_med, enterColumn='admit_time', exitColumn='discharge_time',
                                               specificHour=ddParams['triggerTime'])
  
  # only keep TimeStamp created in the range
  patientsData = patientsData[(patientsData.TimeStamp <= end) & (patientsData.TimeStamp >= start)]
  # we do not want to triger on the discharge day to avid obvious senarios
  patientsData = patientsData[patientsData.TimeStamp.dt.day != patientsData.discharge_time.dt.day]
  # current los of patient based on number of passed nights
  patientsData['TimeStampRounded'] = patientsData['TimeStamp'].apply(lambda x: x.date())
  patientsData['admit_timeRounded'] = patientsData['admit_time'].apply(lambda x: x.date())
  patientsData['currentLOS'] = patientsData.apply(lambda x: (x['TimeStampRounded'] - x['admit_timeRounded']).days, axis=1)
  # we add OR features if patients had surgeries
  if ddParams['raw_or_thru_table']:
    patientsData = ipUtils.addORColumns(data=patientsData, params=ddParams, hospital=hospital, unitFilter=False)
  else:
    patientsData = patientsData.copy()
  end4 = time.time()
  print 'or data', end4 - end3
  return patientsData


@method_cache(60 * 60 * 12)
def cachedTrainModel(ddParams, hospital, start, end):
  patientsData = cachedTrainData(ddParams, hospital, start, end)
  '''
    to filter patients from training! assuming we trigger only for first few days of patient stay,
    filtering only based on currentLOS confuse the model since long stay patients might become discharged
    as a result of next days events. but about we watnt to predict on a current patient with a potential
    long stay then the model does not know about his pattern and his strong indicators.
    I think, since we design testing on only desired patients the model should automatically pick up
    relevant features if there are not many samples of long stay patients. In the end, I rather to filter
    on currentLOS to have info on those patients too!
    '''
  patientsData = patientsData[(patientsData.currentLOS > float(ddParams['min_currentLOS'])) &
              (patientsData.currentLOS < max(float(ddParams['max_currentLOS']), 10.0))]
  data, featureNames = addFeatures(ddParams, hospital, patientsData, end)
  # we do a log transformation for target variable
  data['log_los'] = data['los'].apply(lambda x: np.log(1+x))
  trainX = data[featureNames]
  if len(trainX) <= 0:
    raise ValueError("No training data for discharge disposition prediction!")  
  model = fit_model(ddParams, hospital, data, featureNames, targetColumn='log_los')
  '''
  logger.info(event=event.IP_DISCHARGE_DISPOSITION_PREDICTION_LOOP,
              msg='Discharge Disposition Prediction training data size',
              username='admin',
              sample_size=len(data['targetDispositions'].tolist()))
  '''
  return model, featureNames


def trainModel(ddParams, hospital, recDt):
  trainingStart = recDt - timedelta(days=ddParams['historicalDays'])
  trainingEnd = recDt
  model, featureNames = cachedTrainModel(ddParams, hospital, trainingStart, trainingEnd)
  return model, featureNames


def testModel(ddParams, hospital, featureNames, model, curDate):
  currentpatients = getCurrentPatients(ddParams, hospital, curDate)
  currentpatients['TimeStamp'] = curDate
  # add medication data
  currentpatients = ipUtils.addExtraColumnsToRawData(ddParams, hospital, currentpatients,
                                                     start=curDate - timedelta(days=3),
                                                     end=curDate + timedelta(days=3), enterColumn='admit_time',
                                                     exitColumn='discharge_time')
  # we add OR features if patients had surgeries
  if ddParams['raw_or_thru_table']:
    currentpatients = ipUtils.addORColumns(data=currentpatients, params=ddParams, hospital=hospital, unitFilter=False)
  # current los of patient based on number of passed nights
  currentpatients['TimeStampRounded'] = currentpatients['TimeStamp'].apply(lambda x: x.date())
  currentpatients['admit_timeRounded'] = currentpatients['admit_time'].apply(lambda x: x.date())
  currentpatients['currentLOS'] = currentpatients.apply(lambda x: (x['TimeStampRounded'] - x['admit_timeRounded']).days, axis=1)
  # we predict only for those patients with specific current length of stay
  currentpatients = currentpatients[(currentpatients.currentLOS > float(ddParams['min_currentLOS'])) &
              (currentpatients.currentLOS < float(ddParams['max_currentLOS']))]
  test, featureNames_test = addFeatures(ddParams, hospital, currentpatients, end=curDate - timedelta(days=1))
  # sometimes encounter_master is a day behind then we need to remove patients already discharged
  if ddParams['raw_adt_segment_table']:
    if 'connection' in ddParams.keys() and ddParams['connection'] == 'redshift':
      sqlConnection = connections['redshift']
    else:
      sqlConnection = connection
    query = '''
      SELECT instance2, enter_time, exit_time, current_bed, current_unit
      FROM raw_adt_segment
      WHERE hospital = {hospital} and instance2 in {instances}
      ORDER BY enter_time desc;
    '''.format(**{'hospital': hospital.id, 'instances': "('%s')" % "','".join(set(test.instance2.values))})
    patientLocation = pd.read_sql_query(query, sqlConnection)
    patientLocation = patientLocation.groupby('instance2').head(1)
    patinetsNotDischarged = patientLocation[(patientLocation.exit_time.isnull()) | (patientLocation.exit_time > curDate)]
    test = test[test.instance2.isin(patinetsNotDischarged.instance2.tolist())]
  # sometimes some features are missing
  for f in featureNames:
    if f not in featureNames_test:
      test[f] = 0
  testX = test[featureNames]
  # We already pass in model.
  predictions = model.predict(testX) if len(testX) > 0 else []
  test['prediction'] = predictions
  # since we used log transform of los for fitted model
  test['prediction'] = test['prediction'].apply(lambda x: np.exp(x) - 1)
  test.to_csv('predictions.csv')
  return test


def fit_model(ddParams, hospital, trainData, featureNames, targetColumn='los'):
  '''
   this function fits the model to given trainData and returns a trained model
   if model is given then it simply fits it, otherwise it uses a random grid search CV to tune random
   forest model paramter and then return a fitted mode based on the best found estimator
   since random grid search does not return rpecision-recall then we apply another cv to calcuate it (we could
    avoid it if we write our own grid earch cv)
  '''
  # Create a pipeline of preprocessors and model
  # first we need to have some preprocessors to convert categorical features to numereical values
  admitCategoricalColumns = ['admitting_service', 'admitting_diagnosis', 'admit_source', 'admitting_class', 'admitting_dx_code']
  surgCategoricalColumns = ['procedure_name_or', 'procedure_specialty', 'surgery_room', 'surgeon_name', 'asa_class']
  # we hot coding the rest of remaining categorical columns based on thier top common values
  remainingCategoricalColumns = []
  for col in featureNames:
    if trainData[col].dtype == np.object_ and col not in (surgCategoricalColumns):
      remainingCategoricalColumns.append(col)
  print 'remainingCategoricalColumns', remainingCategoricalColumns
  # we need to inlcude the below columns for preprocessing step in pipeline (it will be drop out automatically)
  featureNames += [targetColumn]
  # we numerize encounter_master features based on their average historical length of stay.
  AdmitfeaturesNumerizer = ipUtils.HistoricalBiasFeat(featureList=admitCategoricalColumns, target=targetColumn, addNewCol=True)
  # we numerize surg features based on their average historical actual duration.
  if ddParams['raw_or_thru_table']:
    featureNames += ['actualDuration']
    surgfeaturesNumerizer = ipUtils.HistoricalBiasFeat(featureList=surgCategoricalColumns, target='actualDuration')
    print 'raw_or_thru_table'
  else:
    surgfeaturesNumerizer = ipUtils.UnitaryTransformer()
  Binarizer = ipUtils.OneHotMapper(columns=remainingCategoricalColumns, n_values=10)
  # simple feature selection to remove zero variance features
  selector = VarianceThreshold()
  if ddParams['model']:
    # if a model is provided then there wont be any auto modeling
    model = ipUtils.getModel(ddParams['model'], ddParams['modelParams'])
    pipeline = make_pipeline(AdmitfeaturesNumerizer, surgfeaturesNumerizer, Binarizer, selector, model)
  else:
    # build a regressor
    clf = RandomForestRegressor()
    estimator = Pipeline([
                   ('feature_selection', SelectPercentile()),
                   ('classification', RandomForestRegressor())])
    
    # specify parameters and distributions to sample from
    param_dist = {"classification__n_estimators": range(20, 1000, 50),
                  "classification__max_depth": [4, 7, 10, None],
                  "classification__max_features": [None, 'sqrt', 'log2'] * 5,
                  "classification__min_samples_split": range(4, 11),
                  "classification__criterion": ["mse"],
                  "feature_selection__percentile": [40, 60, 80, 100] * 5}
    # run randomized search
    randSearchCV = RandomizedSearchCV(estimator,
                                      param_distributions=param_dist,
                                      scoring='mean_absolute_error',
                                      n_jobs=1,
                                      n_iter=20,
                                      cv=3,
                                      random_state=1234)
    pipeline = make_pipeline(AdmitfeaturesNumerizer, surgfeaturesNumerizer, Binarizer, selector, randSearchCV)
    pipeline.fit(trainData[featureNames], trainData[targetColumn])
    # we need to update the last step of pipeline with the best fitted estimator
    model = pipeline.named_steps['randomizedsearchcv'].best_estimator_
    pipeline.steps[-1] = ('bestmodel', model)
  # Fit the model on all available data.
  pipeline.fit(trainData[featureNames], trainData[targetColumn])
  return pipeline
 
# ----------------------------------------------------------------------------------
class LOS_prediction(SpecializedAction):
  """
  Predicting if number of patients in a unit (CVICU in this case) few days ahead (default 3 days)
  at/over a thershold(default 19) as a full, otherwise not full
  """

  VERSION = '1.0.0'
  TRAINING_FREQUENCY = timedelta(days=7)
  NEEDS_ASYNC_TRAINING = True
  HAS_PERSONALIZED_OPTIONS = False
  '''
  numDaysAhead: number of days ahead we will do prediction , the default is end of september 2014
  maxCapWindow : max number of patients cross this window of time will be used as target variable
  triggerTime : hour of day we pan to trigger

  '''
  defaultParameters = {
    'connection': 'rds',
    'num_pred_classes': 1,
    'historicalDays': 100,
    'triggerTime': 7,
    'min_currentLOS': 3.0,
    'max_currentLOS': 10.0,
    'precisionThreshold': 0.6,
    'recallThreshold': 0.1,
    'startSql': 'admit_time',
    'endSql': 'discharge_time',
    'targetDispositions': [
      'Acute Inpatient Hospital',
      'Skilled Nursing Facility/Intermediate Facility',
      'Hospice-Medical Facility',
      'Inpatient Rehab',
      'Residential Care Facility',
      'Long Term Care/Subacute Facility'
    ],
    'extraFilter': None,
    'encounterFilter': None,
    'template': '''
                  Our model predicts the patient {instance2} will be discharged on {pred}.
                ''',
    'personalized': False,
    # Not in the wiki. Largely for testing.
    'model': None,
    'modelParams': {},
    'minPerBiasBucket': 4,
    'raw_er_table': False,
    'raw_or_thru_table': False,
    'raw_adt_segment_table': False,
    'med_admin_table': True,
    'raw_or_schedule_table': False,
    'proc_orders_table': True,
    'med_admin_admin_types': None,
    'med_admin_routes': ['INTRA-ARTICULAR', 'INTRABLADDER', 'INTRANASAL', 'INTRAVENOUS', 'ORAL',
                         'SUBCUTANEOUS', 'SUBLINGUAL'],
    'VasoactiveMedList': ['MILRINONE', 'DOPAMINE', 'DOBUTAMINE', 'EPINEPHRINE', 'NOREPI', 'VASOPRESSIN',
                          'NIPRIDE', 'NITROGLYCERINE', 'NICARDIPINE', 'CALCIUM'],
    'chief_complaintList': ['stroke', 'parkinsons', 'sclerosis', 'hip', 'brain', 'osteoarthritis',
                            'multiple trauma', 'fall', 'altered mental status', 'extremity weakness',
                            'shortness of breath'],
    'proc_orders_order_status_not': 'CANCELED',
    'proc_order_procedures': ['MECHANICAL VENTILATION', 'OSCILLATOR VENT',
                              'OXYGEN VIA DEVICE TO KEEP O2 SAT BETWEEN',
                              'OXYGEN, NASAL CANNULA', 'CPAP TREATMENT- FACE MASK',
                              'INTUBATION', 'JET VENT', 'RT TRACH CARE',
                              'NASOTRACHEAL SUCTIONING', 'MECHANICAL VENT PER RT',
                              'MECHANICAL VENTILATION, NON-INVASIVE',
                              'WOUND OSTOMY EVAL AND TREAT',
                              'DIET DYSPHAGIA', 'DIET NPO'],
    'proc_order_order_types': ['DIET', 'PT', 'OT', 'WOUND OSTOMY', 'PROCEDURES', 'NEUROLOGY', 'IV', 'RESPIRATORY CARE'],
    'raw_or_thru_ProcedureNames': ['TOTAL KNEE ARTHROPLASTY', 'NJX DX/THER SBST EPIDURAL/SUBARACH LUMBAR/SACRAL',
                                   'TOTAL HIP ARTHROPLASTY', 'FEMORAL FX, OPEN TX', 'SPINE SURGERY PROCEDURE UNLISTED',
                                   'NERVOUS SYSTEM SURGERY UNLISTED', 'ALLOGRAFT FOR SPINE SURGERY ONLY STRUCTURAL',
                                   'STEREOTACTIC BRAIN BX,ASPIR,EXC', 'REMOVAL DEEP IMPLANT',
                                   'OPEN SKULL EVAC HEMATOMA, SUPRATENTORIAL, SUB/ EXTRADURAL',
                                   'POSTERIOR NON-SEGMENTAL INSTRUMENTATION',
                                   'ALLOGRAFT FOR SPINE SURGERY ONLY MORSELIZED',
                                   'REVISE KNEE JOINT REPLACE,1 PART',
                                   'REVISE TOTAL HIP REPLACEMENT', 'OPEN TX TIBIAL FRACTURE PROXIMAL UNICONDYLAR',
                                   'SPINE FUSN,POST TECH,EA ADDNL SGMT', 'VASCULAR SURGERY PROCEDURE UNLIST',
                                   'FEMUR/KNEE SURG UNLISTED',
                                   'PELVIS/HIP JOINT SURGERY UNLISTED', 'CRANIAL DECOMPRESSN POST FOSSA',
                                   'LIGMT REVISION,KNEE,EXTRA-ARTIC',
                                   'SUBOCCIPT DECOMP MEDULLA/SP CRD', 'PERC VERTEB AUGMENT/ KYPHOPLAST, THOR',
                                   'CLOSED RX TRAUMA HIP DISLOC,ANESTH',
                                   'AMPUTATE THIGH,THRU FEMUR', 'IMPLANT SPINAL NEUROSTIM/RECEIVER',
                                   'REVISION OF CRANIAL NERVE',
                                   'IMPLANT/REPLACE HEAR AID,TEMP BONE', 'CRANIECT EXPL/DECOM CRANIAL NER',
                                   'REMOVAL OF KNEE PROSTHESIS',
                                   'REVISE KNEE JOINT REPLACE,ALL PARTS', 'CRANIOTOMY,REPAIR,DURAL/CSF LEAK',
                                   'REVISE/REMOVE SPINAL NEUROSTIM/RECEIVER',
                                   'DISARTICULATION OF HIP', 'LUMBAR SPINE FUSION,ANTER APPRCH',
                                   'REMOVAL OF HIP PROSTHESIS,COMPLEX', 'BRAIN AVM SURG,SUPRATENT,SIMPLE'],
    'raw_or_thru_ProcedureSpecialty': ['ORTHOPEDICS', 'NEUROLOGICAL SURGERY', 'CARDIOVASCULAR'],
    'encounter_master_admitServices': ['ORTHOPEDICS', 'NEUROLOGY', 'MEDICAL ICU', 'NEURO TRAUMA ICU'],
    'encounter_master_admitDiagnosises': ['HEADACHE', 'DIZZINESS AND GIDDINESS', 'SYNCOPE AND COLLAPSE',
                                          'ALTERED MENTAL STATUS, UNSPECIFIED',
                                          'THORACIC OR LUMBOSACRAL NEURITIS OR RADICULITIS, UNSPECIFIED',
                                          'ALTERED MENTAL STATUS', 'LUMBOSACRAL SPONDYLOSIS WITHOUT MYELOPATHY',
                                          'RADICULOPATHY, LUMBAR REGION', 'HEAD INJURY, UNSPECIFIED',
                                          'PAIN IN JOINT, PELVIC REGION AND THIGH', 'UNSPECIFIED INJURY OF HEAD, INITIAL ENCOUNTER',
                                          'INJURY, OTHER AND UNSPECIFIED, KNEE, LEG, ANKLE, AND FOOT', 'UNSPECIFIED CONVULSIONS',
                                          'SWELLING, MASS, OR LUMP IN HEAD AND NECK', 'UNSPECIFIED SEPTICEMIA', 'PAIN IN LEFT KNEE',
                                          'PAIN IN RIGHT HIP', 'PAIN IN RIGHT KNEE', 'PAIN IN LEFT HIP',
                                          'OTHER SPECIFIED INJURIES OF HEAD, INITIAL ENCOUNTER', 'CEREBRAL ANEURYSM, NONRUPTURED',
                                          'ENCEPHALOPATHY, UNSPECIFIED', 'INTRACEREBRAL HEMORRHAGE'],
    'encounter_master_admitSources': ['TRANSFER FROM A HOSPITAL DIFFERENT FACILITY',
                                      'TRANSFER FROM ANOTHER HEALTHCARE FACILITY',
                                      'TRANSFER FROM SKILLED NURS FAC (SNF) OR INTERMEDIATE CARE FACILITY'],
    'ICU_unit_names': ['OKLC ICU A', 'OKLC ICU B', 'OKLC ICU C'],
    'mapDispositionNames': {},
    'virtualBedsList': [''],
    'proc_order_procedures_similarNames': ['TUBE', 'BIPAP', 'MASK', 'OXYGEN', 'XR', 'IV'],
    'log_patientList': False
  }

  def log_actuals(self, action, now_dt=None):

    if now_dt is None:
      now_dt = datetime.utcnow()

    now_dt = datetime(now_dt.year,
                      now_dt.month,
                      now_dt.day,
                      now_dt.hour,
                      30 if now_dt.minute >= 10 else 0)

    query = """
    SELECT
      instance2,
      discharge_disposition,
      discharge_time
    FROM
      encounter_master
    WHERE
      discharge_time >= '{}' AND
      discharge_time IS NOT NULL AND
      discharge_disposition IS NOT NULL
    """.format(now_dt - timedelta(days=10))

    actual_data = pd.read_sql_query(query, connection)
    for index, row in actual_data.iterrows():
      discharge_disposition = row['discharge_disposition']
      discharge_time = row['discharge_time']
      instance2 = row['instance2']

      info = {
        'recommendation_id': instance2,
        'dl_name': self.__class__.__name__,
        'hospital_id': action.hospital.id,
        'hospital_name': action.hospital.name,
        'actual_float': discharge_time,
        'execution_dt': discharge_time.replace(tzinfo=pytz.timezone(action.hospital.timezone)).astimezone(pytz.utc),
        'deployment_name': AMD_client,
        'table': 'actuals'
      }
      logger.info(info)

  def generateTestingData(self, userProfile, action, start, end):
    '''
     start and end determines start and end time of the test time window. in case we want score based on random cv
     then it is better to set 25% larger number of historical days.
    '''
    start = datetime(start.year, start.month, start.day, start.hour)
    end = datetime(end.year, end.month, end.day, end.hour)
    defaultParams = LOS_prediction.defaultParameters
    ddParams = self.processParameters(defaultParams, action.parameters)
    hospital = action.hospital
    train = cachedTrainData(ddParams, hospital, start - timedelta(days=ddParams['historicalDays']), start)
    trainData, featureNames = addFeatures(ddParams, hospital, train, start)
    trainData['log_los'] = trainData['los'].apply(lambda x: np.log(1+x))
    trainData = trainData[(trainData.currentLOS > float(ddParams['min_currentLOS'])) &
                           (trainData.currentLOS < float(ddParams['max_currentLOS']))]
    # intentional oversamplinga and downsampling
    '''
    ddParams_temp = ddParams.copy()
    ddParams_temp['historicalDays'] = 2 * ddParams['historicalDays']
    ddParams_temp['encounterFilter'] = "discharge_disposition IN {}".format(tuple(ddParams['targetDispositions']))
    train2 = cachedTrainData(ddParams_temp, hospital, start - timedelta(days=ddParams['historicalDays']), start)
    train2, featureNames = addFeatures(ddParams_temp, hospital, train, start)
    train = train.append(train2)
    '''
    test = cachedTrainData(ddParams, hospital, start, end)
    test, featureNames_test = addFeatures(ddParams, hospital, test, start)
    # we predict only for those patients with specific current length of stay
    test = test[(test.currentLOS > float(ddParams['min_currentLOS'])) &
                (test.currentLOS < float(ddParams['max_currentLOS']))]
    storeData = {'train': trainData, 'featureNames':featureNames, 'test': test}
    import pickle
    f=open('storeData_unit2.pkl', 'wb')
    pickle.dump(storeData, f)
    f.close()
    model = fit_model(ddParams, hospital, trainData, featureNames[:], targetColumn='log_los')
    pred = model.predict(test[featureNames[:]])
    test['prediction'] = pred
    test['prediction'] = test['prediction'].apply(lambda x: np.exp(x) - 1)
    # ret = {'y_true': test['los'], 'y_pred': pred}
    return test

  def trainModel(self, action, recDt=None):
    if recDt is None:
        recDt = datetime.today()
    recDt = datetime(recDt.year, recDt.month, recDt.day, recDt.hour)
    defaultParams = LOS_prediction.defaultParameters
    ddParams = self.processParameters(defaultParams, action.parameters)
    model, featureNames = trainModel(ddParams, action.hospital, recDt)

    return {'featureNames': featureNames, 'model': model}

  def getNotificationTemplate(self, ddParams):
    ret = ddParams['template']
    return ret

  def getOptions(self, action, execDt, recDt):
    originalRecDt = datetime(recDt.year, recDt.month, recDt.day, recDt.hour, recDt.minute, recDt.second)
    # to differenciate triggers for the same patient on different days
    predictionDay = datetime(recDt.year, recDt.month, recDt.day)
    defaultParams = LOS_prediction.defaultParameters
    ddParams = self.processParameters(defaultParams, action.parameters)

    # Check the db to see if there is a fitted model there.
    modelParams = action.getMostRecentFittedModel(self)
     
    if modelParams is None:
      #return []
      modelParams =  self.trainModel(action)

    model = modelParams['model']
    featureNames = modelParams['featureNames']
    results = testModel(ddParams, action.hospital, featureNames, model, recDt)
    
    #results=pd.read_csv('predictions.csv')
    ret = []
    output = {}
    for idx, row in results.iterrows():
      # Log prediction results.
      rec_dt = originalRecDt.replace(tzinfo=pytz.timezone(action.hospital.timezone)).astimezone(pytz.utc)
      recommend_dt = originalRecDt.replace(tzinfo=pytz.timezone(action.hospital.timezone)).astimezone(pytz.utc)
      info = {'execution_dt': rec_dt,
              'recommendation_dt': recommend_dt,
              'plannedaction_id': action.id,
              'dl_version': self.VERSION,
              'dl_name': self.__class__.__name__,
              'recommendation_id': (row['instance2'], predictionDay),
              'hospital_id': action.hospital.id,
              'hospital_name': action.hospital.name,
              'prediction_text': row['prediction'],
              'deployment_name': AMD_client,
              'table': 'predictions',
              'meta': {
                       'instance2': row['instance2'],
                       'admit_time': row['admit_time']
              }}
      if (not ddParams['personalized']) :
        templateVars = {
          'instance2': row['instance2'],
          'pred': row['prediction']
        }
        notifTemplate = self.getNotificationTemplate(ddParams)
        actionString = notifTemplate.format(**templateVars)
        option = ActionOption(action, actionString, 100.0, originalRecDt)
        # Duration because they might change scheduled time tomorrow, when this class might still run.
        option.setUniqueId('%s' % str(row['instance2']))
        option.updateMetaData('prediction', str(row['prediction']))
        ret.append(option)
        print row['instance2'],row['prediction']
        roundedAdmitTime = datetime(row['admit_time'].year, row['admit_time'].month, row['admit_time'].day)
        query = " select instance1c from idr_patient_list where instance1='{instance}' and admit_time > '{start}' and admit_time < '{end}';"
        ins = pd.read_sql_query(query.format(instance=row['instance1'], start=roundedAdmitTime, end=roundedAdmitTime+timedelta(days=1)), connection, coerce_float=True)
        print query.format(instance=row['instance1'], start=roundedAdmitTime, end=roundedAdmitTime+timedelta(days=1))
        print ins
        ins = ins['instance1c'].tolist()[0] if len(ins['instance1c'])!=0 else ''
        print 'ins ', ins
        # find current_bed from encounter_master 
        encounter_master_query = '''
          select admit_time, discharge_time, bed from encounter_master
          where hospital = {hospital} and {patient_id} = '{instance}';
        '''
        encounter_data = pd.read_sql_query(encounter_master_query.format(hospital=action.hospital.pk,patient_id='instance2',  instance=row['instance2']),
                                           connection, coerce_float=True)
        print encounter_master_query.format(hospital=action.hospital.pk,patient_id='instance2',  instance=row['instance2'])
        current_bed = encounter_data['bed'].tolist()[0] if len(encounter_data['bed'])!=0 else ''
        print 'predict----> ', ins, round(row['prediction'],2), current_bed
        
        output, _ = IPLOSOutbound.objects.update_or_create(hospital=action.hospital,
                                                           instance1=row['instance1'],
                                                           instance2=row['instance2'],
                                                           instance3=ins,
                                                           patient_id=row['instance2'],
                                                           plannedAction=action,
                                                           model_output_time=originalRecDt,
                                                           model_output_value=round(row['prediction'],2),
                                                           model_output_confidence=0,
                                                           current_bed=current_bed
                                                          )
    
        # Log the fact that this a warning might be triggered.
        info['meta']['trigger'] = '{}'.format(True)
        info['meta']['trigger_id'] = '{}'.format(option.getUniqueId())
      logger.info(info)

    
    return ret
