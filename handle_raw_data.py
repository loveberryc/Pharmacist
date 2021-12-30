import pandas as pd
import numpy as np
from collections import OrderedDict
import os

import const
import utils

def generateAllBiochemistry():
    print("Generate all biochemistry ...")

    df_patients_and_body_indexes = pd.read_excel(const.patients_in_file)
    results = []
    for i in range(7, 51):
        c_name = df_patients_and_body_indexes.columns[i]
        results = results + [c_name]

    df = pd.DataFrame({'Biochemistry': results}) # Don't sort
    df.to_csv(const.biochemistry_out_file, index = False)

def generateAllDiagnosises():
    print("generate diagnosis category file ...")

    c_name = '所有诊断与编码'
    df_patients_and_diagnosis = pd.read_excel(const.diagnosis_in_file)
    diagnosis_datas = df_patients_and_diagnosis[c_name]
    
    # save all diagnosis
    all_diagnosises = []
    for d in diagnosis_datas:
        all_diagnosises = all_diagnosises + utils.convert2Diagnosises(d)
    all_diagnosises = list(OrderedDict.fromkeys(all_diagnosises))

    print("all_diagnosises length is ", len(all_diagnosises))

    df = pd.DataFrame({'Diagnosis': all_diagnosises}).sort_values(by='Diagnosis')
    df.to_csv(const.diagnosis_out_file, index = False)

def generateAllPatientIPs():
    print("generate all patient IPs ...")

    c_name = 'IP'
    df_patients_and_diagnosis = pd.read_excel(const.diagnosis_in_file)
    ip_datas = df_patients_and_diagnosis[c_name]
    
    # save all diagnosis
    results = []
    for d in ip_datas:
        results = results + [d]
    results = list(OrderedDict.fromkeys(results))

    print("all IPs length is ", len(results))

    df = pd.DataFrame({'IP': results}).sort_values(by='IP')
    df.to_csv(const.ips_out_file, index = False)

def generateAllDrugs():
    print("Start generate drugs file ...")

    c_name = '医嘱项名称'
    df_medical_orders = pd.read_excel(const.medical_advice_in_file)

    drug_datas = df_medical_orders[c_name]
    
    # print(df_medical_orders)

    # save all diagnosis
    results = []
    for d in drug_datas:
        results = results + [d]
    results = list(OrderedDict.fromkeys(results))

    print("all Drugs length is ", len(results))

    df = pd.DataFrame({'Drug': results}).sort_values(by='Drug')
    df.to_csv(const.drugs_out_file, index = False)

def __generateFeatures():
    src = [] # IP
    biochemestry_dst = [] # biochemistry
    diagnosis_dst = [] # diagnosis

    df_patients_and_biochemistry = pd.read_excel(const.patients_in_file)
    df_patients_and_diagnosis = pd.read_excel(const.diagnosis_in_file)
    df_merged = pd.merge(df_patients_and_biochemistry, df_patients_and_diagnosis, on='IP')[['IP', '1阿司匹林过敏', '12肝功能不全', '13严重肝功能不全', '14严重肾功能不全', '16CKD5', '17心功能IV级', '18高出血风险', '19出血体质', '20出血', '21显著出血', '22颅内出血', '23胃肠出血', '24产后出血', '26凝血障碍', '28肝素诱导血小板减少症', '29血小板功能障碍', '30紫癜', '31血友病', '32动脉瘤', '33动静脉畸形', '34消化性溃疡', '35高血压3级', '36恶性高血压', '37急性感染性心内膜炎', '38急性细菌性心内膜炎', '39心房颤动', '40人工心脏瓣膜', '41颅内疾病', '42颅内占位', '43卒中', '44恶性肿瘤', '45_过去6周创伤_替罗非班', '49透析', '50溶栓', '51先兆流产', '52妊娠', '53妊娠晚期', '67CHA2DS2‑VASc', '69年龄≥18岁', '70年龄≥75岁', '71年龄＞75岁', '72年龄≥80岁', '74脑动脉瘤', '75脑血管动静脉畸形', '所有诊断与编码']]

    for i in range(1, 45):
        c_name = df_merged.columns[i]
        print("column: ", c_name)
        print('-----')
        df_merged.loc[df_merged[c_name] != 'Y', c_name] = int(0)
        df_merged.loc[df_merged[c_name] == 'Y', c_name] = int(1)

    df_merged['所有诊断与编码'] = df_merged['所有诊断与编码'].apply(lambda d: ';'.join(utils.convert2Diagnosises(d)))

    return df_merged.sort_values(by='IP')

def __generateLabels():
    print("Start generate labels...")
    src = [] # IP
    dst = [] # Drug

    df_patients_and_drugs = pd.read_excel(const.medical_advice_in_file)
    
    IP = ""
    drugs = []

    for patient_index, patient_row in df_patients_and_drugs.iterrows():
        tmp_IP = patient_row["IP"]
        tmp_drug = patient_row['医嘱项名称']

        if IP != tmp_IP:
            if IP != "" and drugs.__len__() > 0:
                drugs = np.unique(drugs)
                src.append(IP)
                dst.append(";".join(drugs))
            
            IP = tmp_IP
            drugs = [tmp_drug]

        else:
            drugs.append(tmp_drug)

    if IP != "" and drugs.__len__() > 0:
            drugs = np.unique(drugs)
            src.append(IP)
            dst.append(";".join(drugs))

    return pd.DataFrame({ 'IP': src, 'Labels': dst }).sort_values(by='IP')

def generateFeaturesAndLabels():
    df_labels = __generateLabels()
    df_features = __generateFeatures()
    df = pd.merge(df_labels, df_features, on="IP")
    df.to_csv(const.labels_features_output_file, index = False)

generateAllBiochemistry()
generateAllDiagnosises()
generateAllPatientIPs()
generateAllDrugs()
generateFeaturesAndLabels()