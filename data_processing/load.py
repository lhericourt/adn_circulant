from pathlib import Path
from collections import defaultdict
from typing import Tuple

import pandas as pd


DATA_DIR = Path('.') / 'data' / 'raw' / 'results'
METADATA_PATH = Path('.') / 'data' / 'raw' / 'metadata_DMLHT_ctDNA.csv'
MERGED_DIR = Path('.') / 'data' / 'merged'


def load_metadata(metadata_path: Path):
    metadata = pd.read_csv(metadata_path, sep='\t', skiprows=1)
    metadata = metadata.rename(columns={'Code': 'patient_id'})
    metadata = metadata.sort_values('cpa_diagnostic', ascending=False).drop_duplicates(['patient_id'],
                                                                                       keep='first')
    return metadata


def get_infos_from_filename(filename: str) -> Tuple[str, str]:
    filename = filename.split('.')[0]
    splitted_filename = filename.split('_')
    id_patient = f'{splitted_filename[0]}_{splitted_filename[1]}'
    time_point = f'{splitted_filename[2]}_{splitted_filename[3]}' if len(splitted_filename) == 4 else splitted_filename[2]
    return id_patient, time_point


def load_all_data_patient(data_dir: Path) -> Tuple[dict, dict, dict]:
    print('Load raw patient data')
    data_patient = defaultdict(dict)

    for p in data_dir.iterdir():
        if not p.is_file():
            continue
        filename = str(p).split('/')[-1]
        try:
            id_patient, time_point = get_infos_from_filename(filename)
        except IndexError as e:
            print(f'Error reading file : {filename}, {e}')
            continue
        data_patient[id_patient][time_point] = pd.read_csv(p, sep='\t')
    print(f'There are {len(data_patient)} patient')

    patient_diag_time_point = {patient_id: {'DIAG': data['DIAG']} for patient_id, data in data_patient.items() if
                               data.get('DIAG') is not None}
    print(f'There are {len(patient_diag_time_point)} patient with data for DIAG time point')

    patient_chir_time_point = {patient_id: {'AVANT_CHIR': data['AVANT_CHIR']} for patient_id, data in
                               data_patient.items() if data.get('AVANT_CHIR') is not None}
    print(f'There are {len(patient_chir_time_point)} patient with data for AVANT_CHIR time point')

    patient_end_time_point = {patient_id: {'FIN_TT': data['FIN_TT']} for patient_id, data in data_patient.items() if
                              data.get('FIN_TT') is not None}
    print(f'There are {len(patient_end_time_point)} patient with data for FIN_TT time point')

    return patient_diag_time_point, patient_chir_time_point, patient_end_time_point


def transpose_data_one_patient(data: pd.DataFrame) -> pd.DataFrame:
    one_data = data.copy()
    one_data = one_data[['id', 'ratio', 'zscore']].set_index('id').stack().reset_index()
    one_data.columns = ['id', 'score_label', 'value']
    one_data['col_name'] = one_data['id'] + '_' + one_data['score_label']
    return one_data.set_index('col_name')['value'].to_frame().T


def format_all_data_as_df(patient_data: dict) -> pd.DataFrame:
    transpose_data = []
    ids_patient = []

    for patient_id, data in patient_data.items():
        for time_point, data in data.items():
            transpose_data.append(transpose_data_one_patient(data))
            ids_patient.append((patient_id, time_point))

    transpose_data_df = pd.concat(transpose_data, axis=0).reset_index(drop=True)
    ids_patient_df = pd.DataFrame(ids_patient, columns=['patient_id', 'time_point'])

    return pd.concat([ids_patient_df, transpose_data_df], axis=1)


def pivot_data(patient_all_time_points_df: pd.DataFrame) -> pd.DataFrame:
    """create 1 column per time_point x gene col"""
    pivoted = patient_all_time_points_df.pivot(index=['patient_id'], columns=['time_point'])
    pivoted.columns = ['_'.join(col).strip() for col in pivoted.columns.values]
    pivoted = pivoted.reset_index()
    return pivoted


def merge_and_save_patient_data(data_one_time_point: pd.DataFrame, metadata: pd.DataFrame, filepath: Path) -> pd.DataFrame:
    data_one_time_point = pd.merge(metadata, data_one_time_point, on='patient_id', how='inner')
    data_one_time_point.to_csv(filepath, index=False)
    return data_one_time_point


if __name__ == '__main__':
    metadata_df = load_metadata(METADATA_PATH)

    patient_diag_time_point, patient_chir_time_point, patient_end_time_point = load_all_data_patient(DATA_DIR)

    patient_diag_time_point_df = format_all_data_as_df(patient_diag_time_point)
    patient_diag_time_point_df = merge_and_save_patient_data(patient_diag_time_point_df, metadata_df,
                                                             filepath=MERGED_DIR / 'diag_data.csv')

    patient_chir_time_point_df = format_all_data_as_df(patient_chir_time_point)
    patient_chir_time_point_df = merge_and_save_patient_data(patient_chir_time_point_df, metadata_df,
                                                             filepath=MERGED_DIR / 'chir_data.csv')

    patient_end_time_point_df = format_all_data_as_df(patient_end_time_point)
    patient_end_time_point_df = merge_and_save_patient_data(patient_end_time_point_df, metadata_df,
                                                             filepath=MERGED_DIR / 'end_data.csv')


