"""Read patient and duration fields from the fixed EDF header."""


def baseInfo(file_path):
    """Return patient id, sex, age, start time, and duration from the EDF header."""
    with open(file_path, 'rb') as f:
        header_bytes = f.read(256)  # Fixed EDF header prefix.

    if len(header_bytes) < 256:
        raise ValueError("File is less than 256 bytes; not a complete EDF header")

    # EDF header fields in spec order. The result keeps the patient fields and total duration.
    edf_version = header_bytes[0:8].decode('ascii').strip()
    patient_id = header_bytes[8:88].decode('ascii').strip()
    recording_id = header_bytes[88:168].decode('ascii').strip()
    startdate = header_bytes[168:176].decode('ascii').strip()  # dd.mm.yy
    starttime = header_bytes[176:184].decode('ascii').strip()  # hh.mm.ss
    header_bytes_length = header_bytes[184:192].decode('ascii').strip()
    reserved = header_bytes[192:236].decode('ascii').strip()
    num_data_records = header_bytes[236:244].decode('ascii').strip()
    duration_of_record = header_bytes[244:252].decode('ascii').strip()  # Seconds per data record.
    num_signals = header_bytes[252:260].decode('ascii').strip()

    patient_parts = patient_id.split()
    patient_number = patient_parts[0] if len(patient_parts) > 0 else None
    sex = patient_parts[1] if len(patient_parts) > 1 and patient_parts[1] in ['M', 'F'] else None
    age_str = next((p for p in patient_parts if p.startswith("Age:")), None)  # Local header form "Age:XX".
    age = int(age_str[4:]) if age_str else None

    minimal_info = {
        "patient_id": patient_number,
        "sex": sex,
        "age": age,
        "start_date": startdate,
        "start_time": starttime,
        "data_duration": f"[0 - {int(num_data_records) * float(duration_of_record)}]",  # Total seconds.
    }

    minimal_info = {k: v for k, v in minimal_info.items() if v is not None}  # Omit missing header fields.
    return minimal_info


def get_age_factor(age, age_factors):
    """Return the age-factor band whose min_age/max_age contains this age, or None."""
    for factor in age_factors:
        if factor["min_age"] <= age < factor["max_age"]:
            return factor
    return None  # No configured band contains this age.
