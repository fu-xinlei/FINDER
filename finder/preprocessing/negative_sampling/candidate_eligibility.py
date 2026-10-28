"""Select resolved candidates with no database hits; omit measurement terms."""
def candidate_decision(row):
    if row['screening_status']!='no_record_found_in_queried_datasets' or str(row['identifier_resolution_issue'])!='0':
        return 'rejected_database_hit_or_unresolved_disease'
    if 'measurement' in row['disease_name'].casefold():
        return 'rejected_non_disease_measurement_term'
    return 'eligible_not_used'
