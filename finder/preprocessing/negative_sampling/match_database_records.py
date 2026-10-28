"""Evidence screening only: no automatic negative confirmation or label changes."""
import csv
import hashlib
import json
import re
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'database_screening'
OUT.mkdir(exist_ok=True)

def load_csv(name):
    with (ROOT/name).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def save_csv(name,fields,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

def norm(value):
    value=(value or '').strip()
    if value.startswith(('http://','https://')):
        value=value.rstrip('/').rsplit('/',1)[-1]
    m=re.fullmatch(r'([A-Za-z]+)[_:](.+)',value)
    if m:
        return m.group(1).upper()+':'+m.group(2)
    return value

# Contract checks: preserve phenotype identity; no parent/child equivalence.
assert norm('EFO_0000404')==norm('EFO:0000404')
assert norm('http://purl.obolibrary.org/obo/MONDO_0014943')=='MONDO:0014943'
assert norm('EFO_0000404')!=norm('EFO_0000685')
assert norm('HP_0100543')!=norm('EFO_0100543')
assert norm('Orphanet_88619')==norm('Orphanet:88619')

summary=json.loads((ROOT/'chembl_37/download_summary.json').read_text())
assert summary['indication_download_complete'] and not summary['errors'],summary
molecules={row['molecule_chembl_id']:row for row in json.loads((ROOT/'chembl_37/molecules.json').read_text())}
chem_records=json.loads((ROOT/'chembl_37/drug_indications.json').read_text())
assert len(chem_records)==summary['expected_indication_rows']
current_disease_rows=pq.read_table(ROOT/'opentargets_26_06/disease.parquet',columns=['id','name','dbXRefs','obsoleteTerms','obsoleteXRefs','ancestors']).to_pylist()
current_diseases={norm(row['id']):row for row in current_disease_rows}
old_index=defaultdict(set)
xref_index=defaultdict(set)
for row in current_disease_rows:
    for old in row.get('obsoleteTerms') or []:
        old_index[norm(old)].add(norm(row['id']))
    for xref in row.get('dbXRefs') or []:
        xref_index[norm(xref)].add(norm(row['id']))
current_drug_rows=pq.read_table(ROOT/'opentargets_26_06/drug_molecule.parquet',columns=['id','name','parentId','childChemblIds']).to_pylist()
current_drugs={row['id']:row for row in current_drug_rows}
child_parents=defaultdict(set)
for row in current_drug_rows:
    for child in row.get('childChemblIds') or []:
        child_parents[child].add(row['id'])
historical_diseases={}
historical_drugs={}
hist_root=ROOT/'opentargets_25_09'
if (hist_root/'disease.parquet').exists():
    historical_diseases={norm(row['id']):row for row in pq.read_table(hist_root/'disease.parquet',columns=['id','name']).to_pylist()}
if (hist_root/'drug_molecule.parquet').exists():
    f=pq.ParquetFile(hist_root/'drug_molecule.parquet')
    fields=[field for field in ('id','name','parentId') if field in f.schema_arrow.names]
    historical_drugs={row['id']:row for row in pq.read_table(hist_root/'drug_molecule.parquet',columns=fields).to_pylist()}
lookup_path=ROOT/'chembl_37/raw/id_lookup_missing.json'
lookup={row['chembl_id']:row for row in json.loads(lookup_path.read_text())['chembl_id_lookups']} if lookup_path.exists() else {}

chem_by_drug=defaultdict(list)
chem_terms=defaultdict(set)
for row in chem_records:
    drug=row.get('parent_molecule_chembl_id') or row.get('molecule_chembl_id')
    chem_by_drug[drug].append(row)
    if row.get('efo_id') and row.get('efo_term'):
        chem_terms[norm(row['efo_id'])].add(row['efo_term'])
ot_indications=pq.read_table(ROOT/'opentargets_26_06/clinical_indication.parquet').to_pylist()
ot_by_drug=defaultdict(list)
for row in ot_indications:
    ot_by_drug[row['drugId']].append(row)
unique_pairs=load_csv('negative_samples_unique.csv')
original_pairs=load_csv('negative_samples.csv')
assert len(unique_pairs)==7928 and len(original_pairs)==7929

def drug_info(id_):
    molecule=molecules.get(id_)
    current=current_drugs.get(id_)
    historical=historical_drugs.get(id_)
    hierarchy=(molecule or {}).get('molecule_hierarchy') or {}
    parent=hierarchy.get('parent_chembl_id')
    source='ChEMBL_37' if parent else ''
    if not parent and current:
        parent=current.get('parentId') or current['id']
        source='OpenTargets_26.06'
    if not parent and len(child_parents[id_])==1:
        parent=next(iter(child_parents[id_]))
        source='OpenTargets_26.06_childChemblIds'
    if not parent and historical:
        parent=historical.get('parentId') or historical['id']
        source='OpenTargets_25.09_historical'
    name=(molecule or {}).get('pref_name') or (current or {}).get('name') or (historical or {}).get('name') or ''
    name_source='ChEMBL_37' if (molecule or {}).get('pref_name') else ('OpenTargets_26.06' if (current or {}).get('name') else ('OpenTargets_25.09' if (historical or {}).get('name') else ''))
    if molecule:
        state='current_record_found' if parent else 'current_record_parent_unresolved'
    elif source and source!='OpenTargets_25.09_historical':
        state='current_parent_mapping_found_original_record_missing'
    elif historical:
        state='historical_record_only'
    else:
        state='identifier_unresolved'
    return {'drug_id':id_,'drug_name':name,'drug_name_source':name_source,'parent_chembl_id':parent or '', 'parent_mapping_source':source,'drug_metadata_status':state,'chembl_id_status':lookup.get(id_,{}).get('status',''),'chembl_last_active_release':lookup.get(id_,{}).get('last_active','')}

def disease_info(id_):
    key=norm(id_)
    current=current_diseases.get(key)
    historical=historical_diseases.get(key)
    replacements=old_index.get(key,set())
    candidates=replacements or xref_index.get(key,set())
    name=(current or {}).get('name') or (historical or {}).get('name') or ' | '.join(sorted(chem_terms.get(key,set())))
    name_source='OpenTargets_26.06' if current else ('OpenTargets_25.09' if historical else ('ChEMBL_37_indication' if chem_terms.get(key) else ''))
    if not name and len(candidates)==1:
        name=current_diseases[next(iter(candidates))]['name']
        name_source='OpenTargets_26.06_mapped_term'
    state='current_id_found' if current else ('obsolete_id_replacement_found' if replacements else ('cross_reference_candidates_found' if candidates else 'current_identifier_unresolved'))
    return {'disease_id':id_,'disease_name':name or '', 'disease_name_source':name_source,'disease_metadata_status':state,'current_mapping_candidates':' | '.join(sorted(candidates))}

def disease_match(sample_id,indication_id,mesh_id=None):
    sample,indication=norm(sample_id),norm(indication_id)
    if indication and sample==indication:
        return 'exact_identifier'
    replacements=old_index.get(sample,set())
    if indication and indication in replacements:
        return 'obsolete_id_replacement_requires_review'
    if indication and (indication in xref_index.get(sample,set()) or sample in xref_index.get(indication,set())):
        return 'cross_reference_requires_review'
    samples={sample}|replacements
    indications={indication}|old_index.get(indication,set()) if indication else set()
    for s in samples:
        sd=current_diseases.get(s,{})
        ancestors={norm(value) for value in sd.get('ancestors') or []}
        if indications&ancestors:
            return 'indication_is_ancestor_requires_review'
        for i in indications:
            ancestors_i={norm(value) for value in current_diseases.get(i,{}).get('ancestors') or []}
            if s in ancestors_i:
                return 'indication_is_descendant_requires_review'
    if mesh_id:
        mesh=norm('MeSH:'+mesh_id)
        for s in samples:
            xrefs={norm(value) for value in current_diseases.get(s,{}).get('dbXRefs') or []}
            if mesh in xrefs:
                return 'mesh_cross_reference_requires_review'
    return None

# Check an actual ontology edge is classified as a relation, never equality.
for child in current_disease_rows:
    ancestors=child.get('ancestors') or []
    if ancestors and norm(child['id'])!=norm(ancestors[0]):
        assert disease_match(child['id'],ancestors[0])!='exact_identifier'
        break

all_drug_info={id_:drug_info(id_) for id_ in sorted({row['drug_id'] for row in unique_pairs})}
all_disease_info={id_:disease_info(id_) for id_ in sorted({row['disease_id'] for row in unique_pairs})}
evidence=[]
per_pair=defaultdict(list)
for pair in unique_pairs:
    index=int(pair['negative_index'])
    info=all_drug_info[pair['drug_id']]
    drug_ids={pair['drug_id']}
    if info['parent_chembl_id']:
        drug_ids.add(info['parent_chembl_id'])
    seen=set()
    for drug in sorted(drug_ids):
        for source,records in (('ChEMBL_37',chem_by_drug.get(drug,[])),('OpenTargets_26.06',ot_by_drug.get(drug,[]))):
            for record in records:
                record_id=str(record['drugind_id'] if source=='ChEMBL_37' else record['id'])
                if (source,record_id) in seen:
                    continue
                seen.add((source,record_id))
                indication_id=record.get('efo_id') if source=='ChEMBL_37' else record['diseaseId']
                method=disease_match(pair['disease_id'],indication_id,record.get('mesh_id'))
                if not method:
                    continue
                refs=record.get('indication_refs') or []
                phase=record.get('max_phase_for_ind') if source=='ChEMBL_37' else record.get('maxClinicalStage')
                approval=phase=='APPROVAL' if source=='OpenTargets_26.06' else phase is not None and float(phase)>=4
                row={'negative_index':index,'drug_id':pair['drug_id'],'drug_name':info['drug_name'],'parent_chembl_id':info['parent_chembl_id'],'parent_mapping_source':info['parent_mapping_source'],'disease_id':pair['disease_id'],'disease_name':all_disease_info[pair['disease_id']]['disease_name'],'source_dataset':source,'record_id':record_id,'match_type':method,'record_drug_id':drug,'drug_match_type':'original_identifier' if drug==pair['drug_id'] else 'parent_drug_identifier','indication_disease_id':indication_id or '', 'indication_disease_name':record.get('efo_term') or current_diseases.get(norm(indication_id),{}).get('name',''),'phase_or_stage':phase or 'UNKNOWN','phase4_or_approval_flag':int(approval),'reference_ids':' | '.join(ref.get('ref_id','') for ref in refs),'reference_urls':' | '.join(ref.get('ref_url','') for ref in refs),'opentargets_report_ids':' | '.join(record.get('clinicalReportIds') or []),'review_decision':'pending_manual_review'}
                evidence.append(row)
                per_pair[index].append(row)

needed_reports={value for row in evidence for value in row['opentargets_report_ids'].split(' | ') if value}
reports=[]
report_path=ROOT/'opentargets_26_06/clinical_report.parquet'
if needed_reports:
    file=pq.ParquetFile(report_path)
    for batch in file.iter_batches(batch_size=4096):
        reports.extend(row for row in batch.to_pylist() if row['id'] in needed_reports)
report_by_id={row['id']:row for row in reports}
for row in evidence:
    ids=[value for value in row['opentargets_report_ids'].split(' | ') if value]
    selected=[report_by_id[id_] for id_ in ids if id_ in report_by_id]
    row['opentargets_report_urls']=' | '.join(sorted({report['url'] for report in selected if report.get('url')}))
    row['opentargets_report_sources']=' | '.join(sorted({report['source'] for report in selected if report.get('source')}))
    row['opentargets_report_years']=' | '.join(map(str,sorted({report['year'] for report in selected if report.get('year')})))
    row['opentargets_report_quality_controls']=' | '.join(sorted({flag for report in selected for flag in report.get('qualityControls') or []}))
    row['opentargets_reports_with_expert_review']=sum(bool(report.get('hasExpertReview')) for report in selected)

results=[]
for pair in unique_pairs:
    index=int(pair['negative_index'])
    drug=all_drug_info[pair['drug_id']]
    disease=all_disease_info[pair['disease_id']]
    hits=per_pair[index]
    exact=[row for row in hits if row['match_type']=='exact_identifier']
    exact_approval=any(row['phase4_or_approval_flag'] for row in exact)
    unresolved=drug['drug_metadata_status'] in ('identifier_unresolved','historical_record_only','current_record_parent_unresolved') or disease['disease_metadata_status']=='current_identifier_unresolved'
    if exact_approval:
        state='exact_phase4_or_approval_record_requires_review'
    elif exact:
        state='exact_investigational_or_unknown_record_requires_review'
    elif hits:
        state='mapped_or_hierarchical_record_requires_review'
    elif unresolved:
        state='identifier_unresolved'
    else:
        state='no_record_found_in_queried_datasets'
    results.append({**pair,**drug,**disease,'screening_status':state,'identifier_resolution_issue':int(unresolved),'exact_match_records':len(exact),'other_match_records':len(hits)-len(exact),'matched_sources':' | '.join(sorted({row['source_dataset'] for row in hits})),'candidate_approval_flag':int(exact_approval),'manual_review_decision':'unreviewed','negative_confirmed':0})
result_by_index={int(row['negative_index']):row for row in results}
all_rows=[]
for original in original_pairs:
    first=int(original['duplicate_of_negative_index']) if original['is_duplicate']=='1' else int(original['negative_index'])
    all_rows.append({**result_by_index[first],**original})
fields=list(all_rows[0])
save_csv('negative_samples_enriched.csv',fields,all_rows)
save_csv('negative_samples_unique_screened.csv',list(results[0]),results)
save_csv('pairs_requiring_review.csv',list(results[0]),[row for row in results if per_pair[int(row['negative_index'])]])
save_csv('identifier_resolution_issues.csv',list(results[0]),[row for row in results if row['identifier_resolution_issue']])
evidence_fields=['negative_index','drug_id','drug_name','parent_chembl_id','parent_mapping_source','disease_id','disease_name','source_dataset','record_id','match_type','record_drug_id','drug_match_type','indication_disease_id','indication_disease_name','phase_or_stage','phase4_or_approval_flag','reference_ids','reference_urls','opentargets_report_ids','opentargets_report_urls','opentargets_report_sources','opentargets_report_years','opentargets_report_quality_controls','opentargets_reports_with_expert_review','review_decision']
save_csv('matched_evidence.csv',evidence_fields,evidence)
save_csv('drugs_enriched.csv',list(next(iter(all_drug_info.values()))),all_drug_info.values())
save_csv('diseases_enriched.csv',list(next(iter(all_disease_info.values()))),all_disease_info.values())
(OUT/'matched_opentargets_reports.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2,default=str)+'\n')
counts=Counter(row['screening_status'] for row in results)
statistics={'completed_at_utc':datetime.now(timezone.utc).isoformat(),'original_negative_rows':len(original_pairs),'unique_negative_pairs':len(results),'database_versions':['ChEMBL_37','OpenTargets_26.06'],'historical_metadata_versions':['OpenTargets_25.09'] if historical_diseases or historical_drugs else [],'database_rows':{'chembl_molecules':len(molecules),'chembl_indications':len(chem_records),'opentargets_diseases':len(current_disease_rows),'opentargets_drugs':len(current_drug_rows),'opentargets_clinical_indications':len(ot_indications),'opentargets_clinical_reports_total':pq.ParquetFile(report_path).metadata.num_rows},'screening_status_counts_unique_pairs':dict(counts),'pairs_with_any_candidate_record':sum(bool(per_pair[int(row['negative_index'])]) for row in results),'exact_identifier_matched_pairs':sum(row['exact_match_records']>0 for row in results),'exact_phase4_or_approval_matched_pairs':sum(bool(row['candidate_approval_flag']) for row in results),'evidence_rows':len(evidence),'match_type_counts_records':dict(Counter(row['match_type'] for row in evidence)),'identifier_resolution_issue_pairs':sum(row['identifier_resolution_issue'] for row in results),'drug_metadata_status_counts':dict(Counter(row['drug_metadata_status'] for row in all_drug_info.values())),'disease_metadata_status_counts':dict(Counter(row['disease_metadata_status'] for row in all_disease_info.values())),'drug_names_missing':sum(not row['drug_name'] for row in all_drug_info.values()),'disease_names_missing':sum(not row['disease_name'] for row in all_disease_info.values()),'needed_opentargets_reports':len(needed_reports),'matched_opentargets_reports':len(reports),'confirmed_negative_pairs':0,'clinicaltrials_direct_api_queried':False,'pubmed_systematic_search_performed':False,'original_evidence_cutoff_date_applied':False,'limitations':['No record found is not a confirmed negative.','Ancestor, descendant, obsolete-term and cross-reference matches require review; they are not exact disease equivalence.','Phase/stage and source hits do not establish efficacy; reports may involve multiple drugs or diseases and must be inspected.','Current 2026 releases may include evidence added after the original data cutoff.','ChEMBL and OpenTargets can share provenance.','Historical metadata is used for name recovery, not to assert an original experiment release or completed current coverage.']}
(OUT/'summary.json').write_text(json.dumps(statistics,ensure_ascii=False,indent=2)+'\n')
assert sum(counts.values())==7928
assert all(row['negative_confirmed']==0 for row in results)
assert len(all_rows)==7929
assert {row['negative_index'] for row in all_rows}=={row['negative_index'] for row in original_pairs}
assert len(report_by_id)==len(needed_reports),'Some referenced clinical reports were not found'
print(json.dumps(statistics,ensure_ascii=False,indent=2),flush=True)
print('EXACT PAIRS:')
for row in results:
    if row['exact_match_records']:
        print(json.dumps({key:row[key] for key in ('negative_index','drug_id','drug_name','disease_id','disease_name','screening_status')},ensure_ascii=False))
