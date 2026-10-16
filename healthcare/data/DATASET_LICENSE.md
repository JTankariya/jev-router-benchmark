# Dataset Provenance & Licensing: Medical Transcription Samples (MTSamples)

## 1. Overview
This dataset contains real-world clinical patient transcription summaries sourced from the **MTSamples** collection (Medical Transcription Samples), widely used across academic clinical NLP research for specialty classification and triage routing.

Each record represents a de-identified clinical encounter summary documenting patient symptoms, diagnostic findings, procedures, and medical histories.

## 2. Official Licensing & Legal Status

- **Dataset Name:** Medical Transcriptions (MTSamples)
- **Original Source:** `mtsamples.com` (Compiled by Tara Boyle on Kaggle & GitHub)
- **Copyright Status:** **CC0 1.0 Universal (Public Domain Dedication)**
- **License Terms:**
  > *"To the extent possible under law, the creator has waived all copyright and related or neighboring rights to this work. This work is published from the United States. You can copy, modify, distribute and perform the work, even for commercial purposes, all without asking permission."*
- **Reference URL:** [Kaggle MTSamples Dataset](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) (CC0: Public Domain)

## 3. Data Schema & Fields

| Field | Type | Description |
|---|---|---|
| `encounter_id` | String | Unique benchmark tracking ID (e.g. `ENC-0012`) |
| `sample_name` | String | Clinical procedure or encounter type name |
| `medical_specialty_raw` | String | Source medical specialty classification |
| `target_department` | Categorical | Mapped clinical specialist routing destination |
| `clinical_summary` | Text | Clinical narrative of patient presentation and findings |

## 4. Routing Taxonomy for Healthcare & Clinical Agents

The dataset benchmarks triage routing across 4 core specialist hospital departments:

1. **`cardiology` (Cardiovascular / Pulmonary):**
   * *Clinical Scope:* Chest pain, coronary artery disease, acute myocardial infarction, arrhythmias, heart failure, hypertension, valve disease, shortness of breath, pulmonary embolism.
2. **`neurology` (Neurological Sciences):**
   * *Clinical Scope:* Seizures, epilepsy, migraines, neuropathies, stroke/TIA, tremors, Parkinson's disease, multiple sclerosis, cognitive impairment.
3. **`orthopedics` (Musculoskeletal & Spine):**
   * *Clinical Scope:* Bone fractures, joint arthroplasty, arthritis, ligament/tendon tears, disc herniation, back/neck pain, musculoskeletal trauma.
4. **`gastroenterology` (Digestive & Biliary):**
   * *Clinical Scope:* Abdominal pain, GERD, endoscopy/colonoscopy, inflammatory bowel disease, GI bleed, pancreatitis, hepatitis, gallbladder disease.

## 5. HIPAA Compliance & Patient Privacy
- **De-Identification:** All records in this dataset are fully de-identified and contain no Protected Health Information (PHI) under HIPAA Safe Harbor regulations (45 CFR § 164.514).
- No real patient names, social security numbers, medical record numbers (MRNs), phone numbers, or residential addresses are present.
- Safe for open-source AI evaluation, training, and benchmarking.
